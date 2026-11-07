"""Build an offline review queue without changing saved judge scores or ranks."""

import argparse
import re
from pathlib import Path

import yaml

from .context import load_episodes
from .editorial_io import fresh_editor_rows, read, write

NEEDS_REVIEW = "needs_human_review"


def review_queue(rows, episodes, terms, child_danger_terms=()):
    if not terms:
        return []
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(term) for term in terms) + r")(?!\w)", re.I)
    danger = "|".join(re.escape(term) for term in child_danger_terms)
    child = r"\b(?:child|children)\b"
    danger_pattern = re.compile(rf"{child}.{{0,65}}\b(?:{danger})\b|\b(?:{danger})\b.{{0,65}}{child}", re.I) if danger else None
    queue = []
    for row in rows:
        if row.get("ad_safety") not in {"caution", "unsafe"}:
            continue
        segments = row.get("segments", [])
        if not segments and row.get("final_span"):
            start, end = row["final_span"]
            segments = [{"episode": row["episode"], "start": start, "end": end}]
        text = [row.get("ad_safety_reason", "")]
        for span in segments:
            episode = episodes.get(span["episode"])
            if episode is None:
                continue
            for shot in episode.shots:
                if shot.effective_start >= span["start"] - .001 and shot.effective_end <= span["end"] + .001:
                    text.append(shot.camera_text)
                    text.extend(line.text for line in shot.lines)
        found = {match.group().lower() for item in text for match in pattern.finditer(item)}
        if danger_pattern and any(danger_pattern.search(item) for item in text):
            found.add("child-in-danger context")
        if found:
            queue.append({"promo_id": row.get("promo_id") or row["id"], "safety_review": NEEDS_REVIEW,
                          "ad_safety": row["ad_safety"], "reason": row.get("ad_safety_reason", ""), "matched_terms": sorted(found)})
    return queue


def write_safety_review(config, tag):
    output = Path(config.get("output_dir", "outputs"))
    parsed = Path(config.get("parsed_dir", "parsed"))
    final = output / f"final_{tag}"
    report_path = final / "report.md"
    if not report_path.exists():
        return []
    singles = read(final / "promos_ranked.json", [])
    trailers = read(final / "trailers_ranked.json", [])
    unsafe = read(final / "exclusions.json", {}).get("unsafe", [])
    excluded_trailers = read(final / "trailer_exclusions.json", {})
    rejected_trailers = read(output / f"trailers_{tag}" / "validated.json", [])
    trailer_segments = {row["id"]: row["segments"] for row in rejected_trailers}
    unsafe += [dict(row, segments=trailer_segments.get(row["promo_id"], [])) for row in excluded_trailers.get("unsafe", [])]
    episodes = {ep.number: ep for ep in load_episodes(parsed)} if parsed.exists() else {}
    settings = config.get("safety", {})
    queue = review_queue(singles + trailers + unsafe, episodes, settings.get("review_terms", []),
                         settings.get("child_danger_terms", []))
    write(final / "safety_review.json", queue)

    order = {row["promo_id"]: row for row in fresh_editor_rows(output, tag, parsed)}
    ranked = singles + trailers
    ranked.sort(key=lambda row: order.get(row["promo_id"], {}).get("final_rank", float("inf")))
    queued = {row["promo_id"] for row in queue}
    eligible = [row for row in ranked if row["promo_id"] not in queued and row["ad_safety"] != "unsafe"]
    safe = [row for row in eligible if row["ad_safety"] == "safe"]
    report = report_path.read_text(encoding="utf-8").split("\n## Automatic top 10\n")[0]
    report = report.replace("\n## Top 10\n", "\n## Judge shortlist (score detail)\n")
    report = report.replace("\n## Judge shortlist (before editor)\n", "\n## Judge shortlist (score detail)\n")
    for title, rows in (("Automatic top 10", eligible), ("Safe-only top 10", safe)):
        report += f"\n## {title}\n\n"
        if not order:
            report += "Uses saved judge order. A current editor order is unavailable.\n\n"
        report += "| Slot | Promo | Segment | Duration | Judge rank | Editor rank | Final rank | Safety |\n|---|---|---|---|---|---|---|---|\n"
        for index, row in enumerate(rows[:10], 1):
            editor = order.get(row['promo_id'], {})
            report += (f"| {index} | {row['promo_id']} | {row['segment_text']} | {row['duration_s']} s | "
                       f"{editor.get('judge_rank', row['rank'])} | {editor.get('editor_rank', '—')} | "
                       f"{editor.get('final_rank', '—')} | {row['ad_safety']} |\n")
    report += "\n## Needs safety review\n\n"
    for row in queue:
        report += f"- {row['promo_id']} ({row['ad_safety']}) · {row['reason']} · matched {', '.join(row['matched_terms'])}\n"
    report_path.write_text(report + ("None.\n" if not queue else ""), encoding="utf-8")
    return queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    queue = write_safety_review(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.tag)
    print(f"Safety review queue: {len(queue)} promos")
