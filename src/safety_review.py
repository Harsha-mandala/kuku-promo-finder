"""Build an offline review queue without changing saved judge scores or ranks."""

import argparse
import re
from pathlib import Path

import yaml

from .context import load_episodes
from .editorial_io import editor_source_key, fresh_editor_rows, read, write, write_csv

NEEDS_REVIEW = "needs_human_review"
SAFETY = {"safe": 0, "caution": 1, "unsafe": 2}


def spans(row):
    if row.get("segments"):
        return row["segments"]
    if row.get("final_span"):
        start, end = row["final_span"]
        return [{"episode": row["episode"], "start": start, "end": end}]
    return []


def inherit_trailer_safety(trailer, singles):
    original = trailer.get("judge_ad_safety", trailer["ad_safety"])
    reason = trailer.get("judge_ad_safety_reason", trailer.get("ad_safety_reason", ""))
    verified = trailer.get("judge_ad_safety_verified", trailer.get("ad_safety_verified", original == "safe"))
    level = SAFETY[original]
    sources = []
    for single in singles:
        rating = SAFETY.get(single.get("ad_safety"), 0)
        if rating <= SAFETY[original] or rating < level:
            continue
        if any(a["episode"] == b["episode"] and min(a["end"], b["end"]) > max(a["start"], b["start"])
               for a in spans(trailer) for b in spans(single)):
            if rating > level:
                sources.clear()
                level = rating
            sources.append(single)
    effective = next(label for label, value in SAFETY.items() if value == level)
    source_ids = sorted(row.get("promo_id") or row["id"] for row in sources)
    inherited_reasons = [row.get("ad_safety_reason", "") for row in sources]
    effective_verified = (all(row.get("ad_safety_verified", False) for row in sources)
                          if sources else verified)
    return dict(trailer, judge_ad_safety=original, judge_ad_safety_reason=reason,
                judge_ad_safety_verified=verified, ad_safety=effective, ad_safety_verified=effective_verified,
                ad_safety_reason=" · ".join(filter(None, [reason, *inherited_reasons])),
                ad_safety_display=effective + (" · unverified" if not effective_verified else ""),
                safety_inherited_from=source_ids)


def alternate_pair(left, right):
    if (left.get("type") == "trailer") == (right.get("type") == "trailer"):
        return False
    single, trailer = (right, left) if left.get("type") == "trailer" else (left, right)
    parts, whole = spans(trailer), spans(single)
    if not parts or not whole:
        return False
    span = whole[0]
    if all(p["episode"] == span["episode"] and span["start"] <= p["start"] and p["end"] <= span["end"] for p in parts):
        return True
    covered = span["start"]
    for part in sorted(parts, key=lambda p: p["start"]):
        if part["episode"] == span["episode"] and part["start"] <= covered < part["end"]:
            covered = part["end"]
    return covered >= span["end"]


def shortlist(rows, limit=10):
    """Keep the highest-ranked eligible cut and nest its lower-ranked alternatives."""
    kept = []
    for row in rows:
        parent = next((item for item in kept if alternate_pair(item, row)), None)
        if parent is not None:
            parent["alternate_cuts"].append(row)
        elif len(kept) < limit:
            kept.append(dict(row, alternate_cuts=[]))
    return kept


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
        segments = spans(row)
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
    order = {row["promo_id"]: row for row in fresh_editor_rows(output, tag, parsed)}
    unsafe = read(final / "exclusions.json", {}).get("unsafe", [])
    original_trailers = read(final / "trailers_ranked.json", [])
    trailers = [inherit_trailer_safety(row, singles + unsafe) for row in original_trailers]
    if trailers != original_trailers:
        write(final / "trailers_ranked.json", trailers)
        write_csv(final / "trailers_ranked.csv", trailers)
        if order:
            safety_fields = {"judge_ad_safety", "judge_ad_safety_reason", "judge_ad_safety_verified",
                             "ad_safety", "ad_safety_reason", "ad_safety_verified", "ad_safety_display", "safety_inherited_from"}
            for trailer in trailers:
                if trailer["promo_id"] in order:
                    order[trailer["promo_id"]].update({key: trailer[key] for key in safety_fields})
            write(final / "editor_order.json", list(order.values()))
            write_csv(final / "editor_order.csv", list(order.values()))
            manifest = read(final / "editor_inputs.json", {})
            manifest["source_key"] = editor_source_key(output, tag, parsed)
            write(final / "editor_inputs.json", manifest)
        combined = list(order.values())
        for row in [dict(item, type="single") for item in singles] + trailers:
            if row["promo_id"] not in order:
                combined.append(dict(row, judge_rank=row["rank"], final_rank=len(combined) + 1, editor_rank=None))
        write_csv(final / "promos_all.csv", combined)
    excluded_trailers = read(final / "trailer_exclusions.json", {})
    rejected_trailers = read(output / f"trailers_{tag}" / "validated.json", [])
    trailer_segments = {row["id"]: row["segments"] for row in rejected_trailers}
    unsafe += [dict(row, segments=trailer_segments.get(row["promo_id"], [])) for row in excluded_trailers.get("unsafe", [])]
    episodes = {ep.number: ep for ep in load_episodes(parsed)} if parsed.exists() else {}
    settings = config.get("safety", {})
    queue = review_queue(singles + trailers + unsafe, episodes, settings.get("review_terms", []),
                         settings.get("child_danger_terms", []))
    write(final / "safety_review.json", queue)

    ranked = singles + trailers
    ranked.sort(key=lambda row: order.get(row["promo_id"], {}).get("final_rank", float("inf")))
    queued = {row["promo_id"] for row in queue}
    eligible = [row for row in ranked if row["promo_id"] not in queued and row["ad_safety"] != "unsafe"]
    safe = [row for row in eligible if row["ad_safety"] == "safe"]
    report = report_path.read_text(encoding="utf-8").split("\n## Automatic top 10\n")[0]
    report = report.replace("\n## Top 10\n", "\n## Judge shortlist (score detail)\n")
    report = report.replace("\n## Judge shortlist (before editor)\n", "\n## Judge shortlist (score detail)\n")
    if "\n## Trailers\n" in report:
        main, trailer_report = report.split("\n## Trailers\n", 1)
        report = main + "\n## Trailers\n" + trailer_report.replace(". Safety: ", ". Judge safety: ")
    for title, rows in (("Automatic top 10", shortlist(eligible)), ("Safe-only top 10", shortlist(safe))):
        report += f"\n## {title}\n\n"
        if not order:
            report += "Uses saved judge order. A current editor order is unavailable.\n\n"
        report += "| Slot | Promo | Segment | Duration | Judge rank | Editor rank | Final rank | Safety |\n|---|---|---|---|---|---|---|---|\n"
        for index, row in enumerate(rows, 1):
            editor = order.get(row['promo_id'], {})
            report += (f"| {index} | {row['promo_id']} | {row['segment_text']} | {row['duration_s']} s | "
                       f"{editor.get('judge_rank', row['rank'])} | {editor.get('editor_rank', '—')} | "
                       f"{editor.get('final_rank', '—')} | {row['ad_safety']}"
                       f"{' · inherited from ' + ', '.join(row['safety_inherited_from']) if row.get('safety_inherited_from') else ''} |\n")
            for alternate in row["alternate_cuts"]:
                alternate_ranks = order.get(alternate["promo_id"], {})
                report += (f"|  | {alternate['promo_id']} · alternate cut of {row['promo_id']} | "
                           f"{alternate['segment_text']} | {alternate['duration_s']} s | "
                           f"{alternate_ranks.get('judge_rank', alternate['rank'])} | "
                           f"{alternate_ranks.get('editor_rank', '—')} | {alternate_ranks.get('final_rank', '—')} | "
                           f"{alternate['ad_safety']} |\n")
    report += "\n## Trailer safety\n\n"
    for row in trailers:
        sources = row["safety_inherited_from"]
        report += f"- {row['promo_id']}: {row['ad_safety']} (judge: {row['judge_ad_safety']})"
        report += f" · inherited from {', '.join(sources)}\n" if sources else "\n"
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
