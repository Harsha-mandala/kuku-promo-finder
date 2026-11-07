"""Combine independent judge scores with rules and episode diversity."""

import argparse
import csv
import json
from pathlib import Path

import yaml

from .cliprender import render_clip
from .context import load_episodes
from .judge import SavedJudgeBatch, cap_promos, load_promos, judge_fingerprint, verify_safety
from .llm import step_model_config
from .timeutils import format_timestamp

DEFAULT_WEIGHTS = {"hook": 0.25, "cliffhanger": 0.25, "standalone": 0.20, "emotion": 0.10, "post_priority": 0.20,
                   "judge": 0.80, "rule": 0.15, "early_episode": 0.05}
DEFAULT_PENALTIES = {"flagged": 0.05, "hook_reassigned": 0.03, "cliffhanger_reassigned": 0.03, "caution": 0.03}
LEGACY_WEIGHTS = {"hook": 0.30, "cliffhanger": 0.30, "standalone": 0.25, "emotion": 0.15}


def duration_fit(duration):
    if 25 <= duration <= 45:
        return 1.0
    if 20 <= duration < 25 or 45 < duration <= 55:
        return 0.7
    return 0.4 if 55 < duration <= 60 else 0.0


def source_record(promo):
    record = promo.model_dump()
    record["generator_note"] = record.pop("why_it_works")
    record["why_it_works"] = None
    return record


def rank_promos(promos, scores, config):
    """Return selected top slots, remaining ranked clips, and exclusions."""
    weights = DEFAULT_WEIGHTS | config.get("rank", {}).get("weights", {})
    penalties = DEFAULT_PENALTIES | config.get("rank", {}).get("penalties", {})
    score_map = {score.promo_id: score for score in scores}
    selected, capped = cap_promos([p for p in promos if p.status != "rejected"])
    selected_ids = {p.id for p in selected}
    rows, unsafe, rejected, unjudged = [], [], [], []
    for promo in promos:
        if promo.status == "rejected":
            rejected.append(source_record(promo))
            continue
        if promo.id not in selected_ids:
            continue
        judge = score_map.get(promo.id)
        if judge is None:
            unjudged.append(source_record(promo))
            continue
        if judge.ad_safety == "unsafe":
            unsafe.append({**promo.model_dump(), "generator_note": promo.why_it_works, **judge.model_dump(),
                           'ad_safety_display':'unsafe'+(' · unverified' if not judge.ad_safety_verified else '')})
            continue
        normalized = {name: (getattr(judge, f"{name}_score") - 1) / 4 for name in ("hook", "cliffhanger", "standalone", "emotion")}
        legacy = judge.post_priority is None
        if legacy:
            judge_part = sum(LEGACY_WEIGHTS[name] * value for name, value in normalized.items())
        else:
            normalized['post_priority'] = (judge.post_priority - 1) / 9
            judge_part = sum(weights[name] * value for name, value in normalized.items())
        rule_part = duration_fit(promo.duration_s) * (1 if promo.start_clean and promo.end_clean else 0.8)
        early = max(0, 1 - (promo.episode - 1) / 30)
        flags = set(promo.flags) | set(promo.fix_log)
        deduction = (penalties["flagged"] if promo.status == "flagged" else 0)
        deduction += sum(penalties[name] for name in ("hook_reassigned", "cliffhanger_reassigned") if name in flags)
        deduction += penalties["caution"] if judge.ad_safety == "caution" else 0
        final = weights["judge"] * judge_part + weights["rule"] * rule_part + weights["early_episode"] * early - deduction
        row = {"rank": 0, "promo_id": promo.id, "episode": promo.episode,
               "segments": [{"episode": promo.episode, "start": promo.final_span[0], "end": promo.final_span[1]}],
               "segment_text": f"E{promo.episode:02d} {format_timestamp(promo.final_span[0])}-{format_timestamp(promo.final_span[1])}",
               "duration_s": promo.duration_s, "hook_text": promo.hook_text, "cliffhanger_text": promo.cliffhanger_text,
               "generator_note": promo.why_it_works, **judge.model_dump(), "normalized_scores": normalized,
               "ad_safety_display": judge.ad_safety + (" · unverified" if not judge.ad_safety_verified else ""),
               "formula_note": "old formula: post_priority missing" if legacy else "post_priority formula",
               "judge_part": round(judge_part, 6), "rule_part": round(rule_part, 6), "early_episode_bonus": round(early, 6),
               "penalty": round(deduction, 6), "final_score": round(final, 6), "status": promo.status,
               "fix_log": promo.fix_log, "why_post_first": ""}
        rows.append(row)
    same_episode_penalty = config.get("rank", {}).get("same_episode_penalty", 0.05)
    if same_episode_penalty < 0:
        raise ValueError("same_episode_penalty must be non-negative")
    ordered, counts = [], {}
    while rows:
        for row in rows:
            row["score_adj"] = round(row["final_score"] - same_episode_penalty * counts.get(row["episode"], 0), 6)
        row = min(rows, key=lambda item: (-item["score_adj"], -item["cliffhanger_score"], item["duration_s"], item["promo_id"]))
        rows.remove(row)
        row["rank"] = len(ordered) + 1
        ordered.append(row)
        counts[row["episode"]] = counts.get(row["episode"], 0) + 1
    for row in ordered[:5]:
        strongest = sorted(("hook", "cliffhanger", "standalone", "emotion"), key=lambda name: -row[f"{name}_score"])[:2]
        reasons = ", ".join(f"{name} {row[name + '_score']}/5" for name in strongest)
        row["why_post_first"] = f"Strongest dimensions: {reasons}. {row['one_line_pitch']}"
    return {"ranked": ordered, "top": ordered[:10], "unsafe": unsafe, "rejected": rejected, "unjudged": unjudged,
            "capped": [source_record(p) for p in capped]}


def write_ranking(config, tag, *, episode_numbers=None, legacy_prompt=None):
    output = Path(config.get("output_dir", "outputs"))
    promos = load_promos(output / f"validated_{tag}", episode_numbers)
    score_path = output / f"judged_{tag}/scores.json"
    try:
        scores = SavedJudgeBatch.model_validate_json(score_path.read_text(encoding="utf-8")).root if score_path.exists() else []
    except ValueError:
        scores = []
    episodes = {ep.number: ep for ep in load_episodes(config.get("parsed_dir", "parsed"))}
    try:
        inputs = json.loads((output / f"judged_{tag}/score_inputs.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        inputs = {}
    provider, model = step_model_config(config, "judge")
    current = {p.id: judge_fingerprint(episodes[p.episode], p, provider, model, config) for p in promos if p.status != "rejected"}
    by_id = {p.id: p for p in promos}
    selected_scores = []
    old_system = Path(legacy_prompt).read_text(encoding='utf-8') if legacy_prompt else None
    for s in scores:
        promo = by_id.get(s.promo_id)
        if promo is None:
            continue
        expected = current.get(s.promo_id)
        if old_system is not None and s.post_priority is None:
            expected = judge_fingerprint(episodes[promo.episode], promo, provider, model, config, system=old_system)
        if inputs.get(s.promo_id) == expected:
            selected_scores.append(verify_safety(s, episodes[promo.episode], promo))
    scores = selected_scores
    result = rank_promos(promos, scores, config)
    by_id = {p.id: p for p in promos}
    directory = output / f"final_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "promos_ranked.json").write_text(json.dumps(result["ranked"], ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["rank", "promo_id", "episode", "segments", "segment_text", "duration_s", "hook_text", "cliffhanger_text", "why_it_works", "generator_note", "one_line_pitch",
              "hook_score", "cliffhanger_score", "standalone_score", "emotion_score", "judge_part", "rule_part", "early_episode_bonus", "penalty", "final_score",
              "ad_safety", "ad_safety_reason", "ad_safety_verified", "ad_safety_display", "post_priority", "formula_note", "weakest_point", "status", "fix_log", "why_post_first", "normalized_scores", "score_adj"]
    with (directory / "promos_ranked.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in result["ranked"]:
            writer.writerow(row | {"segments": json.dumps(row["segments"]), "fix_log": " | ".join(row["fix_log"])})
    report = [f"# Promo ranking — {tag}", "", f"{len(result['ranked'])} ranked; {len(result['unsafe'])} unsafe excluded; {len(result['rejected'])} rejected; {len(result['unjudged'])} unjudged.",
              "", "## Top 10", "", f"Greedy soft diversity penalty: {config.get('rank', {}).get('same_episode_penalty', 0.05)} per previously placed promo from the same episode. Ranks are consecutive.", ""]
    stats_path = output / f"judged_{tag}/run_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        note = f"Judge run: {stats.get('api_calls', 0)} provider attempts; stop={stats.get('stopped')}; {len(stats.get('pending', []))} pending."
        error_path = stats_path.parent / "errors.log"
        if error_path.exists() and error_path.read_text(encoding="utf-8").strip():
            note += " First error: " + error_path.read_text(encoding="utf-8").splitlines()[0][:240]
        report[4:4] = [note, ""]
    for row in result["top"]:
        report.extend([f"### {row['rank']}. {row['promo_id']} — {row['segment_text']}", "",
                       f"Duration: {row['duration_s']} s | final_score: {row['final_score']:.4f} | score_adj: {row['score_adj']:.4f} | status: {row['status']} | ad_safety: {row['ad_safety_display']}", "",
                       f"Formula: {row['formula_note']} | post priority: {row['post_priority'] if row['post_priority'] is not None else 'not scored'}", "",
                       f"Hook: {row['hook_text']}", "", f"Cliffhanger: {row['cliffhanger_text']}", "",
                       f"Why it works: {row['why_it_works']}", "",
                       f"Why post first: {row['why_post_first'] or row['one_line_pitch']}", "",
                       f"Scores: hook {row['hook_score']}/5; cliffhanger {row['cliffhanger_score']}/5; standalone {row['standalone_score']}/5; emotion {row['emotion_score']}/5.", "",
                       f"Weakest point: {row['weakest_point']}", "", "```text",
                       render_clip(episodes[row['episode']], by_id[row['promo_id']], mark_lines=True), "```", ""])
    report.extend(["## All other ranked promos", "", "| Rank | Promo | Segment | Score |", "|---|---|---|---|"])
    for row in result["ranked"]:
        if row not in result["top"]:
            report.append(f"| {row['rank']} | {row['promo_id']} | {row['segment_text']} | {row['final_score']:.4f} |")
    for label, items in [("Excluded unsafe promos", result["unsafe"]), ("Rejected promos", result["rejected"]), ("Unjudged promos", result["unjudged"]), ("Not judged (cap)", result["capped"])]:
        report.extend(["", f"## {label}", ""])
        for row in items:
            if label == "Not judged (cap)":
                reason = "not judged (cap): at most three highest-self_score promos per episode"
            elif label == "Unjudged promos":
                reason = "No current judge score available"
                if row["fix_log"]:
                    reason += "; validator notes: " + " | ".join(row["fix_log"])
            else:
                reason = row.get("ad_safety_reason") or " | ".join(row["fix_log"])
                if row.get('ad_safety_verified') is False:
                    reason = 'unverified — '+reason
            report.append(f"- {row['id']}: {reason}")
        if not items:
            report.append("None.")
    (directory / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (directory / "exclusions.json").write_text(json.dumps({key: result[key] for key in ("unsafe", "rejected", "unjudged", "capped")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RANK: {len(result['ranked'])} ranked | {len(result['unsafe'])} unsafe | {len(result['unjudged'])} unjudged | {directory / 'report.md'}")
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tag", required=True)
    cli.add_argument("--config", type=Path, default=Path("config.yaml"))
    cli.add_argument("--legacy-prompt", type=Path, help="Archived prompt for audited offline re-ranking of legacy scores")
    args = cli.parse_args()
    try:
        from .safety_review import write_safety_review
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        write_ranking(config, args.tag, legacy_prompt=args.legacy_prompt)
        write_safety_review(config, args.tag)
    except (OSError, ValueError) as error:
        cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
