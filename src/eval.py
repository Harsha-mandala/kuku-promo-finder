"""Evaluate saved promo outputs and export blind sheets without API calls."""

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean

import yaml

from .baselines import DEFAULT_RULES, build_baseline
from .cliprender import render_clip
from .eval_cost import cost_report
from .eval_metrics import average_ranks, evaluate_system, matches, overlap_fraction, pairwise_agreement, percentile, spearman
from .eval_strict import strict_evaluation
from .editorial_io import fresh_editor_rows
from .models import Episode
from .timeutils import format_timestamp, parse_timestamp
from .validate import ValidatedPromo

SYSTEMS = ("random", "rules_only", "generator_self_score", "full_pipeline")
SCORE_FIELDS = ("hook_score", "cliffhanger_score", "standalone_score", "emotion_score", "post_priority")


def read_json(path, fallback, warnings):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        warnings.append(f"{path}: {error}")
        return fallback


def load_saved(config, tag):
    """Read only saved files for this tag; malformed files become warnings."""
    output = Path(config.get("output_dir", "outputs"))
    warnings, episodes, candidates, available, promos = [], {}, [], [], []
    for path in sorted(Path(config.get("parsed_dir", "parsed")).glob("episode_*.json")):
        try:
            episode = Episode.model_validate_json(path.read_text(encoding="utf-8"))
            episodes[episode.number] = episode
        except (ValueError, OSError) as error:
            warnings.append(f"{path}: {error}")
    for path in sorted((output / f"candidates_{tag}").glob("episode_*.json")):
        raw = read_json(path, None, warnings)
        try:
            number = int(path.stem.split("_")[-1])
            if not isinstance(raw, list) or number not in episodes:
                raise ValueError("Expected candidate list and a parsed episode")
            available.append(number)
            for i, item in enumerate(raw, 1):
                span = (float(item["span_start"]), float(item["span_end"]))
                if not all(math.isfinite(v) for v in span) or span[0] >= span[1]:
                    raise ValueError("Invalid candidate span")
                candidates.append(dict(item, episode=number, span=span,
                    id=item.get("candidate_id", f"E{number:03d}_C{i:02d}")))
        except (ValueError, TypeError, KeyError) as error:
            warnings.append(f"{path}: {error}")
    for path in sorted((output / f"validated_{tag}").glob("episode_*.json")):
        for item in read_json(path, [], warnings):
            try:
                promos.append(ValidatedPromo.model_validate(item))
            except ValueError as error:
                warnings.append(f"{path}: {error}")
    ranked = read_json(output / f"final_{tag}/promos_ranked.json", [], warnings)
    editor_rows = fresh_editor_rows(output,tag,Path(config.get('parsed_dir','parsed')))
    edits = {r['promo_id']:r for r in editor_rows if r.get('type')=='single'}
    remaining=0
    for row in ranked:
        row['judge_rank'] = row['rank']
        if row['promo_id'] in edits:
            row['rank'] = edits[row['promo_id']]['final_rank']
        elif edits:
            remaining+=1
            row['rank'] = len(editor_rows)+remaining
    scores = read_json(output / f"judged_{tag}/scores.json", [], warnings)
    exclusions = read_json(output / f"final_{tag}/exclusions.json", {}, warnings)
    return dict(output=output, episodes=episodes, candidates=candidates, available=available,
                promos=promos, ranked=ranked, scores=scores, exclusions=exclusions, warnings=warnings,editor_rows=editor_rows)


def load_gold(path, episodes):
    valid, warnings, declared, count = [], [], set(), 0
    if not path.exists():
        return {"rows_found": 0, "rows_valid": 0, "rows": [], "declared_episodes": [],
                "warnings": [f"Gold file not found: {path}"]}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"episode", "start", "end", "hook", "why"}.issubset(reader.fieldnames or []):
            warnings.append("gold.csv must have episode,start,end,hook,why columns")
        for row_number, row in enumerate(reader, 2):
            count += 1
            try:
                number = int(row["episode"])
                declared.add(number)
                start, end = parse_timestamp(row["start"]), parse_timestamp(row["end"])
                if number not in episodes:
                    raise ValueError("No parsed episode available")
                if not start < end or not 20 <= end-start <= 60:
                    raise ValueError("Gold clip must have start < end and duration 20–60 s")
                if end > episodes[number].duration:
                    raise ValueError("Gold clip is outside episode duration")
                valid.append(dict(row, episode=number, span=(start, end), row_number=row_number))
            except (ValueError, KeyError, TypeError) as error:
                warnings.append(f"gold.csv row {row_number}: {error}")
    return {"rows_found": count, "rows_valid": len(valid), "rows": valid,
            "declared_episodes": sorted(declared), "warnings": warnings}


def promo_record(promo):
    return {"id": promo.id, "episode": promo.episode, "span": promo.final_span,
            "self_score": promo.self_score}


def system_inputs(saved):
    """Use original proposals for self-score; actual final spans for the pipeline."""
    generator = sorted(saved["candidates"], key=lambda p: (p["episode"], -p.get("self_score", 0), p["id"]))
    full = []
    for row in sorted(saved["ranked"], key=lambda r: r["rank"]):
        for segment in row["segments"]:
            full.append({"id": row["promo_id"], "episode": segment["episode"],
                         "span": (segment["start"], segment["end"])})
    capped_ids = {p["id"] for p in saved["exclusions"].get("capped", [])}
    # Older bundles may omit exclusions.json: compute the documented cap deterministically.
    for number in saved["available"]:
        eligible = sorted((p for p in saved["promos"] if p.episode == number and p.status != "rejected"),
                          key=lambda p: (-p.self_score, p.id))
        capped_ids.update(p.id for p in eligible[3:])
    existing = {p["id"] for p in full}
    unsafe = {s["promo_id"] for s in saved["scores"] if s.get("ad_safety") == "unsafe"}
    capped = sorted((p for p in saved["promos"] if p.id in capped_ids and p.id not in existing
                     and p.id not in unsafe and p.status != "rejected"), key=lambda p: (-p.self_score, p.id))
    full.extend(dict(promo_record(p), recall_only=True) for p in capped)
    return {"generator_self_score": generator, "full_pipeline": full}


def pipeline_health(saved, config, tag):
    candidates, promos, scores = saved["candidates"], saved["promos"], saved["scores"]
    n = len(candidates)
    def compliance(check):
        count = sum(check(p) for p in candidates)
        return {"count": count, "n": n, "share": count/n if n else None}
    vectors = Counter(tuple(score.get(f) for f in SCORE_FIELDS) for score in scores)
    providers = config.get("models", {})
    candidate_provider = providers.get("candidates", {}).get("provider", "gemini")
    judge_provider = providers.get("judge", {}).get("provider", "groq")
    # Historical groq is the explicitly named provider comparison tag.
    if tag == "groq":
        candidate_provider = "groq"
    return {
        "n_episodes": len(saved["available"]), "episodes": saved["available"], "n_candidates": n,
        "before_validator": {
            "duration_20_60": compliance(lambda p: 20 <= p["span"][1]-p["span"][0] <= 60),
            "hook_within_3s": compliance(lambda p: 0 <= p.get("hook_offset_s", math.inf) <= 3),
            "clean_start": compliance(lambda p: p.get("start_clean", False)),
            "clean_end": compliance(lambda p: p.get("end_clean", False)),
            "both_clean": compliance(lambda p: p.get("start_clean", False) and p.get("end_clean", False)),
            "all": compliance(lambda p: 20 <= p["span"][1]-p["span"][0] <= 60
                and 0 <= p.get("hook_offset_s", math.inf) <= 3 and p.get("start_clean", False) and p.get("end_clean", False)),
        },
        "validator": {"n": len(promos), "outcomes": {k:sum(p.status == k for p in promos) for k in ("ok", "fixed", "flagged", "rejected")},
            "fix_types": dict(Counter(reason for p in promos for reason in p.fix_log)),
            "flag_counts": dict(Counter(flag for p in promos for flag in p.flags)),
            "hook_reassigned_share": sum("hook_reassigned" in p.flags or "hook_reassigned" in p.fix_log for p in promos)/len(promos) if promos else None},
        "judge": {"n": len(scores), "distributions": {f:dict(sorted(Counter(s[f] for s in scores if s.get(f) is not None).items())) for f in SCORE_FIELDS},
            "missing_post_priority": sum(s.get("post_priority") is None for s in scores),
            "identical_score_vector_tie_share": sum(v for v in vectors.values() if v > 1)/len(scores) if scores else None,
            "ad_safety": dict(Counter(s["ad_safety"] for s in scores)),
            "unverified": sum(not s.get("ad_safety_verified", True) for s in scores)},
        "cost": cost_report(saved["output"], tag, set(saved["available"]), candidate_provider, judge_provider),
    }


def agreement(pairs):
    a, b = zip(*pairs) if pairs else ([], [])
    result = {"n": len(pairs), "spearman": spearman(a, b), "pairwise_agreement": pairwise_agreement(a, b)}
    draws = {"spearman": [], "pairwise": []}
    if len(pairs) > 1:
        rng = random.Random(20261106)
        for _ in range(2000):
            x, y = zip(*rng.choices(pairs, k=len(pairs)))
            rho = spearman(x, y)
            share = pairwise_agreement(x, y)["share"]
            if rho is not None:
                draws["spearman"].append(rho)
            if share is not None:
                draws["pairwise"].append(share)
    result["ci95"] = {key: [percentile(values, .025), percentile(values, .975)] if values else None
                      for key, values in draws.items()}
    return result


def human_agreement(path, saved):
    ranked = {r["promo_id"]: r for r in saved["ranked"]}
    scores = {s["promo_id"]: s for s in saved["scores"]}
    edits = {row['promo_id']: row for row in saved.get('editor_rows', [])}
    known = {p.id for p in saved["promos"]} | set(edits)
    for key, row in edits.items():
        ranked[key] = dict(row, rank=row["final_rank"])
        if row.get("type") == "trailer":
            scores[key] = row
    ratings, warnings = {}, []
    paths = (path,) if isinstance(path, Path) else path
    for sheet in paths:
        if not sheet.exists():
            continue
        with sheet.open(encoding="utf-8-sig", newline="") as stream:
            for number, row in enumerate(csv.DictReader(stream), 2):
                if not row.get("my_rating", "").strip():
                    continue
                try:
                    rating = int(row["my_rating"])
                    key = row["promo_id"]
                    if not 1 <= rating <= 5 or key not in known:
                        raise ValueError("Expected known promo_id and integer rating 1–5")
                    if key in ratings:
                        raise ValueError("Duplicate rated promo_id")
                    ratings[key] = rating
                except (ValueError, KeyError, TypeError) as error:
                    warnings.append(f"{sheet.name} rating row {number}: {error}")
    result = {"n_rated": len(ratings), "warnings": warnings, "ratings": ratings, "biggest_disagreements": []}
    for field, source in (("final_score", ranked), ("post_priority", scores)):
        pairs = [(rating, source[key][field]) for key, rating in ratings.items()
                 if key in source and source[key].get(field) is not None]
        result[field] = agreement(pairs)
    result['order_disagreements'] = {}
    orders = {
        'judge_order': {key: (edits[key]['judge_rank'] if key in edits else
                             len(edits) + row.get('judge_rank', row['rank'])) for key, row in ranked.items()},
        'editor_order': {key: row['editor_rank'] for key, row in edits.items() if row.get('editor_rank') is not None},
        'final_order': {key: row['rank'] for key, row in ranked.items()},
    }
    for label, ranks in orders.items():
        keys = [key for key in ratings if key in ranks]
        result[label] = agreement([(ratings[key], -ranks[key]) for key in keys])
        if label == 'final_order':
            continue
        human_positions = average_ranks([-ratings[key] for key in keys])
        model_positions = average_ranks([ranks[key] for key in keys])
        gaps = []
        for key, human_position, model_position in zip(keys, human_positions, model_positions):
            gaps.append({"promo_id": key, "my_rating": ratings[key], "judge_rank": ranked[key].get('judge_rank', ranked[key]['rank']),
                         "editor_rank": orders['editor_order'].get(key), "rank_gap": abs(human_position - model_position),
                         "weakest_point": scores.get(key, {}).get('weakest_point'),
                         "editor_reason": edits.get(key, {}).get('editor_reason')})
        result['order_disagreements'][label] = sorted(gaps, key=lambda row: (-row['rank_gap'], row['promo_id']))[:3]
    result['top20'] = {}
    for label, top in (('top_20', True), ('rest', False)):
        values = [rating for key, rating in ratings.items()
                  if key in ranked and (ranked[key]['rank'] <= 20) == top]
        result['top20'][label] = {'n': len(values), 'mean': mean(values) if values else None}
    disagreements = []
    for key, rating in ratings.items():
        score, rank = scores.get(key, {}), ranked.get(key, {})
        expected = rank.get("final_score")
        if expected is None and score.get("post_priority") is not None:
            expected = (score["post_priority"] - 1) / 9
        if expected is not None:
            disagreements.append({"promo_id": key, "my_rating": rating, "final_score": rank.get("final_score"),
                "judge_scores": {f:score.get(f) for f in SCORE_FIELDS}, "weakest_point": score.get("weakest_point"),
                "normalized_gap": abs((rating-1)/4 - expected)})
    result["biggest_disagreements"] = sorted(disagreements, key=lambda p: (-p["normalized_gap"], p["promo_id"]))[:3]
    return result


def model_comparison(config, gold, warnings):
    output = Path(config.get("output_dir", "outputs"))
    if not all((output/f"candidates_{tag}").exists() for tag in ("gemini_full", "groq")):
        return []
    bundles = {tag:load_saved(config, tag) for tag in ("gemini_full", "groq")}
    common = set(bundles["gemini_full"]["available"]) & set(bundles["groq"]["available"])
    result = []
    for number in sorted(common):
        targets = [g for g in gold if g["episode"] == number]
        for tag, saved in bundles.items():
            candidates = [p for p in saved["candidates"] if p["episode"] == number]
            score_ids = {p.id for p in saved["promos"] if p.episode == number}
            scores = [s for s in saved["scores"] if s["promo_id"] in score_ids]
            compliant = sum(20 <= p["span"][1]-p["span"][0] <= 60 and 0 <= p.get("hook_offset_s", math.inf) <= 3
                            and p.get("start_clean", False) and p.get("end_clean", False) for p in candidates)
            result.append({"episode": number, "tag": tag, "candidates": len(candidates),
                "all_compliance_share": compliant/len(candidates) if candidates else None, "n_gold": len(targets),
                "candidate_ids_matching_gold": [p["id"] for p in candidates if any(matches(p, g, rule="iou") for g in targets)] if targets else None,
                "judge_mean_scores": {f:mean(s[f] for s in scores if s.get(f) is not None)
                    if any(s.get(f) is not None for s in scores) else None for f in SCORE_FIELDS}})
    return result


def export_rating_sheet(config, tag, n=20, *, eval_dir=Path("eval"), gold_path=None, force=False, top=False):
    saved = load_saved(config, tag)
    settings = config.get("evaluation", {})
    gold = load_gold(Path(gold_path or settings.get("gold_csv", "eval/gold.csv")), saved["episodes"])
    excluded = set(settings.get("gold_episodes", [])) | set(gold["declared_episodes"])
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / f"to_rate_{'top_' if top else ''}{tag}.csv"
    promos = {p.id: p for p in saved["promos"] if p.status != "rejected"}
    if top:
        from .trailers import render_segment
        prior = eval_dir / f"to_rate_{tag}.csv"
        with prior.open(encoding="utf-8-sig", newline="") as stream:
            prior_ids = {row["promo_id"] for row in csv.DictReader(stream)}
        pool = [row for row in saved["editor_rows"] if row["promo_id"] not in prior_ids
                and not any(segment["episode"] in excluded for segment in row["segments"])]
        pool.sort(key=lambda row: row["promo_id"])
        random.Random(settings.get("seed", 20261106)).shuffle(pool)
        if len(pool) < n:
            raise ValueError(f"Only {len(pool)} eligible clips in the saved editor pool for a sheet of {n}")
        chosen = []
        for row in pool[:n]:
            key = row["promo_id"]
            if key in promos:
                promo = promos[key]
                script = render_clip(saved["episodes"][promo.episode], promo)
            else:
                script = "\n--- CUT ---\n".join(render_segment(saved["episodes"][s["episode"]],
                    s["start_shot_id"], s["end_shot_id"], include_final_sound=True) for s in row["segments"])
            episodes = ", ".join(dict.fromkeys(str(s["episode"]) for s in row["segments"]))
            chosen.append(dict(promo_id=key, episode=episodes, span=row["segment_text"], duration=row["duration_s"], clip_script=script))
    else:
        pool = sorted((p for p in promos.values() if p.episode not in excluded), key=lambda p:p.id)
        random.Random(settings.get("seed", 20261106)).shuffle(pool)
        chosen = [dict(promo_id=p.id, episode=p.episode,
                       span=f"{format_timestamp(p.final_span[0])}–{format_timestamp(p.final_span[1])}",
                       duration=p.duration_s, clip_script=render_clip(saved["episodes"][p.episode], p)) for p in pool[:n]]
    if path.exists() and not force:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            if any(row.get("my_rating", "").strip() for row in csv.DictReader(stream)):
                raise ValueError("Existing rating sheet has human ratings; use --force only to intentionally replace them")
    fields = ["promo_id", "episode", "span", "duration", "clip_script", "my_rating"]
    markdown = [f"# Blind rating sheet — {tag}", "", "Rate 1–5: would you post this? No scores, ranks or generator rationales are shown.", ""]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in chosen:
            writer.writerow(dict(row, my_rating=""))
            markdown.extend([f"## {row['promo_id']} | episode {row['episode']} | {row['span']} | {row['duration']:.2f} s",
                             "", "```text", row["clip_script"], "```", "", "my_rating: ____", ""])
    path.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"BLIND SHEET: {path} | {len(chosen)}/{n} promos | excluded gold episodes {sorted(excluded)}")
    return {"path": str(path), "n": len(chosen), "promo_ids": [row["promo_id"] for row in chosen], "excluded_episodes": sorted(excluded)}


def failure_gallery(saved, gold, full, human, eval_dir, target, tag):
    rows = []
    for g in gold:
        choices = [p for p in full if p["episode"] == g["episode"]][:3]
        if any(matches(p, g, rule="iou") for p in choices):
            continue
        all_choices = [p for p in full if p["episode"] == g["episode"]]
        nearest = min(all_choices, key=lambda p:(-overlap_fraction(p["span"], g["span"]), abs(p["span"][0]-g["span"][0]), p["id"])) if all_choices else None
        rows.append({"failure_id": f"miss_gold_row_{g['row_number']}", "kind": "missed_gold", "episode": g["episode"],
            "promo_id": nearest["id"] if nearest else "", "gold_span": g["span"],
            "tool_span": nearest["span"] if nearest else "", "detail": g.get("hook", "")})
    ranks = {p["promo_id"]:p for p in saved["ranked"]}
    for key, rating in human["ratings"].items():
        if rating <= 2 and key in ranks and ranks[key]["rank"] <= 15:
            p = ranks[key]
            rows.append({"failure_id": f"human_{key}", "kind": "top_ranked_low_human_rating", "episode": p["episode"],
                "promo_id": key, "tool_span": p["segments"], "detail": f"rank {p['rank']}, human rating {rating}"})
    for p in saved["promos"]:
        if p.status == "rejected":
            rows.append({"failure_id": f"rejected_{p.id}", "kind": "validator_rejection", "episode": p.episode,
                "promo_id": p.id, "tool_span": p.final_span, "detail": " | ".join(p.fix_log)})
    by_id = {p.id:p for p in saved["promos"]}
    for score in saved["scores"]:
        if score["ad_safety"] in ("unsafe", "caution"):
            p = by_id.get(score["promo_id"])
            rows.append({"failure_id": f"safety_{score['promo_id']}", "kind": score["ad_safety"],
                "episode": p.episode if p else "", "promo_id": score["promo_id"],
                "tool_span": p.final_span if p else "", "detail": score.get("ad_safety_reason", "")})
    previous = {}
    destinations = [Path(eval_dir)/f"failures_{tag}.csv", target/f"failures_{tag}.csv"]
    for path in reversed(destinations):
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                previous.update({r["failure_id"]:r.get("my_reason", "") for r in csv.DictReader(stream)})
    for row in rows:
        row["my_reason"] = previous.get(row["failure_id"], "")
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["failure_id", "kind", "episode", "promo_id", "gold_span", "tool_span", "detail", "my_reason"])
            writer.writeheader()
            writer.writerows(rows)
    return rows


def percent(value):
    return "n/a" if value is None else f"{100*value:.1f}%"


def ci_text(ci):
    return "n/a" if ci is None else f"[{percent(ci[0])}, {percent(ci[1])}]"


def markdown_table(rows):
    if not rows:
        return "No rows available."
    fields = list(rows[0])
    def cell(value):
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(f, "")) for f in fields) + " |" for row in rows)
    return "\n".join(lines)


def comparison_rows(metrics, threshold="0.5"):
    rows = []
    for system, values in metrics["systems"].items():
        m = values[threshold]
        rows.append({"System": system, "n episodes": m["n_episodes"], "n gold": m["n_gold"],
            "Hit@1": percent(m["hit_at_1"])+(" ± "+percent(m["seed_sd"]["hit_at_1"]) if m.get("seed_sd",{}).get("hit_at_1") is not None else ""), "Hit CI95": ci_text(m["ci95"]["hit_at_1"]),
            "Recall@3": percent(m["recall_at_3"])+(" ± "+percent(m["seed_sd"]["recall_at_3"]) if m.get("seed_sd",{}).get("recall_at_3") is not None else ""), "Recall CI95": ci_text(m["ci95"]["recall_at_3"]),
            "Start error (s)": f"{m['mean_start_error_s']:.2f}" if m["mean_start_error_s"] is not None else "n/a",
            "Start ±3s": percent(m["start_within_3s_share"]), "Matched pairs": m["n_matched_pairs"]})
    return rows


def primary_comparison_rows(metrics):
    rows = []
    for system in SYSTEMS:
        values = metrics["primary"][system]
        hook,iou = values["hook"],values["iou"]
        def value(result,field):
            text = percent(result[field])
            if result.get("seed_sd",{}).get(field) is not None:
                text += " ± "+percent(result["seed_sd"][field])
            return text
        def chance(rule):
            c = metrics["chance"][rule]
            return f"{percent(c['hit_chance'])} / {percent(c['recall_chance'])}"
        rows.append({"System":system,"Hook-Hit@1":value(hook,"hit_at_1"),"Hook-Recall@3":value(hook,"recall_at_3"),
            "IoU-Hit@1":value(iou,"hit_at_1"),"IoU-Recall@3":value(iou,"recall_at_3"),
            "Start error (s)":(f"{iou['mean_start_error_s']:.2f} ± {iou['seed_sd']['mean_start_error_s']:.2f}"
                                if 'seed_sd' in iou and iou['mean_start_error_s'] is not None else
                                f"{iou['mean_start_error_s']:.2f}" if iou["mean_start_error_s"] is not None else "n/a"),
            "Chance hook H/R":chance("hook"),"Chance IoU H/R":chance("iou"),"n episodes":iou["n_episodes"],
            "n gold":iou["n_gold"],"Coverage top3":percent(values["coverage"])+(" ± "+percent(values["coverage_sd"]) if values.get("coverage_sd") is not None else ""),
            "Hook CI95 Hit / Recall":ci_text(hook["ci95"]["hit_at_1"])+" / "+ci_text(hook["ci95"]["recall_at_3"]),
            "IoU CI95 Hit / Recall":ci_text(iou["ci95"]["hit_at_1"])+" / "+ci_text(iou["ci95"]["recall_at_3"])})
    return rows


def write_report(metrics, target):
    health, coverage = metrics["health"], metrics["gold"]
    findings = "No gold accuracy comparison is available in this scope."
    if coverage["evaluated_rows"]:
        full = metrics["primary"]["full_pipeline"]["iou"]
        rules = metrics["primary"]["rules_only"]["iou"]
        hook = metrics["primary"]["full_pipeline"]["hook"]
        findings = (f"Full-pipeline IoU-Hit@1 is {percent(full['hit_at_1'])}, versus {percent(rules['hit_at_1'])} "
                    f"for rules-only; IoU-Recall@3 is {percent(full['recall_at_3'])} versus {percent(rules['recall_at_3'])}. "
                    f"Hook-Hit@1 is {percent(hook['hit_at_1'])}; Hook-Recall@3 is {percent(hook['recall_at_3'])}. "
                    f"Only {full['n_matched_pairs']} full-pipeline gold picks match; "
                    f"{percent(full['start_within_3s_share'])} of their starts are within three seconds. "
                    "These are descriptive results on available episodes, not proof of superiority or a basis for tuning.")
    text = [f"# Evaluation — {metrics['tag']}", "", "## System comparison (primary: hook match and IoU)", "",
        markdown_table(primary_comparison_rows(metrics)), "",
        "The shorter-clip overlap rule was replaced because long windows can cover a gold moment without agreeing on its boundaries or hook.", "",
        "Chance is analytical for ONE uniformly sampled clean 20–60 s window, averaged over gold episodes; it is not the chance of three recommendations. Random uses 100 seeds: mean ± population SD. Its bootstrap resamples episode means after averaging seeds.", "",
        markdown_table([dict(Rule=k,Hit_chance=percent(v["hit_chance"]),Recall_chance=percent(v["recall_chance"])) for k,v in metrics["chance"].items()]), "",
        "## Lenient secondary comparison (50% shorter-clip overlap)", "",markdown_table(comparison_rows(metrics)), "",
        f"Available gold: {coverage['evaluated_episodes']} (n={len(coverage['evaluated_episodes'])}, {coverage['evaluated_rows']} gold promos). Pending gold episodes: {coverage['pending_episodes']}.",
        "No gold yet in this evaluation scope." if not coverage["evaluated_rows"] else "These are partial-coverage results, not performance over all gold episodes or all 50 episodes.", "",
        "95% percentile bootstrap intervals resample episodes, preserving all gold picks within each episode. Small n gives unstable uncertainty estimates. Missing gold episodes are not scored as misses. Recall@3 is gold-row weighted; Hit@1 is episode weighted.", "",
        "An all-success sample can yield a degenerate [100%, 100%] bootstrap interval; this does not establish certainty on unseen episodes.", "",
        "## Looser comparison (30% shorter-clip overlap)", "", markdown_table(comparison_rows(metrics, "0.3")), "",
        "Each gold pick may match any top-three proposal; matches are not one-to-one. Start error uses the closest start among matching top-three proposals for each gold pick. Generator self-score evaluates original proposals; full pipeline uses final validated spans. Capped promos follow judged promos by self_score for Recall@3 only. Unsafe/rejected promos are excluded from the full pipeline.", "",
        "## Pipeline health", "", f"{health['n_episodes']} episodes; {health['n_candidates']} original saved candidates.", "",
        markdown_table([{"Before validator": key, "n": value["n"], "Compliant": value["count"], "Share": percent(value["share"])} for key,value in health["before_validator"].items()]), "",
        f"Validator outcomes: {health['validator']['outcomes']}. Flags: {health['validator']['flag_counts']}. Hook reassignment share: {percent(health['validator']['hook_reassigned_share'])}.", "",
        markdown_table([{"Fix / audit type":k,"Count":v} for k,v in Counter(health["validator"]["fix_types"]).most_common()]), "",
        "## Judge distributions", "", markdown_table([{"Dimension":k,"Histogram":v} for k,v in health["judge"]["distributions"].items()]), "",
        f"Identical-score-vector tie share: {percent(health['judge']['identical_score_vector_tie_share'])}; denominator is judged promos in duplicate vectors / all judged promos. Legacy missing post_priority: {health['judge']['missing_post_priority']}. Safety: {health['judge']['ad_safety']}; unverified: {health['judge']['unverified']}.", "",
        "## Judge versus human", "", f"Valid human ratings: {metrics['human']['n_rated']}. No agreement is claimed before ratings are supplied.", "",
        markdown_table([{"Compared with":f, **metrics['human'][f]} for f in ("final_score", "post_priority")]), "",
        markdown_table([{"Preferred order":key,**metrics['human'][key]} for key in ('judge_order','editor_order','final_order')]), "",
        "Order agreement correlates human ratings with negative rank, so a positive coefficient means agreement about which clip to prefer.", "",
        "Pairwise agreement excludes tied human ratings and gives model ties half credit. Intervals use 2,000 bootstrap draws of rated clips within each comparison. Editor order covers only clips the editor ranked.", "",
        markdown_table([{"Group": key, **value} for key, value in metrics['human']['top20'].items()]), "",
        *[part for key, rows in metrics['human']['order_disagreements'].items()
          for part in (f"### Biggest disagreements with {key.replace('_', ' ')}", "", markdown_table(rows), "")],
        markdown_table(metrics["human"]["biggest_disagreements"]), "",
        "### Agreement by blind sheet", "",
        "Editor and final order include any judge-order fallback positions saved by the editor pass.", "",
        markdown_table([{"Sheet": sheet, "Rated": result["n_rated"], "Order": order,
                         **result[order]} for sheet, result in metrics.get("human_by_sheet", {}).items()
                        for order in ("judge_order", "editor_order", "final_order")]), "",
        "## Cost", "", health["cost"]["scope"], "", markdown_table(health["cost"]["stages"]), "",
        health["cost"].get("allocation_note", ""), "", "### Per-episode allocated cost", "",
        markdown_table(health["cost"]["per_episode"]), "", f"Logged stage elapsed seconds: {health['cost']['stage_wall_s']}", "",
        "## Gemini versus Groq (shared episodes)", "", markdown_table(metrics["model_comparison"]), "",
        "No matching-gold claim is made when n_gold is zero. Legacy prompts and missing priority scores make this an observational comparison, not a controlled model experiment.", "",
        "## Findings and failures", "",
        findings, "",
        f"{len(metrics['failures'])} failure-gallery entries. Add your analysis to my_reason in the CSV; reruns preserve existing annotations.", "",
        markdown_table(metrics["failures"]), "",
        "## Rules-only selected windows", "",
        markdown_table([{"Episode": number, "Span": f"{format_timestamp(p['span'][0])}–{format_timestamp(p['span'][1])}",
                         "First line": p["first_line"], "Last line": p["last_line"]}
                        for number, promos in metrics["rules_only_examples"].items() for p in promos]), "",
        "Baselines are fixed screenplay-only heuristics documented in eval/README.md, not fitted to gold. Random is a seeded sanity floor; either baseline may return fewer than three disjoint valid windows.", "",
        "## Warnings", "", "\n".join(f"- {w}" for w in metrics["warnings"]) or "None."]
    (target/"eval_report.md").write_text("\n".join(text)+"\n", encoding="utf-8")


def run_eval(config, tag, *, gold_path=None, eval_dir=Path("eval"), health_only=False):
    saved = load_saved(config, tag)
    settings = config.get("evaluation", {})
    seed, samples = settings.get("seed", 20261106), settings.get("bootstrap_samples", 2000)
    gold = load_gold(Path(gold_path or settings.get("gold_csv", "eval/gold.csv")), saved["episodes"])
    designated = set(settings.get("gold_episodes", [])) or set(gold["declared_episodes"])
    evaluated = [] if health_only else [g for g in gold["rows"] if g["episode"] in saved["available"] and g["episode"] in designated]
    target = saved["output"] / f"eval_{tag}"
    target.mkdir(parents=True, exist_ok=True)
    inputs = system_inputs(saved)
    baseline_examples = {}
    for system in ("random", "rules_only"):
        baseline_promos = []
        for number in saved["available"]:
            promos = build_baseline(saved["episodes"][number], system, seed=seed, rules=settings.get("baseline_rules"))
            directory = target / "baselines" / system
            directory.mkdir(parents=True, exist_ok=True)
            (directory/f"episode_{number:03d}.json").write_text(json.dumps([p.model_dump() for p in promos], ensure_ascii=False, indent=2), encoding="utf-8")
            baseline_promos.extend(promos)
        inputs[system] = [promo_record(p) for p in baseline_promos]
        if system == "rules_only":
            baseline_examples = {str(number):[{"span": p.final_span, "first_line": p.hook_text, "last_line":p.cliffhanger_text}
                for p in baseline_promos if p.episode == number] for number in saved["available"]}
    systems = {system:{str(threshold):evaluate_system(inputs[system], evaluated, threshold=threshold, rule="lenient", seed=seed, samples=samples)
                       for threshold in (.5, .3)} for system in SYSTEMS}
    strict = strict_evaluation(inputs,evaluated,saved["episodes"],seed=seed,samples=samples)
    write_seed = target/'baselines/random/seed_manifest.json'
    write_seed.parent.mkdir(parents=True,exist_ok=True)
    write_seed.write_text(json.dumps({'master_seed':seed,'metric_seeds':strict['random_seeds'],
        'note':'episode_XXX.json files show the master-seed example; reported random metrics average these 100 metric seeds.'},indent=2),encoding='utf-8')
    systems["random"] = strict["lenient_random"]
    whole_sheet = Path(eval_dir)/f"to_rate_{tag}.csv"
    top_sheet = Path(eval_dir)/f"to_rate_top_{tag}.csv"
    human = human_agreement((whole_sheet, top_sheet), saved)
    human_by_sheet = {"whole_range": human_agreement(whole_sheet, saved),
                      "top_of_list": human_agreement(top_sheet, saved), "both": human}
    failures = failure_gallery(saved, evaluated, inputs["full_pipeline"], human, eval_dir, target, tag)
    metrics = {"tag":tag, "api_calls_made":0, "seed":seed, "bootstrap_samples":samples,
        "gold":{"rows_found":gold["rows_found"],"rows_valid":gold["rows_valid"],
                "valid_episodes":sorted({g["episode"] for g in gold["rows"]}),
                "evaluated_episodes":sorted({g["episode"] for g in evaluated}), "evaluated_rows":len(evaluated),
                "pending_episodes": sorted(designated-set(saved["available"])) if not health_only else [], "health_only":health_only},
        "systems":systems,"primary":strict["primary"],"chance":strict["chance"],"random_seeds":strict["random_seeds"],"health":pipeline_health(saved, config, tag),"human":human,
        "human_by_sheet":human_by_sheet, "model_comparison":model_comparison(config, gold["rows"], saved["warnings"]),
        "failures":failures,"rules_only_examples":baseline_examples,
        "baseline_rules":DEFAULT_RULES | settings.get("baseline_rules", {}),
        "warnings":saved["warnings"]+gold["warnings"]+human["warnings"]}
    (target/"metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(metrics, target)
    print(f"EVAL {tag}: gold n={len(metrics['gold']['evaluated_episodes'])} episodes / {len(evaluated)} promos; {saved['available']} available; API calls made=0; {target/'eval_report.md'}")
    print(markdown_table(primary_comparison_rows(metrics)))
    for warning in metrics["warnings"]:
        print("WARNING:", warning)
    return metrics


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("run", "export-rating-sheet"):
        parser = commands.add_parser(name)
        parser.add_argument("--tag", required=True)
        parser.add_argument("--config", type=Path, default=Path("config.yaml"))
        parser.add_argument("--gold", type=Path)
        parser.add_argument("--eval-dir", type=Path, default=Path("eval"))
        parser.add_argument("--parsed-dir", type=Path)
        parser.add_argument("--output-dir", type=Path)
        if name == "run":
            parser.add_argument("--health-only", action="store_true")
        else:
            parser.add_argument("--n", type=int, default=20)
            parser.add_argument("--force", action="store_true")
            parser.add_argument("--top", action="store_true", help="Sample the saved editor pool and omit previously rated IDs")
    args = cli.parse_args()
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.tag):
        cli.error("Invalid tag")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    for flag in ("parsed_dir", "output_dir"):
        if getattr(args, flag):
            config[flag] = str(getattr(args, flag))
    try:
        if args.command == "run":
            run_eval(config, args.tag, gold_path=args.gold, eval_dir=args.eval_dir, health_only=args.health_only)
        else:
            if args.n < 1:
                raise ValueError("n must be positive")
            export_rating_sheet(config, args.tag, args.n, eval_dir=args.eval_dir, gold_path=args.gold, force=args.force, top=args.top)
    except (ValueError, OSError) as error:
        cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
