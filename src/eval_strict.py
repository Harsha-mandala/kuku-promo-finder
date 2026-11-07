"""Strict matching, analytical chance and repeated-seed random evaluation."""

import random
from statistics import mean, pstdev

from .baselines import build_baseline, windows
from .eval_metrics import bootstrap_ci, evaluate_system, matches


def coverage(promos, gold, episodes):
    fractions = []
    for number in sorted({g["episode"] for g in gold}):
        spans = sorted(p["span"] for p in [p for p in promos if p["episode"] == number][:3])
        merged = []
        for start, end in spans:
            start, end = max(0, start), min(episodes[number].duration, end)
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        fractions.append(sum(max(0, b-a) for a,b in merged) / episodes[number].duration)
    return mean(fractions) if fractions else None


def chance_levels(gold, episodes):
    """Enumerate every valid window: chance is for ONE uniformly sampled window."""
    result = {}
    for rule in ("hook", "iou", "lenient"):
        rows = []
        for number in sorted({g["episode"] for g in gold}):
            picks = [g for g in gold if g["episode"] == number]
            valid = [dict(episode=number, span=(a,b)) for _,_,a,b in windows(episodes[number],20,60)]
            any_match = mean(any(matches(p,g,rule=rule) for g in picks) for p in valid) if valid else 0.0
            recall = mean(mean(matches(p,g,rule=rule) for p in valid) for g in picks) if valid else 0.0
            rows.append({"episode":number,"n_windows":len(valid),"hit_chance":any_match,"recall_chance":recall})
        result[rule] = {"hit_chance":mean(r["hit_chance"] for r in rows) if rows else None,
                        "recall_chance":mean(r["recall_chance"] for r in rows) if rows else None,
                        "per_episode":rows}
    return result


def average_runs(runs, seed, samples):
    """Average point estimates over seeds, then bootstrap episode-level averages."""
    result = dict(runs[0])
    fields = ("hit_at_1","recall_at_3","mean_start_error_s","start_within_3s_share")
    result["seed_sd"] = {}
    for field in fields:
        values = [r[field] for r in runs if r[field] is not None]
        result[field] = mean(values) if values else None
        result["seed_sd"][field] = pstdev(values) if values else None
    per_episode = []
    for i, episode in enumerate(runs[0]["per_episode"]):
        row = dict(episode)
        for field in ("hit_at_1","matched_gold","recall_at_3"):
            row[field] = mean(run["per_episode"][i][field] for run in runs)
        per_episode.append(row)
    result["per_episode"] = per_episode
    result["ci95"] = bootstrap_ci(per_episode,seed=seed,samples=samples)
    result["n_matched_pairs"] = mean(r["n_matched_pairs"] for r in runs)
    result["n_seeds"] = len(runs)
    return result


def strict_evaluation(inputs, gold, episodes, *, seed, samples, random_seeds=100):
    primary = {}
    for system, promos in inputs.items():
        primary[system] = {rule:evaluate_system(promos,gold,rule=rule,seed=seed,samples=samples)
                           for rule in ("hook","iou")}
        primary[system]["coverage"] = coverage(promos,gold,episodes)
    rng = random.Random(seed)
    seeds = rng.sample(range(2**32),random_seeds)
    numbers = sorted({g["episode"] for g in gold})
    cache = {n:list(windows(episodes[n],20,60)) for n in numbers}
    runs = {"hook":[],"iou":[],"lenient":[],"lenient_0.3":[]}
    covers = []
    for current_seed in seeds:
        promos = []
        for number in numbers:
            selected = build_baseline(episodes[number],"random",seed=current_seed,window_cache=cache[number])
            promos.extend({"episode":p.episode,"span":p.final_span,"id":p.id} for p in selected)
        for rule in runs:
            runs[rule].append(evaluate_system(promos,gold,rule="lenient" if rule.startswith("lenient") else rule,
                                              threshold=.3 if rule.endswith("0.3") else .5,samples=1))
        covers.append(coverage(promos,gold,episodes))
    for rule in ("hook","iou"):
        primary["random"][rule] = average_runs(runs[rule],seed,samples)
    values = [value for value in covers if value is not None]
    primary["random"]["coverage"] = mean(values) if values else None
    primary["random"]["coverage_sd"] = pstdev(values) if values else None
    return {"primary":primary,"chance":chance_levels(gold,episodes),"random_seeds":seeds,
            "lenient_random":{key:average_runs(runs[rule],seed,samples) for key,rule in (("0.5","lenient"),("0.3","lenient_0.3"))}}
