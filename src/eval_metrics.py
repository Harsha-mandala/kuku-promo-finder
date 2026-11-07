"""Offline overlap metrics, episode bootstrap and human agreement."""

import math
import random
from collections import defaultdict
from statistics import mean


def overlap_fraction(a, b):
    """Use overlap divided by the shorter duration, never intersection/union."""
    shorter = min(a[1] - a[0], b[1] - b[0])
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / shorter if shorter > 0 else 0.0


def intersection_over_union(a, b):
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = a[1]-a[0] + b[1]-b[0] - overlap
    return overlap / union if union > 0 else 0.0


def matches(promo, gold, threshold=0.5, rule="iou"):
    if promo["episode"] != gold["episode"]:
        return False
    if rule == "hook":
        return abs(promo["span"][0]-gold["span"][0]) <= 3 and intersection_over_union(promo["span"], gold["span"]) >= .3
    if rule == "iou":
        return intersection_over_union(promo["span"], gold["span"]) >= threshold
    return overlap_fraction(promo["span"], gold["span"]) >= threshold


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_ci(episodes, *, seed=20261106, samples=2000):
    """Resample entire episodes; recall weights each episode by its gold count."""
    if not episodes:
        return {"hit_at_1": None, "recall_at_3": None}
    rng = random.Random(seed)
    hits, recalls = [], []
    for _ in range(samples):
        chosen = rng.choices(episodes, k=len(episodes))
        hits.append(mean(item["hit_at_1"] for item in chosen))
        recalls.append(sum(item["matched_gold"] for item in chosen) / sum(item["n_gold"] for item in chosen))
    return {"hit_at_1": [percentile(hits, .025), percentile(hits, .975)],
            "recall_at_3": [percentile(recalls, .025), percentile(recalls, .975)]}


def evaluate_system(promos, gold, *, threshold=0.5, rule="iou", seed=20261106, samples=2000):
    by_episode = defaultdict(list)
    for promo in promos:
        by_episode[promo["episode"]].append(promo)
    gold_by_episode = defaultdict(list)
    for item in gold:
        gold_by_episode[item["episode"]].append(item)
    results, distances = [], []
    for number, picks in sorted(gold_by_episode.items()):
        top = by_episode[number][:3]
        first = top[:1] if top and not top[0].get("recall_only") else []
        hit = int(any(matches(promo, target, threshold, rule) for promo in first for target in picks))
        found = 0
        for target in picks:
            matching = [promo for promo in top if matches(promo, target, threshold, rule)]
            if matching:
                found += 1
                distances.append(min(abs(promo["span"][0] - target["span"][0]) for promo in matching))
        results.append({"episode": number, "hit_at_1": hit, "matched_gold": found, "n_gold": len(picks),
                        "recall_at_3": found / len(picks), "n_candidates": len(by_episode[number])})
    return {"threshold": threshold, "n_episodes": len(results), "n_gold": len(gold),
            "hit_at_1": mean(item["hit_at_1"] for item in results) if results else None,
            "recall_at_3": sum(item["matched_gold"] for item in results) / len(gold) if gold else None,
            "mean_start_error_s": mean(distances) if distances else None,
            "start_within_3s_share": mean(value <= 3 for value in distances) if distances else None,
            "n_matched_pairs": len(distances), "per_episode": results,
            "ci95": bootstrap_ci(results, seed=seed, samples=samples)}


def average_ranks(values):
    positions = defaultdict(list)
    for rank, (_, index) in enumerate(sorted((value, i) for i, value in enumerate(values)), 1):
        positions[values[index]].append(rank)
    return [mean(positions[value]) for value in values]


def spearman(a, b):
    """Pearson correlation of midranks, including ties; constant series are undefined."""
    if len(a) < 2:
        return None
    x, y = average_ranks(a), average_ranks(b)
    dx, dy = [value-mean(x) for value in x], [value-mean(y) for value in y]
    denominator = math.sqrt(sum(value*value for value in dx) * sum(value*value for value in dy))
    return sum(u*v for u, v in zip(dx, dy)) / denominator if denominator else None


def pairwise_agreement(a, b):
    comparable, agreement, model_ties = 0, 0, 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] == a[j]:
                continue
            comparable += 1
            product = (a[i]-a[j]) * (b[i]-b[j])
            if product > 0:
                agreement += 1
            elif product == 0:
                agreement += .5
                model_ties += 1
    return {"share": agreement/comparable if comparable else None,
            "comparable_pairs": comparable, "model_ties": model_ties}
