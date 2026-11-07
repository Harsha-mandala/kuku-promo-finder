"""Audit logged costs without inventing tag IDs or batch-level token splits."""

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def cost_report(output, tag, numbers, candidate_provider, judge_provider):
    path = Path(output) / "llm_usage.csv"
    if not path.exists():
        return {"scope": "No usage log available", "stages": [], "per_episode": [], "stage_wall_s": {}}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        usage = list(csv.DictReader(stream))
    events_path = Path(output) / f"pipeline_{tag}_events.jsonl"
    intervals, opened = [], {}
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
                time = datetime.fromisoformat(event["time"])
                stage, state = event["stage"], event["state"]
            except (ValueError, KeyError, TypeError):
                continue
            if state == "running":
                opened[stage] = time
            elif state in ("complete", "stopped") and stage in opened:
                intervals.append((stage, opened.pop(stage), time))
    groups, allocations = {}, {}
    wall = defaultdict(float)
    for stage, start, end in intervals:
        wall[stage] += (end - start).total_seconds()
    for row in usage:
        if row.get("cached") != "False":
            continue
        request_tag = row.get("tag", "")
        stage = "trailer_judge" if request_tag.startswith("judge_trailers_") else next((
            s for s in ("candidates", "judge", "bible", "trailers", "editor") if request_tag.startswith(s)), None)
        if not stage:
            continue
        episode_ids = sorted({int(n) for n in re.findall(r"E(\d+)", request_tag)})
        if intervals:
            try:
                time = datetime.fromisoformat(row["time"])
            except (ValueError, KeyError):
                continue
            event_stage = "trailers" if stage == "trailer_judge" else stage
            if not any(s == event_stage and start <= time <= end for s, start, end in intervals):
                continue
        else:
            # Legacy CSVs have no run tag. Report related history, never exact-tag costs.
            if stage in ("trailers","trailer_judge","editor"):
                continue
            expected = candidate_provider if stage == "candidates" else judge_provider
            if not episode_ids or not set(episode_ids).issubset(numbers) or row.get("provider") != expected:
                continue
        provider = row.get("provider", "unknown")
        key = (stage, provider)
        group = groups.setdefault(key, {"stage": stage, "provider": provider, "api_calls": 0,
                                       "input_tokens_known": 0, "output_tokens_known": 0,
                                       "unknown_usage_attempts": 0, "provider_latency_s": 0.0})
        group["api_calls"] += 1
        group["input_tokens_known"] += int(row.get("in_tokens") or 0)
        group["output_tokens_known"] += int(row.get("out_tokens") or 0)
        group["unknown_usage_attempts"] += int(not row.get("in_tokens") or not row.get("out_tokens"))
        latency = float(row.get("latency_s") or 0)
        group["provider_latency_s"] += latency
        for number in episode_ids:
            per = allocations.setdefault((stage, provider, number), {"stage": stage, "provider": provider,
                "episode": number, "attempt_share": 0.0, "input_tokens_known_allocated": 0.0,
                "output_tokens_known_allocated": 0.0, "unknown_usage_attempt_share": 0.0,
                "provider_latency_s_allocated": 0.0})
            per["attempt_share"] += 1 / len(episode_ids)
            per["input_tokens_known_allocated"] += int(row.get("in_tokens") or 0) / len(episode_ids)
            per["output_tokens_known_allocated"] += int(row.get("out_tokens") or 0) / len(episode_ids)
            per["unknown_usage_attempt_share"] += int(not row.get("in_tokens") or not row.get("out_tokens")) / len(episode_ids)
            per["provider_latency_s_allocated"] += latency / len(episode_ids)
    scope = ("Tag pipeline time intervals matched to usage CSV; pings excluded. Concurrent runs could make temporal attribution ambiguous."
             if intervals else "Legacy related provider/episode history only: CSV has no run tag and no pipeline event intervals exist. Exact tag cost is unavailable.")
    return {"scope": scope, "stages": list(groups.values()), "per_episode": list(allocations.values()),
            "stage_wall_s": dict(wall),
            "allocation_note": "Batch tokens, attempt shares and provider latency are divided equally across requested episodes; these are estimates, not per-episode provider measurements. Unknown usage is not zero. CSV latency excludes throttle/backoff waits."}
