"""Independently judge clips, batching blind script-only requests."""

import argparse
import csv
import copy
import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, RootModel, model_validator

from .cliprender import render_judge_clip
from .context import load_episodes
from .llm import CallBudget, LLMBudgetError, LLMNetworkError, LLMQuotaError, LLMRequestError, LLMOutputError, _json_text, estimate_json_tokens, get_llm, request_token_limit, step_model_config
from .validate import ValidatedPromo

SYSTEM_PATH = Path(__file__).parent / "prompts/judge_system.md"


def judge_settings(config, provider, model):
    reasoning = None
    if provider == "groq" and "gpt-oss" in model:
        reasoning = (config or {}).get("models", {}).get("judge", {}).get("reasoning_effort", "low")
    return {"temperature": 0.0, "reasoning_effort": reasoning}


def judge_fingerprint(ep, promo, provider, model, config=None, *, system=None):
    payload = [provider, model, judge_settings(config, provider, model), system if system is not None else SYSTEM_PATH.read_text(encoding="utf-8"), render_judge_clip(ep, promo)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class JudgeScore(BaseModel):
    promo_id: str
    hook_score: int = Field(ge=1, le=5)
    cliffhanger_score: int = Field(ge=1, le=5)
    standalone_score: int = Field(ge=1, le=5)
    emotion_score: int = Field(ge=1, le=5)
    post_priority: int = Field(ge=1, le=10)
    ad_safety: Literal["safe", "caution", "unsafe"]
    ad_safety_reason: str
    weakest_point: str
    one_line_pitch: str
    why_it_works: str = Field(min_length=1, description="One short line about THIS final clip only.")
    ad_safety_verified: bool = True

    @model_validator(mode="before")
    @classmethod
    def normalize_safe_reason(cls, value):
        if isinstance(value, dict) and value.get("ad_safety") == "safe":
            value = dict(value)
            value["ad_safety_reason"] = ""
        return value

class JudgeBatch(RootModel[list[JudgeScore]]):
    pass


class SavedJudgeScore(JudgeScore):
    """Read older scores without inventing a posting-priority judgment."""

    post_priority: int | None = Field(default=None, ge=1, le=10)


class SavedJudgeBatch(RootModel[list[SavedJudgeScore]]):
    pass


def verify_safety(score, episode, promo):
    positions = {shot.id: i for i, shot in enumerate(episode.shots)}
    shots = episode.shots[positions[promo.start_shot_id]:positions[promo.end_shot_id] + 1]
    ids = {line.id for shot in shots for line in shot.lines}
    citations = re.findall(r"(?<![\w])(?:E\d+_)?L\d+(?![\w])", score.ad_safety_reason)
    cited = {value if value.startswith('E') else f'E{episode.number:03d}_{value}' for value in citations}
    verified = score.ad_safety == 'safe' or bool(ids & cited)
    return score.model_copy(update={"ad_safety_verified": verified})


def recover_raw_scores(raw_dir, promos, episodes, provider, model, config=None):
    """Recover grounded completed responses without changing raw logs or calling APIs."""
    by_id = {p.id: p for p in promos}
    system = SYSTEM_PATH.read_text(encoding="utf-8")
    recovered, provenance = {}, {}
    for path in sorted(Path(raw_dir).glob("judge_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("provider") != provider or record.get("model") != model or record.get("prompt", {}).get("system") != system:
                continue
            if record.get("temperature") != 0:
                continue
            if record.get("reasoning_effort") != judge_settings(config, provider, model)["reasoning_effort"]:
                continue
            user = record["prompt"]["user"]
            scores = JudgeBatch.model_validate_json(_json_text(record.get("raw_response", ""))).root
            expected = re.findall(r"(?m)^CLIP (\S+) \| duration", user)
            if len(scores) != len(expected) or {s.promo_id for s in scores} != set(expected):
                continue
            numbers = [by_id[s.promo_id].episode for s in scores if s.promo_id in by_id]
            if len(numbers) != len(set(numbers)):
                continue
            for score in scores:
                promo = by_id.get(score.promo_id)
                if promo and render_judge_clip(episodes[promo.episode], promo) in user:
                    recovered[score.promo_id] = verify_safety(score, episodes[promo.episode], promo)
                    provenance[score.promo_id] = str(path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return recovered, provenance


def load_promos(directory: Path, episodes: set[int] | None = None) -> list[ValidatedPromo]:
    promos = []
    for path in sorted(directory.glob("episode_*.json")):
        for raw in json.loads(path.read_text(encoding="utf-8")):
            item = ValidatedPromo.model_validate(raw)
            if episodes is None or item.episode in episodes:
                promos.append(item)
    if len({item.id for item in promos}) != len(promos):
        raise ValueError("Duplicate promo IDs in validated inputs")
    return promos


def judge_prompt(promos, episodes):
    return "Judge EACH clip independently. Return exactly one score per supplied promo_id.\n\n" + "\n\n".join(render_judge_clip(episodes[item.episode], item) for item in promos)


def cap_promos(promos, per_episode=3):
    """Prioritize validated self scores locally; never send them to the judge."""
    selected, capped, counts = [], [], {}
    for promo in sorted(promos, key=lambda p: (p.episode, -p.self_score, p.id)):
        if counts.get(promo.episode, 0) >= per_episode:
            capped.append(promo)
        else:
            selected.append(promo)
            counts[promo.episode] = counts.get(promo.episode, 0) + 1
    return selected, capped


def plan_judge(promos, episodes, config, provider, batch_size=3):
    """Split requests by actual serialized prompt size."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    system = SYSTEM_PATH.read_text(encoding="utf-8")
    limit = request_token_limit(config, provider)
    plans = []
    def split(items):
        user = judge_prompt(items, episodes)
        tokens = estimate_json_tokens(system, user, JudgeBatch)
        if limit and tokens > limit and len(items) > 1:
            half = len(items) // 2
            split(items[:half])
            split(items[half:])
        else:
            plans.append({"promos": items, "system": system, "user": user, "tokens": tokens,
                          "oversized": bool(limit and tokens > limit)})
    queues = defaultdict(deque)
    for promo in promos:
        queues[promo.episode].append(promo)
    order = sorted(queues)
    cursor = 0
    while any(queues.values()):
        items = []
        for _ in range(len(order)):
            number = order[cursor]
            cursor = (cursor + 1) % len(order)
            if queues[number]:
                items.append(queues[number].popleft())
                if len(items) == batch_size:
                    break
        split(items)
    return plans


def run_judge(config, tag, *, client=None, provider=None, model=None, batch_size=3,
              max_api_calls=None, force=False, budget=None, episode_numbers=None, dry_run=False, progress_cb=None):
    output = Path(config.get("output_dir", "outputs"))
    if not tag or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in tag):
        raise ValueError("Invalid source tag")
    promos = [p for p in load_promos(output / f"validated_{tag}", episode_numbers) if p.status != "rejected"]
    promos, capped = cap_promos(promos)
    episodes = {ep.number: ep for ep in load_episodes(config.get("parsed_dir", "parsed"))}
    provider, model = step_model_config(config, "judge", provider, model)
    system = SYSTEM_PATH.read_text(encoding="utf-8")
    settings = judge_settings(config, provider, model)
    fingerprints = {p.id: judge_fingerprint(episodes[p.episode], p, provider, model, config) for p in promos}
    directory = output / f"judged_{tag}"
    saved, previous = {}, {}
    if not force:
        try:
            saved = {s.promo_id: s for s in JudgeBatch.model_validate_json((directory / "scores.json").read_text(encoding="utf-8")).root}
            previous = json.loads((directory / "score_inputs.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
    valid = {key: score for key, score in saved.items() if key in fingerprints and previous.get(key) == fingerprints[key]}
    raw_dir = config.get("llm", {}).get("raw_log_dir") or (client.raw_log_dir if client else output / "llm_raw")
    recovered, provenance = recover_raw_scores(raw_dir, promos, episodes, provider, model, config)
    recovered = {key: score for key, score in recovered.items() if key not in valid}
    valid.update(recovered)
    pending = [p for p in promos if p.id not in valid]
    plans = plan_judge(pending, episodes, config, provider, batch_size)
    stats = {"provider": provider, "model": model, "requests": sum(not p["oversized"] for p in plans),
             "estimated_input_tokens": sum(p["tokens"] for p in plans if not p["oversized"]),
             "skipped": list(valid), "done": list(valid), "pending": [p.id for p in pending],
             "oversized": [p["promos"][0].id for p in plans if p["oversized"]],
             "api_calls": 0, "tokens_in": 0, "tokens_out": 0, "stopped": None}
    stats["not_judged_cap"] = [p.id for p in capped]
    stats["recovered_from_raw"] = {key: provenance[key] for key in recovered}
    stats.update(settings)
    stats["batches"] = [[p.id for p in plan["promos"]] for plan in plans]
    if dry_run:
        print(f"JUDGE DRY RUN | {provider}/{model} | {len(promos)} promos | requests={stats['requests']} | estimated_input_tokens={stats['estimated_input_tokens']} | skipped={len(valid)} | oversized={stats['oversized']}")
        return stats
    directory.mkdir(parents=True, exist_ok=True)
    if client is None and plans:
        client_config = copy.deepcopy(config)
        if settings["reasoning_effort"] is not None:
            client_config.setdefault("llm", {}).setdefault("providers", {}).setdefault(provider, {})["reasoning_effort"] = settings["reasoning_effort"]
        client = get_llm(client_config, provider, model)
    if client is not None:
        client.temperature = 0.0
        judge_rpm = config.get("models", {}).get("judge", {}).get("requests_per_minute")
        if judge_rpm is not None:
            if judge_rpm <= 0:
                raise ValueError("models.judge.requests_per_minute must be positive")
            client.limiter.interval = max(client.limiter.interval, 60.0 / judge_rpm)
        if settings["reasoning_effort"] is not None:
            client.reasoning_effort = settings["reasoning_effort"]
        client.call_budget = budget if budget is not None else CallBudget(max_api_calls if max_api_calls is not None else config.get("llm", {}).get("max_api_calls_per_run", 14))
    used_before = client.call_budget.used if client else 0
    network_before = client.call_budget.connection_errors if client else 0
    usage_start = 0
    if client and client.usage_log.exists():
        with client.usage_log.open(encoding="utf-8", newline="") as file:
            usage_start = len(list(csv.DictReader(file)))
    errors = []
    def save():
        # Retain scores for episodes outside a selected subset; invalidate changed selected clips.
        merged = {key: score for key, score in saved.items() if key not in fingerprints}
        merged.update(valid)
        inputs = dict(previous)
        inputs.update({key: fingerprints[key] for key in valid})
        (directory / "scores.json").write_text(JudgeBatch(list(merged.values())).model_dump_json(indent=2), encoding="utf-8")
        (directory / "score_inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    for plan in plans:
        if plan["oversized"]:
            errors.append(f"Request too large: {plan['promos'][0].id}")
            continue
        expected = {p.id for p in plan["promos"]}
        def check(result):
            actual = [score.promo_id for score in result.root]
            return [] if set(actual) == expected and len(actual) == len(expected) else ["[invalid_json] Return each requested promo_id exactly once; no missing/extra IDs"]
        try:
            print("JUDGE BATCH: " + ", ".join(p.id for p in plan["promos"]), flush=True)
            scores = client.generate_json(plan["system"], plan["user"], JudgeBatch,
                                          "judge_" + "_".join(sorted(expected)), extra_validator=check, max_repairs=1, temperature=0.0)
            by_id = {p.id: p for p in plan["promos"]}
            valid.update({s.promo_id: verify_safety(s, episodes[by_id[s.promo_id].episode], by_id[s.promo_id]) for s in scores.root})
            save()
            if progress_cb:
                for promo in plan['promos']:
                    progress_cb({'stage':'judge','state':'progress','episode':promo.episode,'done':len(valid),'total':len(promos)})
        except (LLMQuotaError, LLMBudgetError, LLMNetworkError) as error:
            if config.get("_interactive_run"):
                raise
            stats["stopped"] = type(error).__name__
            errors.append(str(error))
            print(str(error), flush=True)
            break
        except (LLMRequestError, LLMOutputError) as error:
            if config.get("_interactive_run"):
                raise
            errors.append(str(error))
    save()
    if client:
        stats["api_calls"] = client.call_budget.used - used_before
        stats["connection_errors"] = client.call_budget.connection_errors - network_before
        # Provider usage is audited from the new rows, including failures and repairs.
        usage = []
        if client.usage_log.exists():
            with client.usage_log.open(encoding="utf-8", newline="") as file:
                usage = list(csv.DictReader(file))[usage_start:]
        attempts = [r for r in usage if r["cached"] == "False"]
        stats["usage_missing_attempts"] = sum(not r["in_tokens"] or not r["out_tokens"] for r in attempts)
        for field, column in (("tokens_in", "in_tokens"), ("tokens_out", "out_tokens")):
            known = [int(r[column]) for r in attempts if r[column]]
            stats[field] = sum(known) if known or not attempts else None
    stats["done"] = [p.id for p in promos if p.id in valid]
    stats["pending"] = [p.id for p in promos if p.id not in valid]
    (directory / "errors.log").write_text("\n".join(errors), encoding="utf-8")
    (directory / "run_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"JUDGE: {len(valid)} done | {len(stats['pending'])} pending | API calls={stats['api_calls']} | stop={stats['stopped']}")
    return stats


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tag", required=True)
    cli.add_argument("--config", type=Path, default=Path("config.yaml"))
    cli.add_argument("--provider")
    cli.add_argument("--model")
    cli.add_argument("--batch-size", type=int, default=3)
    cli.add_argument("--max-api-calls", type=int)
    cli.add_argument("--force", action="store_true")
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        stats = run_judge(config, args.tag, provider=args.provider, model=args.model, batch_size=args.batch_size,
                          max_api_calls=args.max_api_calls, force=args.force, dry_run=args.dry_run)
        if not args.dry_run and stats["pending"]:
            cli.exit(1, "Some promos remain unjudged; saved scores are ready to resume.\n")
    except (OSError, ValueError, LLMRequestError) as error:
        cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
