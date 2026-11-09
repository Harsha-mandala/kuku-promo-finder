"""Find grounded promo proposals and measure them without duration filtering."""

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import yaml

from .clips import clip_span
from .context import PROMPTS_DIR, bible_source_fingerprint, build_story_bible, load_episodes, load_hints, render_bible_compact, validate_bible
from .llm import LLMClient, LLMOutputError, LLMQuotaError, LLMRequestError, get_llm, estimate_json_tokens, request_token_limit, step_model_config, _json_text
from .models import Candidate, CandidateBatch, CandidateList, Episode, ScoredCandidate, StoryBible
from .render import render_episode_compact
from .llm import CallBudget, LLMBudgetError, LLMNetworkError, LLMBatchSplitError


@dataclass(frozen=True)
class SourceText:
    id: str
    shot_id: str
    text: str
    start: float
    end: float


def canonical_id(episode: Episode, value: str) -> str:
    """Resolve an episode-local display ID without guessing any source text."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    short = re.fullmatch(r"([LS])(\d+)", value)
    if short:
        return f"E{episode.number:03d}_{short.group(1)}{int(short.group(2)):03d}"
    return value


def source_lookup(episode: Episode) -> dict[str, SourceText]:
    """Index attached lines; empty shots expose their camera text by shot ID."""
    sources = {}
    for shot in episode.shots:
        for line in shot.lines:
            sources[line.id] = SourceText(line.id, shot.id, line.text, line.start, line.end)
        if not shot.lines:
            sources[shot.id] = SourceText(shot.id, shot.id, shot.camera_text, shot.start, shot.end)
    return sources


def candidate_validator(episode: Episode, max_candidates: int):
    """Repair only invalid source IDs, clip membership, and shot order."""
    positions = {shot.id: index for index, shot in enumerate(episode.shots)}
    sources = source_lookup(episode)

    def validate(result: CandidateList) -> list[str]:
        errors = []
        if result.episode != episode.number:
            errors.append(f"[episode_mismatch] episode must be {episode.number}, got {result.episode}")
        for number, candidate in enumerate(result.candidates, start=1):
            valid_ids = True
            for field in ("start_shot_id", "end_shot_id"):
                value = canonical_id(episode, getattr(candidate, field))
                if value not in positions:
                    errors.append(
                        f"[unknown_shot_id] candidate {number}: {field} {value!r} does not exist in episode {episode.number}. "
                        f"Use canonical IDs E{episode.number:03d}_S001 etc. for render labels [S001] etc."
                    )
                    valid_ids = False
            if not valid_ids:
                continue
            first = positions[canonical_id(episode, candidate.start_shot_id)]
            last = positions[canonical_id(episode, candidate.end_shot_id)]
            if first > last:
                errors.append(f"[shot_order] candidate {number}: start shot must not come after end shot")
                continue
            for field in ("hook_line_id", "cliffhanger_line_id"):
                value = canonical_id(episode, getattr(candidate, field))
                source = sources.get(value)
                if source is None:
                    hint = ""
                    if value in positions:
                        attached = episode.shots[positions[value]].lines
                        if attached:
                            hint = " This shot has attached lines; choose " + ", ".join(item.id for item in attached) + "."
                    errors.append(
                        f"[unknown_line_id] candidate {number}: {field} {value!r} is not an attached line ID "
                        "or an empty-shot camera ID in this episode. Use the rendered IDs, not quotes." + hint
                    )
                elif not first <= positions[source.shot_id] <= last:
                    errors.append(f"[line_outside_clip] candidate {number}: {field} {value!r} belongs to {source.shot_id}, outside the selected shots")
        return errors

    return validate


def score_candidates(episode: Episode, result: CandidateList) -> list[ScoredCandidate]:
    """Calculate source timing; leave all filtering to Step 4."""
    positions = {shot.id: index for index, shot in enumerate(episode.shots)}
    sources = source_lookup(episode)
    scored = []
    candidate_fields = set(Candidate.model_fields)
    for number, candidate in enumerate(result.candidates, start=1):
        candidate = candidate.model_copy(update={
            field: canonical_id(episode, getattr(candidate, field))
            for field in ("start_shot_id", "end_shot_id", "hook_line_id", "cliffhanger_line_id")
        })
        first, last = positions[candidate.start_shot_id], positions[candidate.end_shot_id]
        shots = episode.shots[first:last + 1]
        start, end = clip_span(episode, candidate.start_shot_id, candidate.end_shot_id)
        hook = sources[candidate.hook_line_id]
        cliffhanger = sources[candidate.cliffhanger_line_id]
        scored.append(ScoredCandidate(
            **candidate.model_dump(include=candidate_fields), candidate_id=f"E{episode.number:03d}_C{number:02d}",
            hook_text=hook.text, cliffhanger_text=cliffhanger.text,
            span_start=start, span_end=end, duration_s=round(end - start, 2),
            hook_offset_s=round(hook.start - start, 2),
            cliffhanger_end_gap_s=round(end - cliffhanger.end, 2),
            start_clean=shots[0].clean_start, end_clean=shots[-1].clean_end,
            n_shots=len(shots),
        ))
    return scored


def postprocess_candidates(episode: Episode, candidates: list[ScoredCandidate], max_candidates: int) -> tuple[list[ScoredCandidate], list[dict]]:
    """Keep higher-scored non-overlapping proposals; never ask the LLM again."""
    positions = {shot.id: index for index, shot in enumerate(episode.shots)}
    retained = []
    events = []
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda pair: (-pair[1].self_score, pair[0]))
    for index, candidate in indexed:
        first, last = positions[candidate.start_shot_id], positions[candidate.end_shot_id]
        conflict = None
        for _, kept in retained:
            other_first, other_last = positions[kept.start_shot_id], positions[kept.end_shot_id]
            shared = max(0, min(last, other_last) - max(first, other_first) + 1)
            if shared / min(last - first + 1, other_last - other_first + 1) > 0.5:
                conflict = kept
                break
        if conflict is not None:
            events.append({"episode": episode.number, "reason": "overlap_drop", "dropped": candidate.candidate_id, "kept": conflict.candidate_id, "dropped_score": candidate.self_score, "kept_score": conflict.self_score})
        elif len(retained) >= max_candidates:
            events.append({"episode": episode.number, "reason": "count_cap", "dropped": candidate.candidate_id})
        else:
            retained.append((index, candidate))
    retained.sort(key=lambda pair: pair[0])
    values = [candidate for _, candidate in retained]
    requested_min = min(3, max_candidates) if episode.duration > 90 else 1
    if len(values) < requested_min:
        events.append({"episode": episode.number, "reason": "count_shortfall", "retained": len(values), "requested_min": requested_min})
    return values, events


def previous_loglines(episode: Episode, episodes: list[Episode]) -> str:
    earlier = sorted((item for item in episodes if item.number < episode.number), key=lambda item: item.number)[-2:]
    if not earlier:
        return "None, this is the first episode"
    return "\n".join(f"Episode {item.number}: {item.logline}" for item in earlier)


def find_candidates(
    episode: Episode, episodes: list[Episode], bible: StoryBible,
    client: LLMClient, max_candidates: int = 5,
) -> list[ScoredCandidate]:
    if max_candidates < 1:
        raise ValueError("candidates_per_episode must be positive")
    system = (PROMPTS_DIR / "candidates_system.md").read_text(encoding="utf-8").format(max_candidates=max_candidates)
    user = (PROMPTS_DIR / "candidates_user.md").read_text(encoding="utf-8").format(
        bible=render_bible_compact(bible), previous_loglines=previous_loglines(episode, episodes),
        episode=episode.number, script=render_episode_compact(episode),
    )
    result = client.generate_json(
        system, user, CandidateList, f"candidates_E{episode.number:03d}",
        extra_validator=candidate_validator(episode, max_candidates),
    )
    values, events = postprocess_candidates(episode, score_candidates(episode, result), max_candidates)
    client.postprocess_events = getattr(client, "postprocess_events", []) + events
    return values


@dataclass
class PlannedBatch:
    episodes: list[Episode]
    system: str
    user: str
    estimated_tokens: int
    oversized: bool = False


def batch_prompts(episodes: list[Episode], all_episodes: list[Episode], bible: StoryBible | None, max_candidates: int) -> tuple[str, str]:
    system = (PROMPTS_DIR / "candidates_system.md").read_text(encoding="utf-8").format(max_candidates=max_candidates)
    user = (PROMPTS_DIR / "candidates_batch_user.md").read_text(encoding="utf-8").format(
        bible=render_bible_compact(bible) if bible else "[Story bible not built: reserve 600 estimated tokens]",
        previous_loglines=previous_loglines(episodes[0], all_episodes),
        scripts="\n\n".join(render_episode_compact(episode) for episode in episodes),
        episodes=", ".join(str(episode.number) for episode in episodes),
    )
    return system, user


def plan_batches(episodes: list[Episode], all_episodes: list[Episode], bible: StoryBible | None, batch_size: int, max_candidates: int, max_request_tokens: int | None) -> list[PlannedBatch]:
    """Pack consecutive episodes, then split batches that exceed the input guard."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    groups = []
    group = []
    for episode in episodes:
        if group and (len(group) == batch_size or episode.number != group[-1].number + 1):
            groups.append(group)
            group = []
        group.append(episode)
    if group:
        groups.append(group)
    plans = []
    def split(items):
        system, user = batch_prompts(items, all_episodes, bible, max_candidates)
        tokens = estimate_json_tokens(system, user, CandidateBatch) + (600 if bible is None else 0)
        if max_request_tokens and tokens > max_request_tokens and len(items) > 1:
            midpoint = len(items) // 2
            split(items[:midpoint])
            split(items[midpoint:])
        else:
            plans.append(PlannedBatch(items, system, user, tokens, bool(max_request_tokens and tokens > max_request_tokens)))
    for group in groups:
        split(group)
    return plans


def short_errors(errors: list[str]) -> str:
    return "; ".join(error.replace("\n", " ")[:180] for error in errors[:4])


def request_batch(plan: PlannedBatch, client: LLMClient, max_candidates: int) -> tuple[dict[int, CandidateList], dict[int, list[str]]]:
    """Call a batch once; recover valid episode results even if another is bad."""
    expected = {episode.number: episode for episode in plan.episodes}
    def validate(batch):
        errors = []
        numbers = [item.episode for item in batch.root]
        if set(numbers) != set(expected) or len(numbers) != len(expected):
            errors.append("[invalid_json] Return exactly one result for each requested episode, with no missing/duplicate/extra episode numbers")
        for item in batch.root:
            if item.episode in expected:
                errors.extend(candidate_validator(expected[item.episode], max_candidates)(item))
        return errors
    tag = "candidates_batch_" + "_".join(f"E{episode.number:03d}" for episode in plan.episodes)
    try:
        result = client.generate_json(plan.system, plan.user, CandidateBatch, tag, extra_validator=validate, max_repairs=0)
        raw_items = result.root
    except LLMOutputError as error:
        try:
            decoded = json.loads(_json_text(error.raw_text))
            raw_items = decoded if isinstance(decoded, list) else []
        except ValueError:
            raw_items = []
    valid, errors = {}, {}
    for number, episode in expected.items():
        matches = [item for item in raw_items if (item.episode if isinstance(item, CandidateList) else item.get("episode") if isinstance(item, dict) else None) == number]
        if len(matches) != 1:
            errors[number] = ["[invalid_json] Missing, duplicate, or malformed episode result"]
            continue
        try:
            item = matches[0] if isinstance(matches[0], CandidateList) else CandidateList.model_validate(matches[0])
        except ValueError as error:
            errors[number] = ["[invalid_json] Episode result does not match CandidateList schema"]
            continue
        failures = candidate_validator(episode, max_candidates)(item)
        if failures:
            errors[number] = failures
        else:
            valid[number] = item
    return valid, errors


def repair_episode(plan: PlannedBatch, episode: Episode, all_episodes: list[Episode], bible: StoryBible, client: LLMClient, max_candidates: int, errors: list[str]) -> CandidateList:
    """Re-ask only this episode once; do not replay other batch episodes."""
    user = (PROMPTS_DIR / "candidates_user.md").read_text(encoding="utf-8").format(
        bible=render_bible_compact(bible), previous_loglines=previous_loglines(plan.episodes[0], all_episodes),
        episode=episode.number, script=render_episode_compact(episode),
    )
    user += "\n\nYour previous output failed validation: " + short_errors(errors) + ". Return only valid JSON for this episode."
    client.repair_events.append({"tag": f"candidates_E{episode.number:03d}", "error_type": "HardCandidateError", "errors": errors})
    return client.generate_json(plan.system, user, CandidateList, f"candidates_repair_E{episode.number:03d}", extra_validator=candidate_validator(episode, max_candidates), max_repairs=0)


def dry_run_candidates(config: dict, selector: str, batch_size: int = 1, provider: str | None = None, model: str | None = None) -> dict:
    """Plan all selected episodes without clients, credentials, cache, or writes."""
    episodes = load_episodes(config.get("parsed_dir", "parsed"))
    selected = parse_episode_selector(selector, [episode.number for episode in episodes])
    bible_path = Path(config.get("output_dir", "outputs")) / "story_bible.json"
    bible = StoryBible.model_validate_json(bible_path.read_text(encoding="utf-8")) if bible_path.exists() else None
    chosen_provider, chosen_model = step_model_config(config, "candidates", provider, model)
    limit = request_token_limit(config, chosen_provider)
    plans = plan_batches([episode for episode in episodes if episode.number in selected], episodes, bible, batch_size, config.get("promo", {}).get("candidates_per_episode", 5), limit)
    executable = [plan for plan in plans if not plan.oversized]
    result = {"provider": chosen_provider, "model": chosen_model, "selected_episodes": selected, "batch_size": batch_size,
              "max_request_tokens": limit, "requests": len(executable), "estimated_input_tokens": sum(plan.estimated_tokens for plan in executable),
              "max_episode_repair_calls": sum(len(plan.episodes) for plan in executable),
              "oversized_episodes": [plan.episodes[0].number for plan in plans if plan.oversized],
              "batches": [{"episodes": [ep.number for ep in plan.episodes], "estimated_input_tokens": plan.estimated_tokens, "oversized": plan.oversized} for plan in plans]}
    print(f"DRY RUN | provider={chosen_provider} model={chosen_model} batch_size={batch_size}")
    print(f"Scope: all {len(selected)} selected episodes; resume skips are not deducted; no API calls or file writes")
    print(f"max_request_tokens={limit} | requests={result['requests']} | estimated_input_tokens={result['estimated_input_tokens']}")
    print(f"maximum_extra_episode_repair_calls={result['max_episode_repair_calls']} | oversized_episodes={result['oversized_episodes']}")
    print("Estimates include system, user, compact bible, previous loglines, scripts and JSON schema; exclude outputs, transport retries and optional repairs.")
    if bible is None:
        print("Bible not built: each request includes a 600-token planning reserve; bible generation is not included.")
    return result


def parse_episode_selector(selector: str, available: list[int]) -> list[int]:
    """Accept all, individual episodes, comma lists, and inclusive ranges."""
    if selector.strip().lower() == "all":
        return sorted(set(available))
    selected = set()
    for part in selector.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if match is None:
            raise ValueError(f"Invalid episode selector: {selector!r}")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) else first
        if first > last:
            raise ValueError("Episode range starts after it ends")
        if last - first > len(available):
            raise ValueError("Episode range includes unavailable episodes")
        selected.update(range(first, last + 1))
    missing = sorted(selected - set(available))
    if missing:
        raise ValueError(f"Episodes not found in parsed data: {missing}")
    return sorted(selected)


def _usage_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_saved_candidates(path: Path, episode: Episode, max_candidates: int) -> list[ScoredCandidate] | None:
    """Resume only when the saved schema, IDs, source text, and measurements agree."""
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        saved = [ScoredCandidate.model_validate(value) for value in values]
        result = CandidateList(episode=episode.number, candidates=saved)
        if candidate_validator(episode, max_candidates)(result):
            return None
        recalculated = score_candidates(episode, result)
        if [item.model_dump(exclude={"candidate_id"}) for item in recalculated] != [item.model_dump(exclude={"candidate_id"}) for item in saved]:
            return None
        ids = [item.candidate_id for item in saved]
        if len(set(ids)) != len(ids) or any(not re.fullmatch(f"E{episode.number:03d}_C[0-9]+", identifier) for identifier in ids):
            return None
        return saved
    except (OSError, ValueError, TypeError, KeyError):
        return None


def run_candidates(config: dict, selector: str, client: LLMClient | None = None, *, force: bool = False, hints: str | Path | None = None, batch_size: int = 1, max_api_calls: int | None = None, tag: str | None = None, budget: CallBudget | None = None, active_episodes: list[Episode] | None = None, force_bible: bool = False, progress_cb=None, provider_budgets=None, force_episodes=None) -> dict:
    """Use one client for context plus all episodes and continue after failures."""
    started = time.perf_counter()
    episodes = active_episodes if active_episodes is not None else load_episodes(config.get("parsed_dir", "parsed"))
    selected = parse_episode_selector(selector, [episode.number for episode in episodes])
    by_number = {episode.number: episode for episode in episodes}
    output = Path(config.get("output_dir", "outputs"))
    if tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Invalid source tag")
    directory = output / (f"candidates_{tag}" if tag else "candidates")
    directory.mkdir(parents=True, exist_ok=True)
    errors_path = output / "candidates_errors.log"
    errors_path.touch(exist_ok=True)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    provider, model = step_model_config(config, "candidates")
    strict = config.get("_interactive_run", False)
    client = client if client is not None else get_llm(config, provider, model)
    client.call_budget = budget if budget is not None else CallBudget(max_api_calls if max_api_calls is not None else config.get("llm", {}).get("max_api_calls_per_run", 14))
    if client.provider_name == "gemini":
        client.thinking_budget = config.get("models", {}).get("candidates", {}).get("thinking_budget", 1024)
    usage_start = len(_usage_rows(client.usage_log))
    repair_start = len(client.repair_events)
    bible_path = output / "story_bible.json"
    alias_hints = load_hints(hints)
    bible = None
    max_candidates = config.get("promo", {}).get("candidates_per_episode", 5)
    all_candidates = []
    successes, failures = [], []
    skipped, not_attempted = [], []
    stopped_quota = False
    stopped_budget = False
    stopped_network = False
    pending = []
    postprocess_events = []
    def save_result(number, values):
        values, events = postprocess_candidates(by_number[number], values, max_candidates)
        postprocess_events.extend(events)
        if events:
            with (output / "candidates_postprocess.log").open("a", encoding="utf-8") as file:
                for event in events:
                    file.write(json.dumps(event, ensure_ascii=False) + "\n")
                    print(f"Episode {number:03d}: CODE | {event['reason']}", flush=True)
        target = directory / f"episode_{number:03d}.json"
        target.write_text(json.dumps([item.model_dump() for item in values], ensure_ascii=False, indent=2), encoding="utf-8")
        all_candidates.extend(values)
        successes.append(number)
        if progress_cb:
            progress_cb({'stage':'candidates','state':'progress','episode':number,'done':len(successes),'total':len(selected)})
        return values

    def log_failure(number, error):
        failures.append(number)
        with errors_path.open("a", encoding="utf-8") as file:
            file.write(f"{datetime.now(timezone.utc).isoformat()} E{number:03d}: {error}\n")
        print(f"Episode {number:03d}: FAILED | {str(error).splitlines()[0]}", flush=True)

    for number in selected:
        episode = by_number[number]
        target = directory / f"episode_{number:03d}.json"
        saved = None if force or number in (force_episodes or set()) else load_saved_candidates(target, episode, max_candidates)
        if saved is not None:
            values = save_result(number, saved)
            skipped.append(number)
            print(f"Episode {number:03d}: SKIP | {len(values)} valid saved candidates", flush=True)
        else:
            pending.append(episode)
    if pending:
        if progress_cb:
            progress_cb({'stage':'bible','state':'running'})
        if bible_path.exists() and not force_bible:
            try:
                bible = StoryBible.model_validate_json(bible_path.read_text(encoding="utf-8"))
                if validate_bible(bible, episodes, alias_hints):
                    bible = None
                manifest = output / "story_bible_build.json"
                if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("identity_hints", {}) != alias_hints:
                    bible = None
                if manifest.exists():
                    source_key = json.loads(manifest.read_text(encoding="utf-8")).get("source_fingerprint")
                    if source_key and source_key != bible_source_fingerprint(episodes, alias_hints):
                        bible = None
            except (ValueError, OSError):
                bible = None
        if bible is None:
            bible_client = client
            if config.get("models", {}).get("bible"):
                bible_provider, bible_model = step_model_config(config, "bible")
                if (bible_provider, bible_model) != (client.provider_name, client.model):
                    bible_client = get_llm(config, bible_provider, bible_model)
            bible_client.call_budget = (provider_budgets or {}).get(bible_client.provider_name, client.call_budget)
            try:
                bible = build_story_bible(episodes, bible_client, output, alias_hints)
            except (LLMBudgetError, LLMNetworkError) as error:
                if strict:
                    raise
                stopped_budget = isinstance(error, LLMBudgetError)
                stopped_network = isinstance(error, LLMNetworkError)
                not_attempted = [ep.number for ep in pending]
                print(str(error), flush=True)
    if progress_cb:
        progress_cb({'stage':'bible','state':'complete' if not stopped_budget and not stopped_network else 'stopped'})
        progress_cb({'stage':'candidates','state':'running'})
    limit = request_token_limit(config, client.provider_name)
    pending.sort(key=lambda episode: episode.number in (force_episodes or set()))
    plans = [] if stopped_budget or stopped_network else plan_batches(pending, episodes, bible, batch_size, max_candidates, limit)
    while plans:
        plan = plans.pop(0)
        if plan.oversized:
            if strict:
                raise LLMRequestError("Episode exceeds the request-size limit.")
            log_failure(plan.episodes[0].number, f"Request-size guard: {plan.estimated_tokens} estimated tokens exceeds {limit}; episode was not sent")
            continue
        print(f"Batch {[ep.number for ep in plan.episodes]} | estimated_input_tokens={plan.estimated_tokens}", flush=True)
        try:
            results, hard_errors = request_batch(plan, client, max_candidates)
            # Save good results before repairing a different episode in the batch.
            for number, result in results.items():
                values = save_result(number, score_candidates(by_number[number], result))
                print(f"Episode {number:03d}: OK | {len(values)} candidates", flush=True)
            for number, errors in hard_errors.items():
                try:
                    result = repair_episode(plan, by_number[number], episodes, bible, client, max_candidates, errors)
                    values = save_result(number, score_candidates(by_number[number], result))
                    print(f"Episode {number:03d}: OK after one isolated repair | {len(values)} candidates", flush=True)
                except (LLMQuotaError, LLMBudgetError, LLMNetworkError):
                    raise
                except (LLMRequestError, LLMOutputError, ValueError, OSError) as error:
                    if strict:
                        raise
                    log_failure(number, error)
        except LLMBatchSplitError as error:
            print(f"Splitting failed five-episode batch into 3 + 2: {error}", flush=True)
            plans[0:0] = plan_batches(plan.episodes, episodes, bible, 3, max_candidates, limit)
            continue
        except (LLMBudgetError, LLMNetworkError) as error:
            if strict:
                raise
            stopped_budget = isinstance(error, LLMBudgetError)
            stopped_network = isinstance(error, LLMNetworkError)
            not_attempted = [value for value in selected if value not in successes and value not in failures]
            print(str(error), flush=True)
            break
        except LLMQuotaError as error:
            if strict:
                raise
            number = next((ep.number for ep in plan.episodes if ep.number not in successes and ep.number not in failures), plan.episodes[0].number)
            log_failure(number, error)
            not_attempted = [value for value in selected if value not in successes and value not in failures]
            stopped_quota = True
            break
        except (LLMRequestError, LLMOutputError, ValueError, OSError) as error:
            if strict:
                raise
            for episode in plan.episodes:
                if episode.number not in successes and episode.number not in failures:
                    log_failure(episode.number, error)
    all_candidates.sort(key=lambda item: item.candidate_id)
    successes.sort()
    fields = ["episode"] + list(ScoredCandidate.model_fields)
    with (output / "candidates_all.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for candidate in all_candidates:
            episode_number = int(candidate.candidate_id.split("_")[0][1:])
            writer.writerow({"episode": episode_number, **candidate.model_dump()})
    rows = _usage_rows(client.usage_log)[usage_start:]
    repair_counts = Counter()
    repair_events = client.repair_events[repair_start:]
    for event in repair_events:
        types = set()
        for error in event["errors"]:
            labels = re.findall(r"\[([a-z_]+)\]", error)
            types.update(labels or [event["error_type"]])
        repair_counts.update(types)
    durations = [candidate.duration_s for candidate in all_candidates]
    count = len(all_candidates)
    stats = {
        "selected_episodes": selected, "episodes_ok": successes, "episodes_failed": failures,
        "episodes_skipped": skipped, "episodes_not_attempted": not_attempted,
        "stopped_daily_quota": stopped_quota,
        "stopped_api_budget": stopped_budget,
        "stopped_network": stopped_network,
        "connection_errors": client.call_budget.connection_errors,
        "api_budget_used": client.call_budget.used,
        "api_budget_limit": client.call_budget.limit,
        "total_candidates": count,
        "candidates_per_episode_distribution": dict(Counter(sum(item.candidate_id.startswith(f"E{number:03d}_") for item in all_candidates) for number in selected)),
        "pct_episodes_ending_on_final_shot": 100 * sum(any(item.candidate_id.startswith(f"E{number:03d}_") and item.end_shot_id == by_number[number].shots[-1].id for item in all_candidates) for number in selected) / len(selected) if selected else 0,
        "avg_per_episode": count / len(selected) if selected else 0,
        "duration_min": min(durations) if count else None,
        "duration_median": median(durations) if count else None,
        "duration_max": max(durations) if count else None,
        "pct_duration_20_60": 100 * sum(20 <= value <= 60 for value in durations) / count if count else 0,
        "pct_hook_within_3": 100 * sum(candidate.hook_offset_s <= 3 for candidate in all_candidates) / count if count else 0,
        "pct_both_clean": 100 * sum(candidate.start_clean and candidate.end_clean for candidate in all_candidates) / count if count else 0,
        "hook_type_counts": dict(Counter(candidate.hook_type for candidate in all_candidates)),
        "cliffhanger_type_counts": dict(Counter(candidate.cliffhanger_type for candidate in all_candidates)),
        "repair_attempts": len(repair_events), "repairs_by_error_type": dict(repair_counts),
        "postprocess_events": postprocess_events,
        "tokens_in": sum(int(row["in_tokens"] or 0) for row in rows),
        "tokens_out": sum(int(row["out_tokens"] or 0) for row in rows),
        "api_attempts": client.call_budget.used,
        "cache_hits": sum(row["cached"] == "True" for row in rows),
        "total_time_s": round(time.perf_counter() - started, 3),
    }
    (output / "candidates_run_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_cb:
        progress_cb({'stage':'candidates','state':'complete' if not failures and not not_attempted else 'stopped','done':len(successes),'total':len(selected)})
    if stopped_budget or stopped_network:
        print(f"DONE episodes: {successes} | PENDING episodes: {not_attempted}", flush=True)
    print(f"TOTAL: {len(successes)} episodes OK | {len(failures)} failed | {count} candidates | "
          f"{stats['tokens_in']} / {stats['tokens_out']} tokens in/out | {stats['total_time_s']:.3f} s", flush=True)
    return stats


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--episodes", default="all")
    cli.add_argument("--config", type=Path, default=Path("config.yaml"))
    cli.add_argument("--provider")
    cli.add_argument("--model")
    cli.add_argument("--force", action="store_true", help="Regenerate valid episode files; LLM response caching still applies")
    cli.add_argument("--hints", type=Path, help="Optional identity hints YAML; omitted by default")
    cli.add_argument("--batch-size", type=int, default=1)
    cli.add_argument("--dry-run", action="store_true", help="Plan selected episodes with no API calls or writes")
    cli.add_argument("--max-api-calls", type=int, help="Maximum real attempts, including retries and repairs")
    cli.add_argument("--tag", help="Source tag for outputs/candidates_<tag>")
    args = cli.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        if args.dry_run:
            dry_run_candidates(config, args.episodes, args.batch_size, args.provider, args.model)
            return
        provider, model = step_model_config(config, "candidates", args.provider, args.model)
        client = get_llm(config, provider, model)
        stats = run_candidates(config, args.episodes, client, force=args.force, hints=args.hints, batch_size=args.batch_size, max_api_calls=args.max_api_calls, tag=args.tag or provider)
        if stats["stopped_api_budget"]:
            cli.exit(1, "Stopped at API call budget; saved outputs are ready to resume.\n")
        if stats["stopped_network"]:
            cli.exit(1, "network blocked — approve network or run locally\n")
        if stats["stopped_daily_quota"]:
            cli.exit(1, "Stopped immediately on daily quota; saved outputs are ready to resume.\n")
    except (OSError, ValueError, LLMRequestError, LLMOutputError) as error:
        cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
