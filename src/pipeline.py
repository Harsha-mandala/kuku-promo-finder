"""Run resumable parsing, generation, validation, judging and ranking."""

import argparse
import copy
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .candidates import load_saved_candidates, parse_episode_selector, plan_batches, run_candidates
from .context import PROMPTS_DIR, StoryOverview, IdentityBatch, bible_source_fingerprint, episode_metadata, validate_bible
from .judge import JudgeBatch, cap_promos, judge_fingerprint, plan_judge, run_judge
from .llm import CallBudget, LLMRequestError, LLMOutputError, LLMBudgetError, LLMQuotaError, LLMNetworkError, estimate_json_tokens, request_token_limit, step_model_config
from .models import Episode, StoryBible
from .parser import parse_file, episode_summary
from .rank import write_ranking
from .validate import ValidatedPromo, run_validation, validate_candidate, dedupe_promos
from .clips import clip_span
from .cliprender import render_judge_clip


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_inputs(input_dir, parsed_dir, *, dry_run=False, force=False, progress_cb=None):
    """Parse in memory during planning; reuse identical saved episodes."""
    paths = sorted(input_dir.glob("*.txt"))
    if not paths:
        raise ValueError(f"No *.txt inputs in {input_dir}")
    episodes = sorted((parse_file(path) for path in paths), key=lambda ep: ep.number)
    if len({ep.number for ep in episodes}) != len(episodes):
        raise ValueError("Duplicate episode numbers in input")
    changed = False
    for ep in episodes:
        if progress_cb:
            progress_cb({'stage':'parse','state':'progress','episode':ep.number,'total':len(episodes)})
        target = parsed_dir / f"episode_{ep.number:03d}.json"
        equal = False
        try:
            equal = Episode.model_validate_json(target.read_text(encoding="utf-8")) == ep
        except (OSError, ValueError):
            pass
        if (force or not equal) and not dry_run:
            parsed_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(ep.model_dump_json(indent=2), encoding="utf-8")
            changed = True
    if not dry_run and (changed or not (parsed_dir / "summary.csv").exists()):
        rows = [episode_summary(ep) for ep in episodes]
        with (parsed_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return episodes


def reusable_bible(output, episodes, force):
    if force:
        return None
    try:
        bible = StoryBible.model_validate_json((output / "story_bible.json").read_text(encoding="utf-8"))
        manifest_path = output / "story_bible_build.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_key = manifest.get("source_fingerprint")
            if source_key and source_key != bible_source_fingerprint(episodes):
                return None
        return bible if not validate_bible(bible, episodes) else None
    except (OSError, ValueError):
        return None


def plan_bible(episodes, config, provider, bible):
    """Estimate missing-bible requests with a reserve for unknown overview text."""
    if bible is not None:
        return {"requests": 0, "tokens": 0, "note": "reuse validated saved bible"}
    system = (PROMPTS_DIR / "story_bible.md").read_text(encoding="utf-8")
    user = episode_metadata(episodes)
    full = estimate_json_tokens(system, user, StoryBible)
    settings = config.get("llm", {})
    tpm = settings.get("providers", {}).get(provider, {}).get("tokens_per_minute", settings.get("tokens_per_minute"))
    limits = [v for v in (request_token_limit(config, provider), max(1, tpm - 1500) if tpm else None) if v]
    if not limits or full <= min(limits):
        return {"requests": 1, "tokens": full, "note": "one complete bible request"}
    payload = json.loads(user)
    metadata = [{k: v for k, v in item.items() if k != "speakers"} for item in payload["episodes"]]
    overview_system = "Summarize only these episode loglines/notes into a factual overview with episode-range arcs. Never invent names or titles."
    tokens = estimate_json_tokens(overview_system, json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), StoryOverview)
    size = min(14000, max(2000, (tpm or 8000) * 4 - 12000))
    groups, group = [], []
    for item in payload["speaker_evidence"]:
        if group and len(json.dumps(group + [item], ensure_ascii=False)) > size:
            groups.append(group)
            group = []
        group.append(item)
    if group:
        groups.append(group)
    for group in groups:
        tokens += estimate_json_tokens("Infer speaker identities from supplied source evidence.", json.dumps(group, ensure_ascii=False), IdentityBatch) + 800
    return {"requests": 1 + len(groups), "tokens": tokens, "note": "overview + evidence; includes approximate 800-token common-context reserve per evidence batch"}


def plan_pipeline(config, tag, episodes, selected, batch_size, force=False, force_episodes=None, no_trailers=False, no_editor=False):
    """Plan future stages without creating clients or writing files."""
    output = Path(config.get("output_dir", "outputs"))
    cp, cm = step_model_config(config, "candidates")
    bp, bm = step_model_config(config, "bible")
    jp, jm = step_model_config(config, "judge")
    bible = reusable_bible(output, episodes, force)
    bible_plan = plan_bible(episodes, config, bp, bible)
    max_candidates = config.get("promo", {}).get("candidates_per_episode", 5)
    pending, actual_promos = [], []
    for ep in selected:
        source = output / f"candidates_{tag}/episode_{ep.number:03d}.json"
        saved = None if force or ep.number in (force_episodes or set()) else load_saved_candidates(source, ep, max_candidates)
        if saved is None:
            pending.append(ep)
        else:
            actual_promos.extend(dedupe_promos([validate_candidate(ep, row.model_dump(), i) for i, row in enumerate(saved, 1)]))
    pending.sort(key=lambda episode: episode.number in (force_episodes or set()))
    candidate_plans = plan_batches(pending, episodes, bible, batch_size, max_candidates, request_token_limit(config, cp))
    # Candidate scripts do not exist yet. Size future judge input using the
    # largest <=60s window per episode, up to the configured candidate ceiling.
    future = []
    for ep in pending:
        largest = None
        length = -1
        for first, shot in enumerate(ep.shots):
            for last in range(first, len(ep.shots)):
                a, b = clip_span(ep, shot.id, ep.shots[last].id)
                if 20 <= b - a <= 60:
                    item = ValidatedPromo(episode=ep.number, id=f"PLAN_E{ep.number:03d}", status="ok", start_shot_id=shot.id,
                                          end_shot_id=ep.shots[last].id, final_span=(a, b), duration_s=b-a)
                    size = len(render_judge_clip(ep, item))
                    if size > length:
                        largest, length = item, size
        if largest:
            future.extend(largest.model_copy(update={"id": f"PLAN_E{ep.number:03d}_C{i:02d}"}) for i in range(1, min(3, max_candidates) + 1))
    eligible, capped = cap_promos([p for p in actual_promos if p.status != "rejected"])
    skipped_judge = set()
    if not force:
        try:
            scores = JudgeBatch.model_validate_json((output / f"judged_{tag}/scores.json").read_text(encoding="utf-8")).root
            old_inputs = json.loads((output / f"judged_{tag}/score_inputs.json").read_text(encoding="utf-8"))
            have = {score.promo_id for score in scores}
            ep_lookup = {ep.number: ep for ep in episodes}
            skipped_judge = {p.id for p in eligible if p.id in have and old_inputs.get(p.id) == judge_fingerprint(ep_lookup[p.episode], p, jp, jm, config)}
        except (OSError, ValueError):
            pass
    judge_promos = [p for p in eligible if p.id not in skipped_judge] + future
    judge_plans = plan_judge(judge_promos, {ep.number: ep for ep in episodes}, config, jp, 3)
    result = {"bible": bible_plan, "candidate_requests": sum(not p.oversized for p in candidate_plans),
              "candidate_tokens": sum(p.estimated_tokens for p in candidate_plans if not p.oversized),
              "candidate_oversized": [p.episodes[0].number for p in candidate_plans if p.oversized],
              "judge_promos": len(judge_promos), "judge_requests": sum(not p["oversized"] for p in judge_plans),
              "judge_tokens": sum(p["tokens"] for p in judge_plans if not p["oversized"]),
              "judge_oversized": [p["promos"][0].id for p in judge_plans if p["oversized"]]}
    print(f"PIPELINE DRY RUN | {len(selected)} episodes | tag={tag} | no API calls or writes")
    print(f"BIBLE {bp}/{bm}: requests={bible_plan['requests']} estimated_input_tokens={bible_plan['tokens']} | {bible_plan['note']}")
    print(f"CANDIDATES {cp}/{cm}: batch_size={batch_size} requests={result['candidate_requests']} estimated_input_tokens={result['candidate_tokens']} | {len(pending)} pending episodes | oversized={result['candidate_oversized']}")
    print(f"JUDGE {jp}/{jm}: batch_size=3 requests={result['judge_requests']} estimated_input_tokens={result['judge_tokens']} | up to {len(judge_promos)} promos | skipped={len(skipped_judge)} | oversized={result['judge_oversized']}")
    print("Judge plan sizes future candidates by largest <=60s compact script windows, at most 3 promos per episode; actual validation/dedupe/safety/resume may reduce it. This is an estimate, not generated candidates.")
    print(f"Shared call budget={config.get('llm', {}).get('max_api_calls_per_run', 14)}; estimates exclude repairs, retries and output tokens. Resume after a budget stop.")
    from .editorial_plan import plan_editorial
    extras = plan_editorial(config, actual_promos + future, {ep.number:ep for ep in episodes}, no_trailers=no_trailers, no_editor=no_editor)
    result.update(extras)
    stages = [{'stage':'bible','provider':bp,'requests':bible_plan['requests'],'tokens':bible_plan['tokens']},
              {'stage':'candidates','provider':cp,'requests':result['candidate_requests'],'tokens':result['candidate_tokens']},
              {'stage':'judge','provider':jp,'requests':result['judge_requests'],'tokens':result['judge_tokens']}] + extras['editorial_stages']
    result['stages'] = stages
    totals = {}
    for row in stages:
        value = totals.setdefault(row['provider'], {'requests':0,'tokens':0})
        value['requests'] += row['requests']; value['tokens'] += row['tokens']
    result['provider_totals'] = totals
    print('PER-PROVIDER TOTALS: '+json.dumps(totals,sort_keys=True))
    return result


def run_pipeline(config, input_dir, tag, overrides=None, progress_cb=None, *, selector="all", batch_size=5, max_api_calls=None, dry_run=False, force=False, force_episodes=None, no_trailers=False, no_editor=False):
    config = copy.deepcopy(config)
    overrides = overrides or {}
    selector = overrides.get('episodes', selector)
    batch_size = overrides.get('batch_size', batch_size)
    max_api_calls = overrides.get('max_api_calls', max_api_calls)
    dry_run = overrides.get('dry_run', dry_run)
    force = overrides.get('force', force)
    no_trailers = overrides.get('no_trailers',no_trailers)
    no_editor = overrides.get('no_editor',no_editor)
    caller_cb=progress_cb
    def progress_cb(event):
        if not dry_run:
            event_dir=Path(config.get('output_dir','outputs'))
            event_dir.mkdir(parents=True,exist_ok=True)
            with (event_dir/f'pipeline_{tag}_events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),**event})+'\n')
        if caller_cb:
            caller_cb(event)
    for flag, step in (('candidate','candidates'),('judge','judge')):
        provider, model = step_model_config(config, step, overrides.get(flag+'_provider'), overrides.get(flag+'_model'))
        config.setdefault('models', {}).setdefault(step, {}).update(provider=provider, model=model)
    def emit(stage, state, **details):
        if progress_cb:
            progress_cb(dict(stage=stage,state=state,**details))
    output = Path(config.get("output_dir", "outputs"))
    parsed = Path(config.get("parsed_dir", "parsed"))
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
        raise ValueError("Invalid source tag")
    if max_api_calls is not None:
        config.setdefault("llm", {})["max_api_calls_per_run"] = max_api_calls
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    emit('parse','running')
    episodes = parse_inputs(input_dir, parsed, dry_run=dry_run, force=force, progress_cb=progress_cb)
    emit('parse','complete',episodes=len(episodes))
    if overrides.get('hosted_keys'):
        if len(episodes)>20:
            raise ValueError('Hosted keys allow at most 20 episodes per run.')
        config.setdefault('llm', {})['max_api_calls_per_run'] = min(14, config.get('llm', {}).get('max_api_calls_per_run',14))
        config['llm'].pop('provider_budgets',None)
    numbers = parse_episode_selector(selector, [ep.number for ep in episodes])
    selected = [ep for ep in episodes if ep.number in numbers]
    forced = set(parse_episode_selector(force_episodes,[ep.number for ep in episodes])) if force_episodes else set()
    forced &= set(numbers)
    if dry_run:
        if config.get('llm',{}).get('provider_budgets'):
            print('Independent provider budgets: '+json.dumps(config['llm']['provider_budgets'],sort_keys=True))
        return plan_pipeline(config, tag, episodes, selected, batch_size, force, forced, no_trailers, no_editor)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"pipeline_{tag}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    budget = CallBudget(config.get("llm", {}).get("max_api_calls_per_run", 14))
    provider_budgets = {name:CallBudget(limit) for name,limit in config.get('llm',{}).get('provider_budgets',{}).items()}
    cp,_ = step_model_config(config,'candidates')
    jp,_ = step_model_config(config,'judge')
    stats = run_candidates(config, selector, force=force, batch_size=batch_size, tag=tag, budget=provider_budgets.get(cp,budget), active_episodes=episodes, force_bible=force, progress_cb=progress_cb, provider_budgets=provider_budgets, force_episodes=forced)
    candidate_budget_stop=stats['stopped_api_budget']
    if stats["stopped_daily_quota"] or stats["stopped_network"] or (candidate_budget_stop and (not provider_budgets or not stats['episodes_ok'])):
        emit('candidates','stopped',daily_quota=stats['stopped_daily_quota'],budget=stats['stopped_api_budget'],network=stats['stopped_network'])
        return {"stopped": "candidates", "stats": stats}
    source = output / f"candidates_{tag}"
    active_sources = [source / f"episode_{ep.number:03d}.json" for ep in selected if (source / f"episode_{ep.number:03d}.json").exists()]
    inputs = active_sources + [parsed / f"episode_{ep.number:03d}.json" for ep in episodes]
    validation_key = fingerprint(inputs + [Path(__file__).parent / "validate.py"])
    targets = [output / f"validated_{tag}" / p.name for p in active_sources]
    if force or manifest.get("validation") != validation_key or not targets or not all(p.exists() for p in targets):
        emit('validate','running')
        run_validation(source, tag, parsed, output, episode_numbers=set(numbers))
        manifest["validation"] = validation_key
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        print("VALIDATE: SKIP unchanged inputs")
    emit('validate','complete')
    emit('judge','running')
    judged = run_judge(config, tag, force=force, budget=provider_budgets.get(jp,budget), episode_numbers=set(numbers), progress_cb=progress_cb)
    emit('judge','stopped' if judged['pending'] else 'complete',pending=len(judged['pending']),reason=judged['stopped'])
    rank_inputs = targets + list((output / f"judged_{tag}").glob("scores.json")) + [Path(__file__).parent / "rank.py"]
    rank_key = fingerprint(rank_inputs) + json.dumps([config.get("rank", {}), numbers], sort_keys=True)
    if force or manifest.get("rank") != rank_key or not (output / f"final_{tag}/report.md").exists():
        emit('rank','running')
        write_ranking(config, tag, episode_numbers=set(numbers))
        manifest["rank"] = rank_key
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        print("RANK: SKIP unchanged scores and rules")
    emit('rank','complete')
    from .safety_review import write_safety_review
    extras = {}
    if not judged['pending']:
        from .trailers import run_trailers
        from .editor_pass import run_editor
        for stage,disabled,runner in [('trailers',no_trailers,run_trailers),('editor',no_editor,run_editor)]:
            if disabled or not config.get(stage,{}).get('enabled',False):
                continue
            emit(stage,'running')
            provider,_ = step_model_config(config,stage)
            try:
                options = {'budget':provider_budgets.get(provider,budget),'force':force}
                if stage == 'trailers':
                    options['judge_budget'] = provider_budgets.get(jp,budget)
                extras[stage] = runner(config,tag,**options)
                emit(stage,'complete')
            except (LLMBudgetError,LLMQuotaError,LLMNetworkError,LLMRequestError,LLMOutputError) as error:
                emit(stage,'stopped',reason=str(error))
                write_safety_review(config, tag)
                return {'candidate_stats':stats,'judge_stats':judged,'editorial':extras,'stopped':stage,'error':str(error),
                        'api_calls':budget.used+sum(b.used for b in provider_budgets.values())}
    write_safety_review(config, tag)
    return {"candidate_stats": stats, "judge_stats": judged, "editorial":extras, "api_calls": budget.used+sum(b.used for b in provider_budgets.values()), "stopped": "judge" if judged["pending"] else 'candidates' if candidate_budget_stop else None}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, default=Path("data"))
    cli.add_argument("--tag", required=True)
    cli.add_argument("--episodes", default="all")
    cli.add_argument("--config", type=Path, default=Path("config.yaml"))
    for stage in ("candidate", "judge"):
        cli.add_argument(f"--{stage}-provider")
        cli.add_argument(f"--{stage}-model")
    cli.add_argument("--batch-size", type=int, default=5)
    cli.add_argument("--max-api-calls", type=int)
    cli.add_argument('--max-calls-gemini',type=int)
    cli.add_argument('--max-calls-groq',type=int)
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--force", action="store_true")
    cli.add_argument("--force-episodes",help="Regenerate only these candidate episodes while reusing the bible")
    cli.add_argument("--no-trailers",action="store_true")
    cli.add_argument("--no-editor",action="store_true")
    args = cli.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for provider in ('gemini','groq'):
        limit=getattr(args,'max_calls_'+provider)
        if limit is not None:
            if limit<0:
                cli.error('Provider call limits must be non-negative')
            config.setdefault('llm',{}).setdefault('provider_budgets',{})[provider]=limit
    for flag, step in (("candidate", "candidates"), ("judge", "judge")):
        provider, model = step_model_config(config, step, getattr(args, flag + "_provider"), getattr(args, flag + "_model"))
        config.setdefault("models", {}).setdefault(step, {}).update(provider=provider, model=model)
    try:
        result = run_pipeline(config, args.input, args.tag, selector=args.episodes, batch_size=args.batch_size,
                              max_api_calls=args.max_api_calls, dry_run=args.dry_run, force=args.force,force_episodes=args.force_episodes,no_trailers=args.no_trailers,no_editor=args.no_editor)
        if result.get("stopped"):
            cli.exit(1, f"Pipeline stopped in {result['stopped']}; progress saved.\n")
    except (OSError, ValueError, LLMRequestError, LLMOutputError) as error:
        cli.exit(1, f"{error}\n")
