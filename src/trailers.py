"""Build, validate and independently judge source-grounded multi-segment trailers."""

import re
from pathlib import Path

from pydantic import BaseModel, Field, RootModel

from .cliprender import render_judge_clip
from .clips import clip_span
from .context import load_episodes
from .editorial_io import digest, read, stage_client, write, write_csv, single_source_key
from .judge import JudgeScore
from .llm import LLMOutputError, LLMRequestError, estimate_json_tokens, request_token_limit, step_model_config
from .models import Trailer, TrailerBatch, TrailerSegment
from .rank import rank_promos
from .timeutils import format_timestamp
from .validate import ValidatedPromo, is_dead_shot

PROMPTS = Path(__file__).parent / "prompts"


class FinalSegment(BaseModel):
    episode: int
    start_shot_id: str
    end_shot_id: str
    start: float
    end: float
    clean_start: bool
    clean_end: bool


class ValidatedTrailer(BaseModel):
    id: str
    segments: list[FinalSegment] = Field(default_factory=list)
    duration_s: float = 0
    hook_line_id: str
    cliffhanger_line_id: str
    hook_text: str = ""
    cliffhanger_text: str = ""
    story: str
    why_it_works: str
    spoiler_risk: str
    self_score: int
    status: str = "rejected"
    flags: list[str] = Field(default_factory=list)
    fix_log: list[str] = Field(default_factory=list)
    original: dict = Field(default_factory=dict)


class TrailerJudgeScore(JudgeScore):
    coherence: int = Field(ge=1, le=5)


class TrailerJudgeBatch(RootModel[list[TrailerJudgeScore]]):
    pass


def pool_promos(ranked, config):
    maximum = config.get("trailers",{}).get("max_episode",15)
    eligible = [p for p in ranked if p["status"] in ("ok","fixed","flagged") and p["ad_safety"] != "unsafe"]
    return sorted(eligible,key=lambda p:(p["episode"] > maximum,p["rank"]))[:20]


def shot_selection(segment, episodes):
    match = re.fullmatch(r"E(\d+)_S\d+",segment.start_shot_id)
    if not match or int(match[1]) not in episodes:
        raise ValueError("unknown start shot ID")
    episode = episodes[int(match[1])]
    positions = {shot.id:i for i,shot in enumerate(episode.shots)}
    if segment.end_shot_id not in positions or segment.start_shot_id not in positions:
        raise ValueError("unknown end shot ID or cross-episode segment")
    first,last = positions[segment.start_shot_id],positions[segment.end_shot_id]
    if first > last:
        raise ValueError("start after end")
    return episode,first,last


def source(episode, first, last, source_id):
    for i in range(first,last+1):
        shot = episode.shots[i]
        if not shot.lines and shot.id == source_id:
            return i,shot.effective_start,shot.effective_end,shot.camera_text
        for line in shot.lines:
            if line.id == source_id:
                if line.type not in ('dialogue','action'):
                    raise ValueError(f"line ID {source_id} was not supplied in the dialogue/action pool")
                return i,line.start,line.end,line.text
    raise ValueError(f"unknown or outside segment line ID {source_id}")


def allowed_shots(pool, episodes):
    result = set()
    for row in pool:
        for span in row["segments"]:
            ep = episodes[span["episode"]]
            result.update(s.id for s in ep.shots if s.effective_start >= span["start"]-.001 and s.effective_end <= span["end"]+.001)
    return result


def hard_errors(trailer, episodes, allowed=None):
    errors = []
    for i,segment in enumerate(trailer.segments):
        try:
            ep,first,last = shot_selection(segment,episodes)
            if allowed is not None and any(s.id not in allowed for s in ep.shots[first:last+1]):
                raise ValueError("shot range includes IDs outside the supplied pool")
            if i == 0:
                source(ep,first,last,trailer.hook_line_id)
            if i == len(trailer.segments)-1:
                source(ep,first,last,trailer.cliffhanger_line_id)
        except ValueError as error:
            code = "unknown_line_id" if "line ID" in str(error) else "unknown_shot_id"
            errors.append(f"[{code}] segment {i+1}: {error}")
    return errors


def validate_trailer(trailer, episodes, index=1, allowed=None, *, min_gap_s=10):
    """Fix whole-shot boundaries without applying a 20-second minimum per segment."""
    result = ValidatedTrailer(id=f"T{index:03d}",hook_line_id=trailer.hook_line_id,
        cliffhanger_line_id=trailer.cliffhanger_line_id,story=trailer.story,why_it_works=trailer.why_it_works,
        spoiler_risk=trailer.spoiler_risk,self_score=trailer.self_score,original=trailer.model_dump())
    errors = hard_errors(trailer,episodes,allowed)
    if errors:
        result.fix_log.extend(errors)
        return result
    ranges = []
    for i,segment in enumerate(trailer.segments):
        ep,first,last = shot_selection(segment,episodes)
        if i == 0:
            hook = source(ep,first,last,trailer.hook_line_id)
            if hook[1]-ep.shots[first].effective_start > 3:
                first = hook[0]
                result.fix_log.append("start moved to hook shot")
        # Existing single-clip boundary rules: at most two earlier start shots,
        # then the next clean end or the preceding clean end. Never add unseen IDs.
        for _ in range(2):
            if ep.shots[first].clean_start or first == 0 or (allowed is not None and ep.shots[first-1].id not in allowed):
                break
            first -= 1
            result.fix_log.append(f"segment {i+1}: unsafe start moved back one shot")
        while last > first and is_dead_shot(ep.shots[last]):
            last -= 1
            result.fix_log.append(f"segment {i+1}: dropped trailing dead shot")
        if not ep.shots[last].clean_end:
            if last+1 < len(ep.shots) and ep.shots[last+1].clean_end and not is_dead_shot(ep.shots[last+1]) and (allowed is None or ep.shots[last+1].id in allowed):
                last += 1
                result.fix_log.append(f"segment {i+1}: end extended to next clean shot")
            elif last > first and ep.shots[last-1].clean_end:
                last -= 1
                result.fix_log.append(f"segment {i+1}: end pulled back to previous clean shot")
        ranges.append([ep,first,last])

    def duration(items):
        return sum(clip_span(ep,ep.shots[a].id,ep.shots[b].id)[1]-clip_span(ep,ep.shots[a].id,ep.shots[b].id)[0] for ep,a,b in items)

    total = duration(ranges)
    ep,first,last = ranges[-1]
    if total > 60 or total < 20:
        candidates = range(last-1,first-1,-1) if total > 60 else range(last+1,len(ep.shots))
        for end in candidates:
            if not ep.shots[end].clean_end or is_dead_shot(ep.shots[end]):
                continue
            if allowed is not None and any(s.id not in allowed for s in ep.shots[first:end+1]):
                continue
            try:
                source(ep,first,end,trailer.cliffhanger_line_id)
            except ValueError:
                continue
            proposed = ranges[:-1]+[[ep,first,end]]
            if 20 <= duration(proposed) <= 60 and ep.shots[end].effective_end-ep.shots[first].effective_start >= 3:
                ranges[-1] = [ep,first,end]
                result.fix_log.append("last segment trimmed within 60 s" if total > 60 else "last segment extended to meet 20 s")
                break
    try:
        first_ep,a,b = ranges[0]
        hook = source(first_ep,a,b,trailer.hook_line_id)
        final_ep,a,b = ranges[-1]
        cliff = source(final_ep,a,b,trailer.cliffhanger_line_id)
        if not 0 <= hook[1]-first_ep.shots[ranges[0][1]].effective_start <= 3:
            raise ValueError("hook offset outside first three seconds")
        result.hook_text,result.cliffhanger_text = hook[3],cliff[3]
        for ep,first,last in ranges:
            start,end = clip_span(ep,ep.shots[first].id,ep.shots[last].id)
            if end-start < 3 or start < 0 or end > ep.duration:
                raise ValueError("segment duration below 3 s or outside episode")
            if any(s.episode == ep.number and min(s.end,end) > max(s.start,start) for s in result.segments):
                raise ValueError("overlapping segments")
            for previous in result.segments:
                if previous.episode == ep.number:
                    gap = max(start - previous.end, previous.start - end)
                    if gap + 1e-6 < min_gap_s:
                        raise ValueError(f"same-episode segment gap {gap:g} s below {min_gap_s:g} s")
            result.segments.append(FinalSegment(episode=ep.number,start_shot_id=ep.shots[first].id,end_shot_id=ep.shots[last].id,
                start=start,end=end,clean_start=ep.shots[first].clean_start,clean_end=ep.shots[last].clean_end))
            if not ep.shots[first].clean_start or not ep.shots[last].clean_end:
                result.flags.append(f"segment {len(result.segments)} unsafe boundary")
        result.duration_s = round(duration(ranges),3)
        if not 20 <= result.duration_s <= 60:
            raise ValueError("total duration outside 20–60 s")
        result.status = "flagged" if result.flags else "fixed" if result.fix_log else "ok"
    except ValueError as error:
        result.fix_log.append(f"rejected: {error}")
    return result


def render_segment(ep, start_id, end_id, *, include_final_sound=False):
    _,first,last = shot_selection(TrailerSegment(start_shot_id=start_id,end_shot_id=end_id),{ep.number:ep})
    rows = []
    for shot in ep.shots[first:last+1]:
        row = f"[{shot.id}] {format_timestamp(shot.effective_start)}–{format_timestamp(shot.effective_end)}"
        if not shot.lines:
            row += " CAM: "+shot.camera_text[:60]
        rows.append(row)
        for line in shot.lines:
            if line.type in ("dialogue","action") or (include_final_sound and shot.id==end_id and line.type=='sound'):
                label = line.speaker or line.type.upper()
                if line.emotion:
                    label += f" ({line.emotion})"
                rows.append(f"[{line.id}] {label} {format_timestamp(line.start)}: {line.text}")
    return "\n".join(rows)


def pool_prompt(pool,episodes):
    rows = []
    for row in pool:
        rows.append("POOL "+row["promo_id"])
        for span in row["segments"]:
            ep = episodes[span["episode"]]
            shots = [s for s in ep.shots if s.effective_start >= span["start"]-.001 and s.effective_end <= span["end"]+.001]
            if shots:
                rows.append(render_segment(ep,shots[0].id,shots[-1].id))
    return "\n".join(rows)


def render_trailer(trailer,episodes):
    parts = []
    for segment in trailer.segments:
        promo = ValidatedPromo(episode=segment.episode,id=trailer.id,start_shot_id=segment.start_shot_id,
            end_shot_id=segment.end_shot_id,final_span=(segment.start,segment.end),duration_s=segment.end-segment.start)
        parts.append(render_judge_clip(episodes[segment.episode],promo))
    return f"TRAILER {trailer.id} | TOTAL DURATION: {trailer.duration_s:.2f} s\n"+"\n--- CUT ---\n".join(parts)


def rank_trailers(trailers,scores,config):
    rows = []
    by_id = {s.promo_id:s for s in scores}
    for trailer in trailers:
        score = by_id.get(trailer.id)
        if trailer.status == "rejected" or score is None or score.ad_safety == "unsafe":
            continue
        proxy = ValidatedPromo(episode=trailer.segments[0].episode,id=trailer.id,status=trailer.status,
            final_span=(0,trailer.duration_s),duration_s=trailer.duration_s,start_clean=all(s.clean_start for s in trailer.segments),
            end_clean=all(s.clean_end for s in trailer.segments),hook_text=trailer.hook_text,cliffhanger_text=trailer.cliffhanger_text,
            why_it_works=trailer.why_it_works,flags=trailer.flags,fix_log=trailer.fix_log)
        row = rank_promos([proxy],[score],config)["ranked"][0]
        row.update(type="trailer",segments=[s.model_dump() for s in trailer.segments],story=trailer.story,spoiler_risk=trailer.spoiler_risk)
        row["final_score"] += .10 * (score.coherence-1)/4
        row["score_adj"] = row["final_score"]
        row["segment_text"] = " → ".join(f"E{s.episode:02d} {format_timestamp(s.start)}–{format_timestamp(s.end)}" for s in trailer.segments)
        rows.append(row)
    ordered,counts = [],{}
    penalty = config.get('rank',{}).get('same_episode_penalty',.05)
    while rows:
        for row in rows:
            row['score_adj']=row['final_score']-penalty*counts.get(row['episode'],0)
        row=min(rows,key=lambda r:(-r['score_adj'],-r['cliffhanger_score'],r['duration_s'],r['promo_id']))
        rows.remove(row)
        row['rank']=len(ordered)+1
        ordered.append(row)
        counts[row['episode']]=counts.get(row['episode'],0)+1
    return ordered


def run_trailers(config,tag,*,budget=None,judge_budget=None,client=None,judge_client=None,force=False):
    output = Path(config.get("output_dir","outputs"))
    directory,final = output/f"trailers_{tag}",output/f"final_{tag}"
    episodes = {ep.number:ep for ep in load_episodes(config.get("parsed_dir","parsed"))}
    pool = pool_promos(read(final/"promos_ranked.json",[]),config)
    settings = config.get("trailers", {})
    count = settings.get("count", 5)
    min_gap_s = settings.get("min_gap_s", 10)
    system = (PROMPTS/"trailers_system.md").read_text(encoding="utf-8").format(count=count, min_gap_s=min_gap_s)
    user = pool_prompt(pool,episodes)
    key = digest([system,user,config.get("models",{}).get("trailers")])
    manifest = read(directory/"manifest.json",{})
    if not pool:
        return {"trailers":[],"empty_pool":True}
    allowed = allowed_shots(pool,episodes)
    if not force and manifest.get("proposals") == key and (directory/"proposals.json").exists():
        proposals = TrailerBatch.model_validate(read(directory/"proposals.json"))
    else:
        client = client or stage_client(config,"trailers",budget)
        if budget is not None:
            client.call_budget = budget
        def check(batch):
            return [error for t in batch.root for error in hard_errors(t,episodes,allowed)]
        proposals = client.generate_json(system,user,TrailerBatch,f"trailers_{tag}",extra_validator=check,max_repairs=1)
        write(directory/"proposals.json",proposals.model_dump())
        manifest["proposals"] = key
        write(directory/"manifest.json",manifest)
    validated = [validate_trailer(t, episodes, i, allowed, min_gap_s=min_gap_s)
                 for i, t in enumerate(proposals.root[:count], 1)]
    write(directory/"validated.json",[t.model_dump() for t in validated])
    eligible = [t for t in validated if t.status != "rejected"]
    judge_system = (PROMPTS/"judge_system.md").read_text(encoding="utf-8").replace("Clips in one request come from different episodes.","Trailers may use several episodes; judge each independently.")+"\nFor trailers also score coherence 1–5: does the sequence read as ONE story? Return one score per trailer."
    judge_user = "\n\n".join(render_trailer(t,episodes) for t in eligible)
    judge_key = digest([judge_system,judge_user,config.get("models",{}).get("judge")])
    if not force and manifest.get("judge") == judge_key and (directory/"scores.json").exists():
        scores = TrailerJudgeBatch.model_validate(read(directory/"scores.json")).root
    else:
        scores = []
    known = {s.promo_id for s in scores}
    pending = [t for t in eligible if t.id not in known]
    if pending:
        judge_client = judge_client or stage_client(config,"judge",judge_budget)
        if judge_budget is not None:
            judge_client.call_budget = judge_budget
        judge_client.temperature = 0
        limit = request_token_limit(config,judge_client.provider_name)
        queue = [pending]
        while queue:
            batch = queue.pop(0)
            user = "\n\n".join(render_trailer(t,episodes) for t in batch)
            if limit and estimate_json_tokens(judge_system,user,TrailerJudgeBatch) > limit and len(batch)>1:
                midpoint = len(batch)//2
                queue[:0] = [batch[:midpoint],batch[midpoint:]]
                continue
            expected = {t.id for t in batch}
            def check_scores(answer):
                ids = [s.promo_id for s in answer.root]
                return [] if len(ids)==len(expected) and set(ids)==expected else ["[invalid_json] Return every trailer ID exactly once"]
            try:
                new_scores = judge_client.generate_json(judge_system,user,TrailerJudgeBatch,f"judge_trailers_{tag}",temperature=0,
                                                       extra_validator=check_scores,max_repairs=1).root
            except (LLMRequestError, LLMOutputError) as error:
                if judge_client.provider_name in {'groq','openai','ollama'} and getattr(error, 'status', None) == 413 and len(batch) > 1:
                    queue[:0] = [[item] for item in batch]
                    continue
                error.stage = 'trailer judge'
                error.episodes = sorted({segment.episode for item in batch for segment in item.segments})
                raise
            for score in new_scores:
                trailer = next(t for t in eligible if t.id == score.promo_id)
                line_ids = {line.id for s in trailer.segments for shot in episodes[s.episode].shots
                            if shot.effective_start >= s.start-.001 and shot.effective_end <= s.end+.001 for line in shot.lines}
                score.ad_safety_verified = score.ad_safety == "safe" or bool(line_ids.intersection(re.findall(r"E\d+_L\d+",score.ad_safety_reason)))
            scores.extend(new_scores)
            write(directory/"scores.json",[s.model_dump() for s in scores])
            manifest["judge"] = judge_key
            write(directory/"manifest.json",manifest)
    ranked = rank_trailers(eligible,scores,config)
    excluded = {"rejected":[t.model_dump() for t in validated if t.status=='rejected'],
                "unsafe":[s.model_dump() for s in scores if s.ad_safety=='unsafe']}
    write(final/"trailer_exclusions.json",excluded)
    write(final/"trailers_ranked.json",ranked)
    manifest['source_key'] = single_source_key(output,tag,config.get('parsed_dir','parsed'))
    write(directory/'manifest.json',manifest)
    write_csv(final/"trailers_ranked.csv",ranked)
    singles = [dict(row,type="single") for row in read(final/"promos_ranked.json",[])]
    write_csv(final/"promos_all.csv",singles+ranked)
    report = (final/"report.md").read_text(encoding="utf-8").split("\n## Trailers\n")[0]
    report += "\n## Trailers\n\n"
    for row in ranked:
        report += (f"### {row['promo_id']} — {row['segment_text']}\n\nTotal: {row['duration_s']:.2f} s | coherence: {row['coherence']}/5 | score: {row['final_score']:.4f}\n\n"
                   f"Hook: {row['hook_text']}\n\nCliffhanger: {row['cliffhanger_text']}\n\nStory: {row['story']}\n\nWhy it works: {row['why_it_works']}\n\n"
                   f"Scores: hook {row['hook_score']}; cliffhanger {row['cliffhanger_score']}; standalone {row['standalone_score']}; emotion {row['emotion_score']}; post priority {row['post_priority']}. Safety: {row['ad_safety_display']}\n\n")
    if not ranked:
        report += "No eligible trailers.\n"
    for item in excluded['rejected']:
        report += f"Rejected {item['id']}: {'; '.join(item['fix_log'])}\n\n"
    for item in excluded['unsafe']:
        report += f"Unsafe excluded {item['promo_id']}: {item['ad_safety_reason']}\n\n"
    (final/"report.md").write_text(report,encoding="utf-8")
    return {"trailers":ranked,"validated":len(validated),"rejected":sum(t.status=="rejected" for t in validated)}
