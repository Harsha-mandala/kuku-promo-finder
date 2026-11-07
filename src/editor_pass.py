"""Blind comparative ordering, normalized rank blending and resumable exports."""

import random
from pathlib import Path

from pydantic import BaseModel, Field, RootModel

from .context import load_episodes
from .editorial_io import digest, read, stage_client, write, write_csv, editor_source_key
from .trailers import ValidatedTrailer, render_segment
from .validate import ValidatedPromo


class EditorChoice(BaseModel):
    promo_id: str
    reason: str = Field(min_length=1)


class EditorOrder(RootModel[list[EditorChoice]]):
    pass


def editor_inputs(output,tag,episodes,seed):
    final = output/f"final_{tag}"
    singles = read(final/"promos_ranked.json",[])[:15]
    trailers = read(final/"trailers_ranked.json",[])[:5]
    items = [dict(row,type="single") for row in singles]+[dict(row,type="trailer") for row in trailers]
    promos = {p["id"]:ValidatedPromo.model_validate(p) for path in (output/f"validated_{tag}").glob("episode_*.json") for p in read(path,[])}
    multi = {p["id"]:ValidatedTrailer.model_validate(p) for p in read(output/f"trailers_{tag}/validated.json",[])}
    rng = random.Random(seed)
    # Rank-only changes should not alter the blind prompt or waste a model call.
    shuffled = sorted(items,key=lambda item:item['promo_id'])
    rng.shuffle(shuffled)
    rows = []
    for item in shuffled:
        key = item["promo_id"]
        if item['type']=='trailer':
            script = '\n--- CUT ---\n'.join(render_segment(episodes[s.episode],s.start_shot_id,s.end_shot_id,
                include_final_sound=True) for s in multi[key].segments)
        else:
            promo=promos[key]
            script=render_segment(episodes[item['episode']],promo.start_shot_id,promo.end_shot_id,include_final_sound=True)
        script=f"DURATION: {item['duration_s']:.2f} s\n"+script
        rows.append(f"PROMO_ID: {key}\n{script}")
    return items,"\n\n".join(rows)


def judge_order(items):
    return sorted(items,key=lambda r:(-r.get("score_adj",r["final_score"]),-r["cliffhanger_score"],r["duration_s"],r["promo_id"]))


def complete_order(items, order):
    """Keep the first occurrence and fill omitted positions by judge rank."""
    seen, duplicates, cleaned = set(), set(), []
    for choice in order:
        if choice.promo_id in seen:
            duplicates.add(choice.promo_id)
            continue
        seen.add(choice.promo_id)
        cleaned.append(choice)
    missing = []
    for item in judge_order(items):
        key = item["promo_id"]
        if key not in seen:
            missing.append(key)
            cleaned.append(EditorChoice(promo_id=key, reason="Appended in judge order after the editor omitted this ID."))
    return cleaned, missing, sorted(duplicates)


def blend_ranks(items,order,*,judge_weight=.6,editor_weight=.4):
    if judge_weight < 0 or editor_weight < 0 or abs(judge_weight+editor_weight-1) > 1e-9:
        raise ValueError("Editor blending weights must be nonnegative and sum to one")
    judge_ranks = {r["promo_id"]:i for i,r in enumerate(judge_order(items),1)}
    editor_ranks = {r.promo_id:(i,r.reason) for i,r in enumerate(order,1)}
    if len(editor_ranks) != len(order) or set(editor_ranks) != set(judge_ranks):
        raise ValueError("Editor must return every given ID exactly once")
    rows = []
    n = len(items)
    for item in items:
        key = item["promo_id"]
        jr, (er,reason) = judge_ranks[key],editor_ranks[key]
        js,es = ((n-jr)/(n-1),(n-er)/(n-1)) if n > 1 else (1.,1.)
        rows.append(dict(item,judge_rank=jr,editor_rank=er,editor_reason=reason,why_post_first=reason,
                         final_rank_score=judge_weight*js+editor_weight*es,rank_disagreement=abs(jr-er)>=5))
    rows.sort(key=lambda r:(-r["final_rank_score"],r["judge_rank"],r["promo_id"]))
    for i,row in enumerate(rows,1):
        row["final_rank"] = i
    return rows


def run_editor(config,tag,*,budget=None,client=None,force=False):
    output = Path(config.get("output_dir","outputs"))
    final = output/f"final_{tag}"
    episodes = {e.number:e for e in load_episodes(config.get("parsed_dir","parsed"))}
    settings = config.get("editor",{})
    items,user = editor_inputs(output,tag,episodes,settings.get("seed",20261106))
    if not items:
        return {"order":[],"empty_pool":True}
    system = (Path(__file__).parent/"prompts/editor_system.md").read_text(encoding="utf-8")
    key = digest([system,user,config.get("models",{}).get("editor")])
    manifest = read(final/"editor_inputs.json",{})
    if not force and manifest.get("key") == key and (final/"editor_response.json").exists():
        order = EditorOrder.model_validate(read(final/"editor_response.json")).root
    else:
        client = client or stage_client(config,"editor",budget)
        if budget is not None:
            client.call_budget = budget
        expected = {row["promo_id"] for row in items}
        def validate(response):
            ids = [row.promo_id for row in response.root]
            unknown = sorted(set(ids) - expected)
            return [f"[unknown_promo_id] Unknown promo IDs: {', '.join(unknown)}"] if unknown else []
        order = client.generate_json(system,user,EditorOrder,f"editor_{tag}",extra_validator=validate,max_repairs=1).root
        write(final/"editor_response.json",[row.model_dump() for row in order])
        write(final/"editor_inputs.json",{"key":key})
    order, missing, duplicates = complete_order(items, order)
    rows = blend_ranks(items,order,judge_weight=settings.get("judge_weight",.6),editor_weight=settings.get("editor_weight",.4))
    for row in rows:
        row["editor_fallback"] = row["promo_id"] in missing or row["promo_id"] in duplicates
    fallback = {"missing_appended": missing, "duplicates_dropped": duplicates}
    write(final/"editor_fallback.json", fallback)
    if missing or duplicates:
        print(f"EDITOR fallback: {fallback}")
    write(final/"editor_order.json",rows)
    write(final/'editor_inputs.json',{'key':key,'source_key':editor_source_key(output,tag,config.get('parsed_dir','parsed'))})
    write_csv(final/"editor_order.csv",rows)
    selected = {row["promo_id"] for row in rows}
    remaining = [dict(row,type="single") for row in read(final/"promos_ranked.json",[]) if row["promo_id"] not in selected]
    remaining.extend(dict(row,type="trailer") for row in read(final/"trailers_ranked.json",[]) if row["promo_id"] not in selected)
    for i,row in enumerate(remaining,len(rows)+1):
        row.update(final_rank=i,judge_rank=row["rank"],editor_rank=None,editor_reason="",final_rank_score=None,rank_disagreement=False)
    write_csv(final/"promos_all.csv",rows+remaining)
    report = (final/"report.md").read_text(encoding="utf-8").split("\n## Automatic top 10\n")[0].split("\n## Final editor pass\n")[0]
    report = report.replace("## Top 10", "## Judge shortlist (before editor)")
    by_id = {r['promo_id']:r for r in rows}
    active = None
    lines = []
    import re
    for line in report.splitlines():
        match = re.match(r"### \d+\. (\S+) —",line)
        if match:
            active = match[1]
        if line.startswith('Why post first:') and active in by_id:
            line = 'Why post first: '+by_id[active]['editor_reason']
        lines.append(line)
    report = '\n'.join(lines)+'\n'
    report += "\n## Final editor pass\n\n"
    report += f"Blend: {settings.get('judge_weight',.6)} × normalized judge rank + {settings.get('editor_weight',.4)} × normalized editor rank. Higher is better.\n\n"
    if missing or duplicates:
        report += f"Fallback · appended: {', '.join(missing) or 'none'} · duplicates dropped: {', '.join(duplicates) or 'none'}.\n\n"
    report += "| Promo | Judge rank | Editor rank | Final rank | Disagreement ≥5 | Why post first |\n|---|---|---|---|---|---|\n"
    for row in rows:
        report += f"| {row['promo_id']} | {row['judge_rank']} | {row['editor_rank']} | {row['final_rank']} | {row['rank_disagreement']} | {row['editor_reason'].replace('|','/')} |\n"
    (final/"report.md").write_text(report,encoding="utf-8")
    return {"order":rows, "fallback":fallback}
