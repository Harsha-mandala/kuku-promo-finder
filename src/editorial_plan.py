"""Conservative editorial-stage estimates when future selections are unknown."""

from .cliprender import render_judge_clip
from .editor_pass import EditorOrder
from .llm import estimate_json_tokens, request_token_limit, step_model_config
from .models import TrailerBatch
from .trailers import PROMPTS, TrailerJudgeBatch, render_segment


def plan_editorial(config,promos,episodes,*,no_trailers=False,no_editor=False):
    scripts = sorted((render_judge_clip(episodes[p.episode],p) for p in promos if p.status != "rejected"),key=len,reverse=True)
    rows = []
    count = config.get('trailers',{}).get('count',5)
    trailer_on = not no_trailers and config.get('trailers',{}).get('enabled',False)
    if scripts and trailer_on:
        provider,model = step_model_config(config,'trailers')
        min_gap_s = config.get('trailers', {}).get('min_gap_s', 10)
        system = (PROMPTS/'trailers_system.md').read_text(encoding='utf-8').format(count=count, min_gap_s=min_gap_s)
        tokens = estimate_json_tokens(system,'\n'.join(scripts[:20]),TrailerBatch)
        rows.append({'stage':'trailers','provider':provider,'model':model,'requests':1,'tokens':tokens})
        jp,jm = step_model_config(config,'judge')
        # Three source windows per unknown trailer is a conservative script reserve.
        each = estimate_json_tokens((PROMPTS/'judge_system.md').read_text(encoding='utf-8'),scripts[0]*3,TrailerJudgeBatch)
        limit = request_token_limit(config,jp)
        per_request = max(1,min(count,(limit//each if limit else count)))
        requests = (count+per_request-1)//per_request
        rows.append({'stage':'trailer_judge','provider':jp,'model':jm,'requests':requests,'tokens':count*each})
    if scripts and not no_editor and config.get('editor',{}).get('enabled',False):
        provider,model = step_model_config(config,'editor')
        compact=sorted((render_segment(episodes[p.episode],p.start_shot_id,p.end_shot_id,include_final_sound=True)
                        for p in promos if p.status!='rejected'),key=len,reverse=True)
        payload = '\n'.join(compact[:15]) + (compact[0]*3*count if trailer_on else '')
        tokens = estimate_json_tokens((PROMPTS/'editor_system.md').read_text(encoding='utf-8'),payload,EditorOrder)
        rows.append({'stage':'editor','provider':provider,'model':model,'requests':1,'tokens':tokens})
    for row in rows:
        print(f"{row['stage'].upper()} {row['provider']}/{row['model']}: requests={row['requests']} estimated_input_tokens={row['tokens']} (future selection reserve)")
    return {'editorial_stages':rows,'trailers_requests':sum(r['requests'] for r in rows if r['stage']=='trailers'),
            'editor_requests':sum(r['requests'] for r in rows if r['stage']=='editor')}
