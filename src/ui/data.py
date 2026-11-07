"""Read explicitly selected bundles and configure upload workspaces."""

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from ..context import load_episodes
from ..judge import load_promos
from ..editorial_io import fresh_editor_rows, single_source_key

PROJECT_ROOT=Path(__file__).resolve().parents[2]
DEMO_ROOT=PROJECT_ROOT/'demo'


def local_runs_allowed(environ, headers):
    if environ.get('PUBLIC_DEPLOY') == '1' or Path('/mount/src').is_dir():
        return False
    host = headers.get('Host', '')
    forwarded = headers.get('X-Forwarded-Host', host)
    try:
        return all(urlsplit('//'+value).hostname in {'127.0.0.1', 'localhost', '::1'}
                   for value in (host, forwarded))
    except ValueError:
        return False


def existing_run_tags(output):
    return sorted(path.name.removeprefix('final_') for path in Path(output).glob('final_*')
                  if path.is_dir() and not path.is_symlink()
                  and re.fullmatch(r'final_[A-Za-z0-9_-]+', path.name))


def read_json(path, fallback):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else fallback


def load_bundle(parsed, output, tag):
    parsed,output=Path(parsed),Path(output)
    bundle = {'episodes':load_episodes(parsed) if parsed.exists() and list(parsed.glob('episode_*.json')) else [],
            'ranked':read_json(output/f'final_{tag}/promos_ranked.json',[]),
            'validated':load_promos(output/f'validated_{tag}'),
            'bible':read_json(output/'story_bible.json',{}),
            'scores':read_json(output/f'judged_{tag}/scores.json',[]),
            'exclusions':read_json(output/f'final_{tag}/exclusions.json',{}),
            'output':output,'tag':tag,
            'trailers':read_json(output/f'final_{tag}/trailers_ranked.json',[]),
            'trailer_validated':read_json(output/f'trailers_{tag}/validated.json',[]),
            'editor_order':fresh_editor_rows(output,tag,parsed)}
    trailer_manifest=read_json(output/f'trailers_{tag}/manifest.json',{})
    if trailer_manifest.get('source_key') and trailer_manifest['source_key'] != single_source_key(output,tag,parsed):
        bundle['trailers']=[]
    edits = {row['promo_id']:row for row in bundle['editor_order']}
    remaining=0
    for row in bundle['ranked']+bundle['trailers']:
        if row['promo_id'] in edits:
            row.update(edits[row['promo_id']])
            row['rank'] = row['final_rank']
        elif edits:
            remaining+=1
            row.update(judge_rank=row['rank'],editor_rank=None,final_rank=len(edits)+remaining)
            row['rank']=row['final_rank']
    bundle['ranked'].sort(key=lambda r:(r.get('final_rank',len(edits)+r['rank']),r['promo_id']))
    bundle['safety_queue'] = read_json(output/f'final_{tag}/safety_review.json', [])
    queue = {row['promo_id']: row['safety_review'] for row in bundle['safety_queue']}
    for row in bundle['ranked'] + bundle['trailers']:
        if row['promo_id'] in queue:
            row['safety_review'] = queue[row['promo_id']]
    return bundle


def load_demo():
    return load_bundle(DEMO_ROOT/'precomputed/episodes',DEMO_ROOT/'precomputed/results','demo')


def private_config(base, root, keys):
    import copy
    config=copy.deepcopy(base)
    root=Path(root)
    config['parsed_dir']=str(root/'parsed')
    config['output_dir']=str(root/'outputs')
    config.setdefault('llm',{})['cache_dir']=str(root/'.cache')
    config['llm']['raw_log_dir']=str(root/'outputs/llm_raw')
    config['_api_keys']=dict(keys)
    return config


def upload_config(base, root, keys, mock=False):
    config = private_config(base, root, {} if mock else keys)
    if mock:
        config['_mock_upload'] = True
        for stage in ('bible', 'candidates', 'judge', 'trailers', 'editor'):
            config['models'][stage] = {'provider': 'mock', 'model': 'mock'}
        config['llm'].update(max_request_tokens=None, tokens_per_minute=None, max_retries=1)
    return config
