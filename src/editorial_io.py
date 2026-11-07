"""Shared persistence and budget setup for optional editorial stages."""

import copy
import csv
import hashlib
import json
from pathlib import Path

from .llm import get_llm, step_model_config


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def read(path, fallback=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["promo_id"]
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def single_source_key(output,tag,parsed):
    """Fingerprint clip provenance independently of workspace location."""
    files = [output/f'final_{tag}/promos_ranked.json']
    files += sorted((output/f'validated_{tag}').glob('episode_*.json'))
    files += sorted(Path(parsed).glob('episode_*.json'))
    return digest([(path.parent.name,path.name,hashlib.sha256(path.read_bytes()).hexdigest()) for path in files if path.exists()])


def editor_source_key(output,tag,parsed):
    files = [output/f'final_{tag}/trailers_ranked.json',output/f'trailers_{tag}/validated.json']
    return digest([single_source_key(output,tag,parsed),
                   [(p.name,hashlib.sha256(p.read_bytes()).hexdigest()) for p in files if p.exists()]])


def fresh_editor_rows(output,tag,parsed):
    metadata = read(output/f'final_{tag}/editor_inputs.json',{})
    if metadata.get('source_key') and metadata['source_key'] != editor_source_key(output,tag,parsed):
        return []
    return read(output/f'final_{tag}/editor_order.json',[])


def stage_client(config, stage, budget=None):
    config = copy.deepcopy(config)
    if stage == "judge":
        settings = config.get("models",{}).get("judge",{})
        config.setdefault("llm",{})["temperature"] = 0
        config["llm"]["requests_per_minute"] = settings.get("requests_per_minute",config["llm"].get("requests_per_minute",8))
        config["llm"].setdefault("providers",{}).setdefault("groq",{})["reasoning_effort"] = settings.get("reasoning_effort","low")
    provider, model = step_model_config(config,stage)
    client = get_llm(config,provider,model)
    if budget is not None:
        client.call_budget = budget
    if provider == "gemini":
        client.thinking_budget = config.get("models",{}).get(stage,{}).get("thinking_budget",1024)
    return client
