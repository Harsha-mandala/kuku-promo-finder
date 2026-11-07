import hashlib
import json
from pathlib import Path

import pytest

from src import candidates
from src.candidates import dry_run_candidates, load_saved_candidates, plan_batches, run_candidates
from src.context import load_episodes
from src.llm import step_model_config
from src.models import CandidateList
from test_candidates import client_for, prepare_run, proposal
from test_context import synthetic_bible
from src.parser import parse_file


def test_busy_gemini_five_batch_splits_after_two_failures(tmp_path, monkeypatch):
    from src import llm
    from test_llm import APIError
    config, _, _ = setup_run(tmp_path, (99, 100, 101, 102, 103))
    replies = [APIError(503, "overloaded"), APIError(504, "timeout"),
               json.dumps([result(n).model_dump() for n in (99, 100, 101)]),
               json.dumps([result(n).model_dump() for n in (102, 103)])]
    client = client_for(tmp_path, replies, max_retries=3)
    client.provider_name = "gemini"
    waits = []
    monkeypatch.setattr(llm, "_retry_sleep", waits.append)
    stats = run_candidates(config, "all", client, batch_size=5, max_api_calls=5)
    assert stats["episodes_ok"] == [99, 100, 101, 102, 103]
    assert client.call_budget.used == 4
    assert waits == [30]
    assert "EPISODE 102 | duration" not in client.calls[2].user
    assert "EPISODE 102 | duration" in client.calls[3].user
    assert client.thinking_budget == 1024

FIXTURE = Path(__file__).parent / "fixtures" / "sample_episode.txt"


def setup_run(tmp_path, numbers=(99, 100)):
    episode = parse_file(FIXTURE)
    bible = synthetic_bible()
    config = prepare_run(tmp_path, episode, bible, numbers)
    return config, load_episodes(config["parsed_dir"]), bible


def result(number, **changes):
    candidate = proposal(start_shot_id="S002", end_shot_id="S004", **changes)
    return CandidateList(episode=number, candidates=[candidate])


def test_batch_split_respects_token_guard_and_consecutive_order(tmp_path):
    config, episodes, bible = setup_run(tmp_path, (99, 100, 101))
    singles = plan_batches(episodes, episodes, bible, 1, 5, None)
    limit = max(plan.estimated_tokens for plan in singles) + 1
    plans = plan_batches(episodes, episodes, bible, 3, 5, limit)
    assert [ep.number for plan in plans for ep in plan.episodes] == [99, 100, 101]
    assert len(plans) == 3
    assert all(not plan.oversized and plan.estimated_tokens <= limit for plan in plans)
    gaps = plan_batches([episodes[0], episodes[2]], episodes, bible, 5, 5, None)
    assert [[ep.number for ep in plan.episodes] for plan in gaps] == [[99], [101]]


@pytest.mark.parametrize("kind", ["unknown_id", "schema_error"])
def test_only_bad_episode_is_reasked_once(tmp_path, kind):
    config, episodes, _ = setup_run(tmp_path)
    good = result(99).model_dump()
    bad = result(100, hook_line_id="L999").model_dump()
    if kind == "schema_error":
        bad["candidates"][0].pop("hook_line_id")
    client = client_for(tmp_path, [json.dumps([good, bad]), result(100).model_dump_json()])
    stats = run_candidates(config, "all", client, batch_size=2)
    assert stats["episodes_ok"] == [99, 100]
    assert len(client.calls) == 2
    assert "EPISODE 99 | duration" in client.calls[0].user
    assert "EPISODE 100 | duration" in client.calls[0].user
    assert "EPISODE 99 | duration" not in client.calls[1].user
    assert "EPISODE 100 | duration" in client.calls[1].user
    assert "Previous output:" not in client.calls[1].user
    assert stats["repair_attempts"] == 1
    for episode in episodes:
        assert load_saved_candidates(Path(config["output_dir"]) / f"candidates/episode_{episode.number:03d}.json", episode, 5)


def test_second_bad_response_does_not_trigger_third_call(tmp_path):
    config, _, _ = setup_run(tmp_path)
    bad = result(100, hook_line_id="L999")
    client = client_for(tmp_path, [json.dumps([result(99).model_dump(), bad.model_dump()]), bad.model_dump_json()], max_retries=10)
    stats = run_candidates(config, "all", client, batch_size=2)
    assert stats["episodes_ok"] == [99]
    assert stats["episodes_failed"] == [100]
    assert len(client.calls) == 2


def test_invalid_batch_json_repairs_each_episode_alone(tmp_path):
    config, _, _ = setup_run(tmp_path)
    client = client_for(tmp_path, ["not JSON", result(99).model_dump_json(), result(100).model_dump_json()])
    stats = run_candidates(config, "all", client, batch_size=2)
    assert stats["episodes_ok"] == [99, 100]
    assert len(client.calls) == 3
    assert stats["repair_attempts"] == 2


def test_batch_overlap_is_logged_and_deduped_locally(tmp_path):
    config, episodes, _ = setup_run(tmp_path, (99,))
    low = proposal(start_shot_id="S001", end_shot_id="S004", hook_line_id="L002", self_score=2)
    high = proposal(start_shot_id="S002", end_shot_id="S005", cliffhanger_line_id="L010", self_score=5)
    response = CandidateList(episode=99, candidates=[low, high])
    client = client_for(tmp_path, [json.dumps([response.model_dump()])])
    stats = run_candidates(config, "99", client, batch_size=1)
    assert len(client.calls) == 1
    path = Path(config["output_dir"]) / "candidates/episode_099.json"
    saved = load_saved_candidates(path, episodes[0], 5)
    assert len(saved) == 1 and saved[0].self_score == 5
    assert saved[0].candidate_id == "E099_C02"
    log = (Path(config["output_dir"]) / "candidates_postprocess.log").read_text(encoding="utf-8")
    assert "overlap_drop" in log and "E099_C01" in log and "E099_C02" in log
    assert stats["repair_attempts"] == 0


def test_dry_run_never_constructs_client_or_writes_files(tmp_path, monkeypatch, capsys):
    config, _, _ = setup_run(tmp_path)
    config["models"] = {"candidates": {"provider": "mock", "model": "planned-model"}}
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not construct an LLM client")
    monkeypatch.setattr(candidates, "get_llm", forbidden)
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.rglob("*") if path.is_file()}
    plan = dry_run_candidates(config, "all", 2)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    assert plan["requests"] == 1
    assert plan["provider"] == "mock" and plan["model"] == "planned-model"
    assert plan["max_episode_repair_calls"] == 2
    assert "no API calls or file writes" in capsys.readouterr().out


def test_oversized_single_episode_is_not_sent(tmp_path):
    config, _, _ = setup_run(tmp_path, (99,))
    config["llm"] = {"max_request_tokens": 10}
    client = client_for(tmp_path, [])
    stats = run_candidates(config, "99", client, batch_size=5)
    assert not client.calls
    assert stats["episodes_failed"] == [99]
    plan = dry_run_candidates(config, "99", 5, "mock", "mock")
    assert plan["requests"] == 0 and plan["oversized_episodes"] == [99]


def test_step_model_selection_and_cli_overrides():
    config = {"llm": {"provider": "gemini", "model": "fallback"}, "models": {"candidates": {"provider": "groq", "model": "candidate-model"}, "bible": {"provider": "mock", "model": "bible-model"}, "judge": {"provider": "mock", "model": "judge-model"}}}
    assert step_model_config(config, "candidates") == ("groq", "candidate-model")
    assert step_model_config(config, "bible") == ("mock", "bible-model")
    assert step_model_config(config, "judge") == ("mock", "judge-model")
    assert step_model_config(config, "candidates", "mock", "override") == ("mock", "override")


def test_run_budget_stops_with_saved_progress_and_pending_episodes(tmp_path, capsys):
    config, _, _ = setup_run(tmp_path)
    client = client_for(tmp_path, [json.dumps([result(99).model_dump()]), json.dumps([result(100).model_dump()])])
    stats = run_candidates(config, "all", client, max_api_calls=1)
    assert len(client.calls) == 1
    assert stats["episodes_ok"] == [99]
    assert stats["episodes_not_attempted"] == [100]
    assert stats["stopped_api_budget"] and stats["api_budget_used"] == 1
    assert (Path(config["output_dir"]) / "candidates/episode_099.json").exists()
    assert "DONE episodes: [99] | PENDING episodes: [100]" in capsys.readouterr().out
