import hashlib
import json
from pathlib import Path

import pytest

from src import judge, candidates
from src.judge import JudgeBatch, JudgeScore, plan_judge, run_judge
from src.llm import MockProvider
from src.models import CandidateList
from src.pipeline import run_pipeline
from src.rank import duration_fit, rank_promos, write_ranking
from src.validate import ValidatedPromo, validate_candidate
from test_validate import timeline, proposal
from test_context import synthetic_bible
from test_candidates import proposal as llm_proposal


def score(promo_id="E099_C01", **changes):
    return JudgeScore(promo_id=promo_id, **({"hook_score": 4, "cliffhanger_score": 3,
        "standalone_score": 4, "emotion_score": 5, "post_priority":10, "ad_safety": "safe", "ad_safety_reason": "",
        "weakest_point": "Slow camera opening.", "one_line_pitch": "A stranger witnesses a threat.",
        "why_it_works": "This final clip opens with conflict and ends with an unanswered threat."} | changes))


def setup_judge(tmp_path):
    ep = timeline()
    ep.logline = "SECRET_LOGLINE"
    out, parsed = tmp_path / "outputs", tmp_path / "parsed"
    parsed.mkdir()
    target = out / "validated_test"
    target.mkdir(parents=True)
    (parsed / "episode_099.json").write_text(ep.model_dump_json(), encoding="utf-8")
    promos = [validate_candidate(ep, proposal(1, 5, 1, 5, why_it_works="SECRET_GENERATOR")),
              validate_candidate(ep, proposal(6, 10, 6, 10, candidate_id="E099_C02"))]
    (target / "episode_099.json").write_text(json.dumps([p.model_dump() for p in promos]), encoding="utf-8")
    config = {"parsed_dir": str(parsed), "output_dir": str(out), "models": {"judge": {"provider": "mock", "model": "mock"}}}
    return config, ep, promos


def mock(tmp_path, responses):
    client = MockProvider(responses=responses, cache_dir=tmp_path / "cache", raw_log_dir=tmp_path / "raw")
    client.limiter.wait = lambda *args: None
    return client


def snapshot(path):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob("*") if p.is_file()}


def test_judge_blind_prompt_and_resume_invalidate_changed_script(tmp_path):
    config, ep, promos = setup_judge(tmp_path)
    client = mock(tmp_path, [JudgeBatch([score(p.id)]).model_dump_json() for p in promos] + [JudgeBatch([score()]).model_dump_json()])
    stats = run_judge(config, "test", client=client)
    assert stats["api_calls"] == 2 and not stats["pending"]
    request = client.calls[0]
    assert "SECRET_LOGLINE" not in request.user and "SECRET_GENERATOR" not in request.user
    assert "self_score" not in request.user
    assert "E099_L001" in request.user and "duration 25.00 s" in request.user
    run_judge(config, "test", client=client)
    assert len(client.calls) == 2
    ep.shots[0].lines[0].text = "A newly revised line of dialogue?"
    Path(config["parsed_dir"], "episode_099.json").write_text(ep.model_dump_json(), encoding="utf-8")
    run_judge(config, "test", client=client)
    assert len(client.calls) == 3
    assert "E099_C02 | duration" not in client.calls[-1].user


def test_judge_splits_by_serialized_size_and_rejects_oversize(tmp_path):
    config, ep, promos = setup_judge(tmp_path)
    single = plan_judge(promos[:1], {99: ep}, config, "mock")[0]["tokens"]
    config["llm"] = {"max_request_tokens": single + 5}
    plans = plan_judge(promos, {99: ep}, config, "mock")
    assert len(plans) == 2 and all(len(p["promos"]) == 1 for p in plans)
    config["llm"]["max_request_tokens"] = 1
    client = mock(tmp_path, [])
    stats = run_judge(config, "test", client=client)
    assert len(stats["oversized"]) == 2 and not client.calls


def test_judge_invalid_ids_repaired_once_then_saved(tmp_path):
    config, _, promos = setup_judge(tmp_path)
    client = mock(tmp_path, [JudgeBatch([score("WRONG")]).model_dump_json()] +
                            [JudgeBatch([score(p.id)]).model_dump_json() for p in promos])
    stats = run_judge(config, "test", client=client)
    assert stats["api_calls"] == 3 and not stats["pending"]
    assert "Your previous output failed validation" in client.calls[1].user


def test_judge_daily_quota_stops_without_later_batches(tmp_path):
    from test_llm import APIError
    config, _, _ = setup_judge(tmp_path)
    client = mock(tmp_path, [APIError(429, "daily token quota exhausted limit")])
    stats = run_judge(config, "test", client=client, batch_size=1)
    assert stats["stopped"] == "LLMQuotaError" and stats["api_calls"] == 1
    assert len(client.calls) == 1 and len(stats["pending"]) == 2


def test_judge_budget_preserves_first_batch(tmp_path):
    config, _, _ = setup_judge(tmp_path)
    client = mock(tmp_path, [JudgeBatch([score()]).model_dump_json()])
    stats = run_judge(config, "test", client=client, batch_size=1, max_api_calls=1)
    assert stats["done"] == ["E099_C01"] and stats["pending"] == ["E099_C02"]
    assert stats["stopped"] == "LLMBudgetError" and len(client.calls) == 1


def test_judge_dry_run_never_constructs_client_or_writes(tmp_path, monkeypatch):
    config, _, _ = setup_judge(tmp_path)
    monkeypatch.setattr(judge, "get_llm", lambda *a: pytest.fail("API client in dry run"))
    before = snapshot(tmp_path)
    stats = run_judge(config, "test", dry_run=True)
    assert stats["requests"] == 2 and snapshot(tmp_path) == before


@pytest.mark.parametrize("duration,expected", [(20,.7),(25,1),(45,1),(55,.7),(60,.4)])
def test_duration_fit_edges(duration, expected):
    assert duration_fit(duration) == expected


def test_rank_formula_penalties_and_unsafe_exclusion():
    ep = timeline()
    base = validate_candidate(ep, proposal())
    base.episode = 1
    full = score(hook_score=5, cliffhanger_score=5, standalone_score=5, emotion_score=5)
    assert rank_promos([base], [full], {})["ranked"][0]["final_score"] == 1
    base.status = "flagged"
    base.flags = ["hook_reassigned", "cliffhanger_reassigned"]
    caution = full.model_copy(update={"ad_safety": "caution", "ad_safety_reason": "Violence"})
    row = rank_promos([base], [caution], {})["ranked"][0]
    assert row["final_score"] == pytest.approx(.86)
    unsafe = caution.model_copy(update={"ad_safety": "unsafe"})
    result = rank_promos([base], [unsafe], {})
    assert not result["ranked"] and len(result["unsafe"]) == 1
    result = rank_promos([base], [full], {"rank": {"weights": {"judge": 0}}})
    assert result["ranked"][0]["final_score"] == pytest.approx(.09)


def test_soft_diversity_and_consecutive_ranks_are_deterministic():
    promos, scores = [], []
    for episode in (1,2,3):
        for i in range(3):
            promo = validate_candidate(timeline(), proposal())
            promo.id = f"E{episode:03d}_C{i:02d}"
            promo.episode = episode
            promos.append(promo)
            scores.append(score(promo.id))
    result = rank_promos(promos, scores, {})
    assert [r["episode"] for r in result["top"][:3]] == [1,2,3]
    assert len(result["top"]) == 9
    assert all(sum(r["episode"]==e for r in result["top"]) == 3 for e in (1,2,3))
    assert all(r["why_post_first"] for r in result["top"][:5])
    assert [r["rank"] for r in result["ranked"]] == list(range(1,10))
    for ep in (1,2,3):
        ep_rows = [r for r in result["ranked"] if r["episode"] == ep]
        assert [round(r['final_score']-r['score_adj'],2) for r in ep_rows] == [0,.05,.1]


def test_rank_report_full_scripts_markers_and_stale_score_exclusion(tmp_path):
    config, ep, promos = setup_judge(tmp_path)
    client = mock(tmp_path, [JudgeBatch([score(p.id)]).model_dump_json() for p in promos])
    run_judge(config, "test", client=client)
    result = write_ranking(config, "test")
    report = Path(config["output_dir"], "final_test/report.md").read_text(encoding="utf-8")
    assert ">> [E099_L001]" in report and "<<" in report
    assert "Synthetic line 5?" in report and len(result["ranked"]) == 2
    assert result["ranked"][0]["why_it_works"] == score().why_it_works
    assert result["ranked"][0]["generator_note"] != score().why_it_works
    ep.shots[0].lines[0].text = "Changed dialogue?"
    Path(config["parsed_dir"], "episode_099.json").write_text(ep.model_dump_json(), encoding="utf-8")
    result = write_ranking(config, "test")
    assert len(result["unjudged"]) == 1


def setup_pipeline(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    fixture = Path(__file__).parent / "fixtures/sample_episode.txt"
    (source / "episode.txt").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    config = {"parsed_dir": str(tmp_path / "parsed"), "output_dir": str(tmp_path / "outputs"),
              "models": {"candidates": {"provider": "mock", "model": "mock"}, "bible": {"provider": "mock", "model": "mock"},
                         "judge": {"provider": "mock", "model": "mock"}}}
    return source, config


def test_new_show_pipeline_all_stages_and_resume(tmp_path, monkeypatch):
    source, config = setup_pipeline(tmp_path)
    candidate = llm_proposal(start_shot_id="S001", end_shot_id="S005", hook_line_id="L002", cliffhanger_line_id="L010")
    generator = mock(tmp_path / "generator", [synthetic_bible().model_dump_json(),
                      json.dumps([CandidateList(episode=99,candidates=[candidate]).model_dump()])])
    reviewer = mock(tmp_path / "reviewer", [JudgeBatch([score()]).model_dump_json()])
    monkeypatch.setattr(candidates, "get_llm", lambda *a: generator)
    monkeypatch.setattr(judge, "get_llm", lambda *a: reviewer)
    result = run_pipeline(config, source, "test", max_api_calls=3)
    assert not result["stopped"] and result["api_calls"] == 3
    report = Path(config["output_dir"], "final_test/report.md")
    assert report.exists() and "E099_C01" in report.read_text()
    before = report.read_bytes()
    run_pipeline(config, source, "test", max_api_calls=0)
    assert len(generator.calls)==2 and len(reviewer.calls)==1 and report.read_bytes()==before
    plan = run_pipeline(config, source, "test", max_api_calls=0, dry_run=True)
    assert plan["candidate_requests"] == 0 and plan["judge_requests"] == 0


def test_new_show_pipeline_dry_run_is_read_only(tmp_path, monkeypatch):
    source, config = setup_pipeline(tmp_path)
    monkeypatch.setattr(candidates, "get_llm", lambda *a: pytest.fail("API client in dry run"))
    before = snapshot(tmp_path)
    result = run_pipeline(config, source, "test", dry_run=True)
    assert result["bible"]["requests"]==1 and result["candidate_requests"]==1
    assert result["judge_requests"] > 0 and snapshot(tmp_path)==before


def test_saved_bible_source_fingerprint_detects_a_new_show(tmp_path):
    from src.context import bible_source_fingerprint
    from src.pipeline import reusable_bible
    output = tmp_path / "outputs"
    output.mkdir()
    from src.parser import parse_file
    ep = parse_file(Path(__file__).parent / "fixtures/sample_episode.txt")
    (output / "story_bible.json").write_text(synthetic_bible().model_dump_json(), encoding="utf-8")
    (output / "story_bible_build.json").write_text(json.dumps({"source_fingerprint": bible_source_fingerprint([ep])}), encoding="utf-8")
    assert reusable_bible(output, [ep], False) is not None
    ep.logline = "A different show sharing the same speaker names."
    assert reusable_bible(output, [ep], False) is None


def test_compact_judge_omits_redundant_cameras_and_early_sounds():
    from src.cliprender import render_judge_clip, render_clip
    from src.models import Line
    ep = timeline(count=5)
    ep.shots[0].camera_text = "REDUNDANT CAMERA TEXT"
    ep.shots[1].lines = []
    ep.shots[1].camera_text = "C" * 100
    for index, text in [(0, "EARLY SOUND"), (4, "FINAL SOUND")]:
        ep.shots[index].lines.append(Line(id=f"E099_L{90+index}", episode=99, scene=1, type="sound",
                                         start=ep.shots[index].start, end=ep.shots[index].end, text=text))
    promo = validate_candidate(ep, proposal())
    rendered = render_judge_clip(ep, promo)
    assert "REDUNDANT CAMERA TEXT" not in rendered
    assert "EARLY SOUND" not in rendered.split("SCRIPT:")[1]
    assert "FINAL SOUND" in rendered and "C" * 57 + "..." in rendered
    assert "C" * 61 not in rendered
    assert "REDUNDANT CAMERA TEXT" in render_clip(ep, promo)


def test_judge_caps_three_highest_scores_and_reports_rest(tmp_path):
    config, ep, promos = setup_judge(tmp_path)
    for i, value in [(3, 5), (4, 1)]:
        promos.append(validate_candidate(ep, proposal(candidate_id=f"E099_C{i:02d}", self_score=value)))
    target = Path(config["output_dir"], "validated_test/episode_099.json")
    target.write_text(json.dumps([p.model_dump() for p in promos]), encoding="utf-8")
    from src.judge import cap_promos
    kept, capped = cap_promos(promos)
    assert capped[0].id == "E099_C04"
    client = mock(tmp_path, [JudgeBatch([score(p.id)]).model_dump_json() for p in kept])
    stats = run_judge(config, "test", client=client)
    assert stats["not_judged_cap"] == ["E099_C04"]
    assert "CLIP E099_C04" not in client.calls[0].user
    result = write_ranking(config, "test")
    assert len(result["capped"]) == 1
    report = Path(config["output_dir"], "final_test/report.md").read_text(encoding="utf-8")
    assert "Not judged (cap)" in report and "E099_C04: not judged (cap)" in report


@pytest.mark.parametrize("values", [{"hook_score": 0}, {"emotion_score": 6},
    {"ad_safety": "unknown"}])
def test_judge_schema_rejects_bad_ranges_or_safety_reasons(values):
    with pytest.raises(ValueError):
        score(**values)


def test_safe_reasons_are_normalized_without_repair():
    assert score(ad_safety_reason="No explicit content.").ad_safety_reason == ""


def test_completed_raw_scores_are_recovered_offline_without_api(tmp_path, monkeypatch):
    config, ep, promos = setup_judge(tmp_path)
    raw_dir = Path(config["output_dir"], "llm_raw")
    raw_dir.mkdir()
    raw_scores = [score(p.id).model_dump() | {"ad_safety_reason": "No explicit content."} for p in promos]
    for promo, raw_score in zip(promos, raw_scores):
        raw_path = raw_dir / f"judge_{promo.id}.json"
        raw_path.write_text(json.dumps({"provider": "mock", "model": "mock", "temperature": 0,
            "prompt": {"system": judge.SYSTEM_PATH.read_text(encoding="utf-8"), "user": judge.judge_prompt([promo], {99: ep})},
            "raw_response": json.dumps([raw_score]), "ok": False}), encoding="utf-8")
    before = snapshot(raw_dir)
    monkeypatch.setattr(judge, "get_llm", lambda *a: pytest.fail("Recovery must be offline"))
    stats = run_judge(config, "test", max_api_calls=0)
    assert stats["api_calls"] == 0 and not stats["pending"]
    assert len(stats["recovered_from_raw"]) == 2 and snapshot(raw_dir) == before


@pytest.mark.parametrize("numbers", [(1,2,3), (1,2), (1,)])
def test_judge_batches_round_robin_never_repeat_an_episode(numbers):
    promos = []
    episodes = {number: timeline() for number in numbers}
    for number in numbers:
        for i in range(1,4):
            p = validate_candidate(timeline(), proposal())
            p.episode, p.id = number, f"E{number:03d}_C{i:02d}"
            promos.append(p)
    plans = plan_judge(promos, episodes, {}, "mock", 3)
    assert len(plans) == 3
    for plan in plans:
        assert len({p.episode for p in plan["promos"]}) == len(plan["promos"])
    assert [[p.id for p in plan['promos']] for plan in plans] == [
        [f"E{n:03d}_C{i:02d}" for n in numbers] for i in range(1,4)]


def test_judge_view_opening_uses_exact_three_second_window_and_final_camera():
    from src.cliprender import render_judge_clip
    from src.models import Line
    ep = timeline(count=5)
    for start,text in [(3.0,"AT THREE"),(3.01,"TOO LATE")]:
        ep.shots[0].lines.append(Line(id=f"E099_L{int(start*100)}",episode=99,scene=1,type="action",
                                      start=start,end=start+.1,text=text))
    ep.shots[-1].camera_text = "Final unresolved frozen face."
    p = validate_candidate(ep, proposal())
    view = render_judge_clip(ep,p)
    opening = next(line for line in view.splitlines() if line.startswith('FIRST 3 SECONDS:'))
    ending = next(line for line in view.splitlines() if line.startswith('FINAL BEAT:'))
    assert 'AT THREE' in opening and 'TOO LATE' not in opening
    assert 'Synthetic line 1?' in opening
    assert 'Synthetic line 5?' in ending and 'Final unresolved frozen face.' in ending
    assert view.index('DURATION:') < view.index('SCRIPT:')


def test_judge_opening_falls_back_to_first_camera_when_no_early_lines():
    from src.cliprender import render_judge_clip
    ep = timeline(count=5)
    ep.shots[0].lines[0].start = 4
    p = ValidatedPromo(episode=99,id='test',start_shot_id='E099_S001',end_shot_id='E099_S005',duration_s=25)
    opening = next(line for line in render_judge_clip(ep,p).splitlines() if line.startswith('FIRST 3 SECONDS:'))
    assert opening == 'FIRST 3 SECONDS: CAM: Close-up on Anna.'


def test_judge_is_temperature_zero_and_uses_configured_groq_reasoning(tmp_path):
    config, _, promos = setup_judge(tmp_path)
    config['models']['judge'] = {'provider':'groq','model':'openai/gpt-oss-120b','reasoning_effort':'medium'}
    client = mock(tmp_path,[JudgeBatch([score(p.id)]).model_dump_json() for p in promos])
    run_judge(config,'test',client=client)
    assert all(request.temperature == 0 for request in client.calls)
    assert client.reasoning_effort == 'medium'
    ep = timeline()
    low = {'models': {'judge': {'reasoning_effort': 'low'}}}
    assert judge.judge_fingerprint(ep,promos[0],'groq','openai/gpt-oss-120b',config) != judge.judge_fingerprint(ep,promos[0],'groq','openai/gpt-oss-120b',low)


def test_soft_ranking_ties_use_cliffhanger_then_duration_and_penalty_is_configurable():
    first = validate_candidate(timeline(), proposal(candidate_id='A'))
    second = first.model_copy(update={'id':'B','episode':100,'duration_s':30})
    first.episode=100
    a = score('A',hook_score=5,cliffhanger_score=3)
    b = score('B',hook_score=3,cliffhanger_score=5)
    result = rank_promos([first,second],[a,b],{'rank':{'same_episode_penalty':.2}})
    assert [r['promo_id'] for r in result['ranked']] == ['B','A']
    assert result['ranked'][1]['score_adj'] == pytest.approx(result['ranked'][1]['final_score']-.2)
    a = score('A')
    b = score('B')
    result = rank_promos([second,first],[b,a],{})
    assert result['ranked'][0]['promo_id'] == 'A'
