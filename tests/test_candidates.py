import csv
import json
from pathlib import Path

import pytest

from src.candidates import (
    candidate_validator, find_candidates, source_lookup, parse_episode_selector,
    previous_loglines, run_candidates, score_candidates, load_saved_candidates,
    postprocess_candidates, plan_batches, dry_run_candidates,
)
from src.context import PROMPTS_DIR, render_bible_compact
from src.llm import MockProvider
from types import SimpleNamespace
from src.models import Candidate, CandidateList, StoryBible
from src.parser import parse_file
from src.render import render_episode_compact

FIXTURE = Path(__file__).parent / "fixtures" / "sample_episode.txt"


@pytest.fixture
def episode():
    return parse_file(FIXTURE)


@pytest.fixture
def bible():
    return StoryBible(title_guess="Synthetic test", premise="Anna confronts Victor.", characters=[], main_conflicts=[], arcs=[])


def proposal(**changes):
    values = {
        "start_shot_id": "E099_S002", "end_shot_id": "E099_S004",
        "hook_line_id": "L004", "hook_type": "confrontation",
        "cliffhanger_line_id": "L008", "cliffhanger_type": "threat",
        "context_needed": "", "why_it_works": "A confrontation with an unresolved threat.", "self_score": 4,
    }
    return Candidate(**(values | changes))


def client_for(tmp_path, responses, **settings):
    client = MockProvider(
        responses=responses, cache_dir=tmp_path / "cache", raw_log_dir=tmp_path / "raw", **settings,
    )
    client.limiter.wait = lambda *args: None
    return client


def test_valid_response_computed_fields_and_verbatim_templates(tmp_path, episode, bible):
    result = CandidateList(episode=99, candidates=[proposal()])
    client = client_for(tmp_path, [result.model_dump_json()])
    scored = find_candidates(episode, [episode], bible, client, 5)[0]
    assert scored.candidate_id == "E099_C01"
    assert (scored.span_start, scored.span_end, scored.duration_s) == (4, 16.5, 12.5)
    assert scored.hook_offset_s == 1
    assert scored.cliffhanger_end_gap_s == 0
    assert scored.start_clean and scored.end_clean
    assert scored.n_shots == 3
    assert scored.hook_line_id == "E099_L004"
    assert scored.hook_text == "You were here all along?"
    assert scored.cliffhanger_text == "Go."
    assert client.calls[0].system == (PROMPTS_DIR / "candidates_system.md").read_text(encoding="utf-8").format(max_candidates=5)
    expected_user = (PROMPTS_DIR / "candidates_user.md").read_text(encoding="utf-8").format(
        bible=render_bible_compact(bible), previous_loglines="None, this is the first episode",
        episode=99, script=render_episode_compact(episode),
    )
    assert client.calls[0].user.startswith(expected_user)
    rescored = score_candidates(episode, CandidateList(episode=99, candidates=[scored]))
    assert rescored == [scored]


def test_unknown_shot_id_is_repaired(tmp_path, episode, bible):
    invalid = CandidateList(episode=99, candidates=[proposal(start_shot_id="E099_S999")])
    valid = CandidateList(episode=99, candidates=[proposal()])
    client = client_for(tmp_path, [invalid.model_dump_json(), valid.model_dump_json()])
    assert find_candidates(episode, [episode], bible, client)[0].start_shot_id == "E099_S002"
    assert len(client.calls) == 2
    assert "unknown_shot_id" in client.calls[1].user
    assert "candidate 1" in client.calls[1].user


def test_unknown_line_id_is_repaired(tmp_path, episode, bible):
    invalid = CandidateList(episode=99, candidates=[proposal(hook_line_id="L999")])
    valid = CandidateList(episode=99, candidates=[proposal()])
    client = client_for(tmp_path, [invalid.model_dump_json(), valid.model_dump_json()])
    scored = find_candidates(episode, [episode], bible, client)
    assert scored[0].hook_line_id == "E099_L004"
    assert scored[0].hook_text == "You were here all along?"
    assert len(client.calls) == 2
    assert "unknown_line_id" in client.calls[1].user


def test_more_than_half_overlap_is_deduped_without_repair(tmp_path,episode,bible):
    first=proposal(start_shot_id='S001',end_shot_id='S004',hook_line_id='L002')
    second=proposal(start_shot_id='S002',end_shot_id='S005',cliffhanger_line_id='L010',self_score=5)
    invalid=CandidateList(episode=99,candidates=[first,second])
    assert candidate_validator(episode,5)(invalid)==[]
    client=client_for(tmp_path,[invalid.model_dump_json()])
    values=find_candidates(episode,[episode],bible,client)
    assert len(values)==1 and values[0].self_score==5
    assert values[0].candidate_id=='E099_C02'
    assert len(client.calls)==1
    assert client.postprocess_events[0]['reason']=='overlap_drop'


def test_exactly_half_overlap_is_allowed(episode):
    first=proposal(start_shot_id='S001',end_shot_id='S004',hook_line_id='L002')
    second=proposal(start_shot_id='S003',end_shot_id='S004',hook_line_id='S003')
    errors=candidate_validator(episode,5)(CandidateList(episode=99,candidates=[first,second]))
    assert errors==[]  # Soft overlap does not trigger repair.
    assert len(postprocess_candidates(episode,score_candidates(episode,CandidateList(episode=99,candidates=[first,second])),5)[0])==1
    second=proposal(start_shot_id='S003',end_shot_id='S004',hook_line_id='S003',cliffhanger_line_id='L008')
    first=proposal(start_shot_id='S002',end_shot_id='S003',hook_line_id='L004',cliffhanger_line_id='S003')
    assert not candidate_validator(episode,5)(CandidateList(episode=99,candidates=[first,second]))
    assert len(postprocess_candidates(episode,score_candidates(episode,CandidateList(episode=99,candidates=[first,second])),5)[0])==2


def test_count_and_duration_do_not_trigger_repair(episode):
    episode.duration=91
    errors=candidate_validator(episode,5)(CandidateList(episode=99,candidates=[proposal()]))
    assert errors==[]
    values,events=postprocess_candidates(episode,score_candidates(episode,CandidateList(episode=99,candidates=[proposal()])),5)
    assert len(values)==1 and values[0].duration_s==12.5
    assert events[0]['reason']=='count_shortfall'


def test_unknown_foreign_and_outside_line_ids_rejected(episode):
    validate = candidate_validator(episode, 5)
    for value in ("L999", "E001_L004", "You were here all along?"):
        invalid = CandidateList(episode=99, candidates=[proposal(hook_line_id=value)])
        assert "unknown_line_id" in validate(invalid)[0]
    outside_clip = CandidateList(episode=99, candidates=[proposal(cliffhanger_line_id="L010")])
    assert "line_outside_clip" in validate(outside_clip)[0]


def test_episode_count_order_and_foreign_id_validation(episode):
    validate = candidate_validator(episode, 1)
    assert "episode_mismatch" in validate(CandidateList(episode=1, candidates=[proposal()]))[0]
    assert validate(CandidateList(episode=99, candidates=[]))==[]
    assert validate(CandidateList(episode=99, candidates=[proposal(), proposal()]))==[]
    reversed_range = proposal(start_shot_id="E099_S004", end_shot_id="E099_S002")
    assert "shot_order" in validate(CandidateList(episode=99, candidates=[reversed_range]))[0]
    foreign = proposal(start_shot_id="E001_S002")
    assert "unknown_shot_id" in validate(CandidateList(episode=99, candidates=[foreign]))[0]


def test_action_sound_and_empty_camera_ids(episode):
    sources = source_lookup(episode)
    assert sources["E099_L002"].text == "Anna walks in slowly."
    assert sources["E099_L010"].text == "Thunder crashes."
    assert sources["E099_S003"].text == "Close-up on Victor."
    assert "E099_S001" not in sources  # Contains an action; cannot use camera fallback.
    validate = candidate_validator(episode, 5)
    empty_camera = proposal(hook_line_id="S003")
    assert validate(CandidateList(episode=99, candidates=[empty_camera])) == []
    assert score_candidates(episode,CandidateList(episode=99,candidates=[empty_camera]))[0].hook_text == "Close-up on Victor."
    forbidden = proposal(start_shot_id="S001", hook_line_id="S001")
    assert "unknown_line_id" in validate(CandidateList(episode=99, candidates=[forbidden]))[0]


@pytest.mark.parametrize("selector,expected", [
    ("all", list(range(1, 51))), ("1-3", [1, 2, 3]), ("1,4,7", [1, 4, 7]),
    ("5", [5]), ("3,1-2,2", [1, 2, 3]),
])
def test_episode_selector(selector, expected):
    assert parse_episode_selector(selector, list(range(1, 51))) == expected


@pytest.mark.parametrize("selector", ["", "3-1", "1,bad", "51", "1-999999", "-1"])
def test_invalid_episode_selector(selector):
    with pytest.raises(ValueError):
        parse_episode_selector(selector, list(range(1, 51)))


def test_previous_loglines_are_only_two_earlier_episodes(episode):
    earlier = [episode.model_copy(update={"number": number, "logline": f"Logline {number}"}) for number in (96, 97, 98)]
    assert previous_loglines(episode, earlier + [episode]) == "Episode 97: Logline 97\nEpisode 98: Logline 98"


def test_run_builds_bible_with_same_client_and_continues_after_failure(tmp_path, episode, bible):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    for number in (98, 99):
        item = episode.model_copy(update={"number": number})
        (parsed / f"episode_{number:03d}.json").write_text(item.model_dump_json(), encoding="utf-8")
    # Existing synthetic aliases must be covered by the story bible validator.
    from src.models import Character
    bible.characters = [Character(name=name, aliases=[name], role="Synthetic", relationships="Synthetic",confidence="high",evidence="E099_L004") for name in ("ANNA", "VICTOR")]
    bad = CandidateList(episode=98, candidates=[proposal(start_shot_id="unknown")])
    valid = CandidateList(episode=99, candidates=[proposal()])
    client = client_for(tmp_path, [bible.model_dump_json(), json.dumps([bad.model_dump()]), bad.model_dump_json(), json.dumps([valid.model_dump()])], max_retries=1)
    config = {"parsed_dir": str(parsed), "output_dir": str(tmp_path / "outputs")}
    stats = run_candidates(config, "all", client)
    assert len(client.calls) == 4
    assert stats["episodes_ok"] == [99]
    assert stats["episodes_failed"] == [98]
    output = tmp_path / "outputs"
    assert (output / "story_bible.json").exists()
    assert (output / "candidates" / "episode_099.json").exists()
    assert "E098" in (output / "candidates_errors.log").read_text(encoding="utf-8")
    with (output / "candidates_all.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "E099_C01"


def prepare_run(tmp_path, episode, bible, numbers):
    from src.models import Character
    parsed=tmp_path/"parsed"
    parsed.mkdir()
    output=tmp_path/"outputs"
    output.mkdir()
    bible.characters=[Character(name=name,aliases=[name],role="Synthetic",relationships="Synthetic",confidence="high",evidence="E099_L004") for name in ("ANNA","VICTOR")]
    (output/"story_bible.json").write_text(bible.model_dump_json(),encoding="utf-8")
    for number in numbers:
        item=episode.model_copy(deep=True)
        item.number=number
        for line in item.lines:
            line.episode=number
            line.id=line.id.replace("E099",f"E{number:03d}")
        for shot in item.shots:
            shot.episode=number
            shot.id=shot.id.replace("E099",f"E{number:03d}")
            for line in shot.lines:
                line.episode=number
                line.id=line.id.replace("E099",f"E{number:03d}")
        (parsed/f"episode_{number:03d}.json").write_text(item.model_dump_json(),encoding="utf-8")
    return {"parsed_dir":str(parsed),"output_dir":str(output)}


def test_resume_skips_valid_files_and_force_processes_them(tmp_path, episode, bible):
    config=prepare_run(tmp_path,episode,bible,[99])
    response=json.dumps([CandidateList(episode=99,candidates=[proposal()]).model_dump()])
    client=client_for(tmp_path,[response,response],use_cache=False)
    run_candidates(config,"99",client)
    path=Path(config["output_dir"])/"candidates/episode_099.json"
    assert load_saved_candidates(path,episode,5)
    stats=run_candidates(config,"99",client)
    assert stats["episodes_skipped"]==[99]
    assert len(client.calls)==1
    bible_path=Path(config["output_dir"])/"story_bible.json"
    saved_bible=bible_path.read_text(encoding="utf-8")
    bible_path.write_text("not JSON",encoding="utf-8")
    assert run_candidates(config,"99",client)["episodes_skipped"]==[99]
    assert len(client.calls)==1
    bible_path.write_text(saved_bible,encoding="utf-8")
    run_candidates(config,"99",client,force=True)
    assert len(client.calls)==2
    saved=json.loads(path.read_text(encoding="utf-8"))
    saved[0]["hook_text"]="tampered"
    path.write_text(json.dumps(saved),encoding="utf-8")
    assert load_saved_candidates(path,episode,5) is None


def test_daily_quota_stops_entire_run_without_later_episodes(tmp_path, episode, bible):
    config=prepare_run(tmp_path,episode,bible,[99,100])
    class DailyLimit(Exception):
        status_code=429
    client=client_for(tmp_path,[DailyLimit("RESOURCE_EXHAUSTED daily limit")])
    stats=run_candidates(config,"all",client)
    assert len(client.calls)==1
    assert stats["episodes_failed"]==[99]
    assert stats["episodes_not_attempted"]==[100]
    assert stats["stopped_daily_quota"]
    error=(Path(config["output_dir"])/"candidates_errors.log").read_text(encoding="utf-8")
    assert "E100" not in error
