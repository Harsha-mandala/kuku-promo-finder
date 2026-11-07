import json
from pathlib import Path

import pytest

from src.clips import clip_span
from src.models import Episode, Line, Scene, Shot
from src.parser import parse_file
from src.validate import dedupe_promos, is_dead_shot, run_validation, validate_candidate

FIXTURE = Path(__file__).parent / "fixtures/sample_episode.txt"


def timeline(count=16, width=5):
    shots, lines = [], []
    for i in range(count):
        line = Line(id=f"E099_L{i + 1:03d}", episode=99, scene=1, type="dialogue", speaker="ANNA",
                    start=i * width + 0.2, end=i * width + 1, text=f"Synthetic line {i + 1}?")
        shots.append(Shot(id=f"E099_S{i + 1:03d}", index=i + 1, episode=99, scene=1,
                          start=i * width, end=(i + 1) * width, camera_text="Close-up on Anna.", lines=[line]))
        lines.append(line)
    return Episode(number=99, source_file="synthetic", duration=count * width, shots=shots, lines=lines,
                   scenes=[Scene(number=1, heading="INT. TEST HALL", start=0, end=count * width)])


def proposal(first=1, last=5, hook=1, cliff=5, **changes):
    return {"candidate_id": "E099_C01", "start_shot_id": f"E099_S{first:03d}", "end_shot_id": f"E099_S{last:03d}",
            "hook_line_id": f"E099_L{hook:03d}", "cliffhanger_line_id": f"E099_L{cliff:03d}",
            "self_score": 4, "why_it_works": "Synthetic unresolved question.", **changes}


def assert_whole_shots(ep, promo):
    assert promo.final_span == clip_span(ep, promo.start_shot_id, promo.end_shot_id)
    assert promo.duration_s == pytest.approx(promo.final_span[1] - promo.final_span[0])


def test_hook_six_seconds_late_moves_to_its_shot():
    ep = timeline(width=6)
    result = validate_candidate(ep, proposal(last=5, hook=2))
    assert result.orig_hook_offset_s == 6.2
    assert result.start_shot_id == "E099_S002"
    assert result.hook_offset_s == 0.2 and result.status == "fixed"
    assert "start moved to hook shot" in result.fix_log
    assert_whole_shots(ep, result)


@pytest.mark.parametrize("start,end,hook,cliff,expected_start,expected_hook,offset", [
    (2, 8, 6, 22, 2, 4, 1.4),
    (12, 17, 32, 49, 13, 35, 0.0),
])
def test_e018_timing_repairs_find_an_opening_hook(start, end, hook, cliff, expected_start, expected_hook, offset):
    ep = Episode.model_validate_json((FIXTURE.parent / "e018_timing.json").read_text(encoding="utf-8"))
    raw = {"start_shot_id": f"S{start:03d}", "end_shot_id": f"S{end:03d}",
           "hook_line_id": f"L{hook:03d}", "cliffhanger_line_id": f"L{cliff:03d}"}
    result = validate_candidate(ep, raw)
    assert result.status in {"fixed", "flagged"}
    assert result.start_shot_id == f"E018_S{expected_start:03d}"
    assert result.hook_line_id == f"E018_L{expected_hook:03d}"
    assert result.hook_offset_s == pytest.approx(offset)
    assert_whole_shots(ep, result)


def test_long_clip_trimmed_and_cliffhanger_reassigned():
    ep = timeline()
    result = validate_candidate(ep, proposal(last=15, cliff=15))
    assert result.orig_duration == 75
    assert result.duration_s == 60 and result.end_shot_id == "E099_S012"
    assert result.cliffhanger_line_id == "E099_L012"
    assert result.status == "flagged" and "cliffhanger_reassigned" in result.flags
    assert_whole_shots(ep, result)


def test_short_clip_extended_to_first_clean_boundary():
    ep = timeline(width=6)
    result = validate_candidate(ep, proposal(last=2, cliff=2))
    assert result.orig_duration == 12
    assert result.duration_s == 24 and result.end_shot_id == "E099_S004"
    assert result.cliffhanger_line_id == "E099_L004"
    assert result.status == "flagged"
    assert_whole_shots(ep, result)


def test_short_episode_is_rejected_without_crossing_episode_end():
    ep = timeline(count=2, width=6)
    result = validate_candidate(ep, proposal(last=2, cliff=2))
    assert result.status == "rejected" and result.final_span[1] == 12
    assert "rejected: duration outside 20-60 s" in result.fix_log
    assert_whole_shots(ep, result)


def test_trailing_black_shot_is_dropped_and_final_beat_marked():
    ep = timeline(count=6)
    ep.shots[-1].lines = []
    ep.shots[-1].camera_text = "Cut to black."
    result = validate_candidate(ep, proposal(last=6, cliff=5))
    assert result.end_shot_id == "E099_S005" and result.duration_s == 25
    assert result.ends_on_episode_final_beat and result.status == "fixed"
    assert "dropped trailing dead shot" in result.fix_log


def test_black_camera_cliffhanger_is_reassigned_after_drop():
    ep = timeline(count=6)
    ep.shots[-1].lines = []
    ep.shots[-1].camera_text = "Fade out."
    result = validate_candidate(ep, proposal(last=6, cliffhanger_line_id="E099_S006"))
    assert result.cliffhanger_line_id == "E099_L005"
    assert result.status == "flagged"


@pytest.mark.parametrize("text,expected", [("Cut to black.", True), ("Dissolve to next scene.", True),
    ("Close-up on Anna as the image fades to black.", False), ("Anna turns away.", False)])
def test_dead_transition_detection_is_conservative(text, expected):
    shot = timeline().shots[0]
    shot.lines = []
    shot.camera_text = text
    assert is_dead_shot(shot) == expected
    shot.lines = timeline().shots[0].lines
    assert not is_dead_shot(shot)


def test_eighty_percent_time_overlap_keeps_higher_score():
    ep = timeline()
    low = validate_candidate(ep, proposal(self_score=2))
    high = validate_candidate(ep, proposal(2, 6, 2, 6, candidate_id="E099_C02", self_score=5))
    dedupe_promos([low, high])
    assert high.status == "ok" and low.status == "rejected"
    assert low.fix_log[-1] == "rejected: duplicate of E099_C02"


def test_dedupe_tie_prefers_shorter_and_exact_half_is_allowed():
    ep = timeline()
    longer = validate_candidate(ep, proposal())
    shorter = validate_candidate(ep, proposal(2, 5, 2, 5, candidate_id="E099_C02"))
    dedupe_promos([longer, shorter])
    assert longer.status == "rejected" and shorter.status == "ok"
    first = validate_candidate(ep, proposal(1, 4, 1, 4))
    second = validate_candidate(ep, proposal(3, 6, 3, 6, candidate_id="E099_C02"))
    dedupe_promos([first, second])
    assert first.status == second.status == "ok"


def test_synthetic_fixture_extends_short_proposal_without_cutting_lines():
    ep = parse_file(FIXTURE)
    result = validate_candidate(ep, proposal(2, 4, 4, 8))
    assert result.duration_s == 26 and result.end_shot_id == "E099_S005"
    assert result.cliffhanger_text == "Go."
    assert_whole_shots(ep, result)


def test_unsafe_boundaries_flagged_if_adjacent_fixes_impossible():
    ep = timeline(count=4)
    ep.shots[0].clean_start = False
    ep.shots[-1].clean_end = False
    # Pulling back produces 15s and cannot extend to a clean >=20s boundary.
    ep.shots[-2].clean_end = False
    result = validate_candidate(ep, proposal(last=4, cliff=4))
    assert result.status == "flagged"
    assert {"unsafe_start", "unsafe_end"} <= set(result.flags)


def test_unsafe_start_can_move_one_shot_back_without_delaying_hook():
    ep = timeline(count=30, width=1)
    ep.shots[1].clean_start = False
    result = validate_candidate(ep, proposal(2, 25, 2, 25))
    assert result.start_shot_id == "E099_S001" and result.status == "fixed"
    assert result.hook_offset_s == 1.2


def test_cliffhanger_trim_takes_priority_over_extension_with_more_dialogue():
    ep = timeline()
    ep.shots[4].clean_end = False
    result = validate_candidate(ep, proposal())
    assert result.end_shot_id == "E099_S005" and not result.end_clean
    assert result.status == "flagged"
    assert "trimmed after cliffhanger" in result.fix_log
    assert "unsafe_end" in result.flags


def test_trim_after_cliffhanger_keeps_only_one_short_silent_reaction():
    ep = timeline(width=5)
    ep.shots[4].lines = []
    ep.shots[4].end = 23
    result = validate_candidate(ep, proposal(last=7, cliff=4))
    assert result.end_shot_id == "E099_S005"
    assert result.duration_s == 23
    assert "trimmed after cliffhanger" in result.fix_log


def test_trim_never_makes_clip_shorter_than_twenty_seconds():
    result = validate_candidate(timeline(), proposal(last=6, cliff=2))
    assert result.end_shot_id == "E099_S006"
    assert "trimmed after cliffhanger" not in result.fix_log


@pytest.mark.parametrize("kind", ["sound", "camera"])
def test_weak_hook_is_soft_flag_and_promo_survives(kind):
    ep = timeline()
    raw = proposal()
    if kind == "sound":
        ep.shots[0].lines[0].type = "sound"
    else:
        ep.shots[0].lines = []
        raw["hook_line_id"] = ep.shots[0].id
    result = validate_candidate(ep, raw)
    assert result.status == "flagged"
    assert "weak_hook_kind" in result.flags


def test_legacy_quote_resolution_is_exact_and_audited():
    ep = timeline()
    raw = proposal()
    raw.pop("hook_line_id")
    raw.pop("cliffhanger_line_id")
    raw.update(hook_line='"Synthetic line 1?"', cliffhanger_line="Synthetic line 5?")
    result = validate_candidate(ep, raw)
    assert result.status == "ok" and result.hook_line_id == "E099_L001"
    assert any("resolved exactly" in log for log in result.fix_log)
    raw["hook_line"] = "Invented paraphrase"
    assert validate_candidate(ep, raw).status == "rejected"


@pytest.mark.parametrize("changes", [{"hook_line_id": "L999"}, {"cliffhanger_line_id": "L010"},
    {"start_shot_id": "S999"}, {"start_shot_id": "S006"}])
def test_bad_source_ids_or_order_rejected(changes):
    assert validate_candidate(timeline(), proposal(**changes)).status == "rejected"


def test_cli_outputs_preserve_sources_and_include_rejections(tmp_path):
    ep = timeline()
    parsed, source, out = [tmp_path / name for name in ("parsed", "candidates", "outputs")]
    parsed.mkdir()
    source.mkdir()
    (parsed / "episode_099.json").write_text(ep.model_dump_json(), encoding="utf-8")
    path = source / "episode_099.json"
    path.write_text(json.dumps([proposal(), proposal(hook_line_id="L999", candidate_id="E099_C02")]), encoding="utf-8")
    before = path.read_bytes()
    promos = run_validation(source, "synthetic", parsed, out)
    assert path.read_bytes() == before
    assert len(promos) == 2
    assert (out / "validated_synthetic/episode_099.json").exists()
    assert "E099_C02" in (out / "rejections_synthetic.log").read_text()
    assert "final_span" in (out / "validated_synthetic_all.csv").read_text()


def test_proposal_near_episode_end_recovers_by_extending_start():
    ep = timeline()
    result = validate_candidate(ep, proposal(12, 16, 15, 16))
    assert result.status == "fixed"
    assert result.start_shot_id == "E099_S013" and result.duration_s == 20
    assert result.hook_line_id == "E099_L013" and result.hook_offset_s == 0.2
    assert "hook_reassigned" in result.fix_log
    assert_whole_shots(ep, result)


def test_mid_sentence_start_moves_back_and_reassigns_hook():
    ep = timeline()
    ep.shots[2].clean_start = False
    result = validate_candidate(ep, proposal(3, 8, 3, 8))
    assert result.status == "fixed" and result.start_clean
    assert result.start_shot_id == "E099_S002" and result.hook_line_id == "E099_L002"
    assert "hook_reassigned" in result.flags


def test_legacy_two_consecutive_lines_resolve_to_first_id():
    raw = proposal()
    raw.pop("hook_line_id")
    raw["hook_line"] = "Synthetic line 1? Synthetic line 2?"
    result = validate_candidate(timeline(), raw)
    assert result.hook_line_id == "E099_L001" and result.status == "ok"


def test_unsafe_start_search_stops_after_two_shots():
    ep = timeline()
    for index in (1, 2, 3):
        ep.shots[index].clean_start = False
    result = validate_candidate(ep, proposal(4, 9, 4, 9))
    assert result.start_shot_id == "E099_S002"
    assert result.status == "flagged" and "unsafe_start" in result.flags


def test_three_line_legacy_quote_resolves_to_first_id():
    raw = proposal()
    raw.pop("hook_line_id")
    raw["hook_line"] = "Synthetic line 1? Synthetic line 2? Synthetic line 3?"
    result = validate_candidate(timeline(), raw)
    assert result.hook_line_id == "E099_L001" and result.status == "ok"
