import csv
import subprocess
import sys
from pathlib import Path

import pytest

from src.models import Episode
from src.clips import clip_span
from src.parser import parse_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample_episode.txt"


@pytest.fixture
def episode():
    return parse_file(FIXTURE)


def test_episode_metadata(episode):
    assert episode.number == 99
    assert episode.logline == "A test heiress confronts a test villain."
    assert episode.continuity_notes == "Synthetic fixture for unit tests."
    assert len(episode.scenes) == 2
    assert len(episode.shots) == 5
    assert len(episode.lines) == 10
    assert episode.duration == 30.0
    assert episode.warnings == []
    assert episode.scenes[0].heading == "INT. TEST HALL – NIGHT"


def test_anna_text_and_assignment(episode):
    anna = next(line for line in episode.lines if line.speaker == "ANNA")
    assert anna.emotion == "Shocked, Breathless"
    assert anna.text == "You were here all along?"
    assert not anna.off_screen
    assert anna in episode.shots[1].lines


def test_victor_assignments(episode):
    always = next(line for line in episode.lines if line.text == "Always.")
    go = next(line for line in episode.lines if line.text == "Go.")
    assert always.off_screen
    assert always.emotion == "Cold"
    assert always in episode.shots[3].lines
    assert go.emotion is None
    assert go in episode.shots[3].lines


def test_clean_boundaries_and_sound(episode):
    assert episode.shots[1].clean_end
    assert not episode.shots[2].clean_start
    assert episode.shots[0].clean_start
    sound = next(line for line in episode.lines if line.type == "sound")
    assert sound in episode.shots[4].lines


def test_order_unique_attachment_and_json_roundtrip(episode):
    assert [line.start for line in episode.lines] == sorted(line.start for line in episode.lines)
    assert episode.shots[1].id == "E099_S002"
    attached = [line.id for shot in episode.shots for line in shot.lines]
    expected = [line.id for line in episode.lines if line.type != "camera"]
    assert sorted(attached) == sorted(expected)
    assert len(attached) == len(set(attached))
    assert Episode.model_validate_json(episode.model_dump_json()) == episode


def test_missing_timing_recovers_next_block(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8")
    text = text.replace("Start: (00:01.00) | End: (00:03.50)", "")
    path = tmp_path / "malformed.txt"
    path.write_text(text, encoding="utf-8")
    episode = parse_file(path)
    assert episode.warnings
    assert len(episode.shots) == 5
    assert len(episode.lines) == 9
    assert any(line.text == "Go." for line in episode.lines)


def test_fallback_shots(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8").replace("**CAMERA:**", "**ACTION:**")
    path = tmp_path / "no_cameras.txt"
    path.write_text(text, encoding="utf-8")
    episode = parse_file(path)
    assert len(episode.shots) == 2
    assert episode.shots[0].start == 0
    assert episode.shots[1].end == 30
    assert sum(len(shot.lines) for shot in episode.shots) == len(episode.lines)
    assert any("fallback" in warning for warning in episode.warnings)


def test_reversed_outside_and_odd_header_warnings(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8")
    text = text.replace("**VICTOR:**", "**odd ? header:**")
    text = text.replace("Start: (00:16.00) | End: (00:16.50)", "Start: (00:32.00) | End: (00:31.00)")
    path = tmp_path / "odd.txt"
    path.write_text(text, encoding="utf-8")
    episode = parse_file(path)
    assert any("start 32 > end 31" in warning for warning in episode.warnings)
    assert any("outside scene" in warning for warning in episode.warnings)
    assert any("unknown/odd header" in warning for warning in episode.warnings)
    assert episode.duration == 31


def test_overlap_ties_nearest_ties_and_voiceover(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8")
    text = text.replace("Start: (00:05.00) | End: (00:09.50)", "Start: (00:06.00) | End: (00:10.00)")
    text = text.replace("(O.S.)", "(V.O.)")
    text = text.replace("Start: (00:16.00) | End: (00:16.50)", "Start: (00:10.40) | End: (00:10.60)")
    path = tmp_path / "ties.txt"
    path.write_text(text, encoding="utf-8")
    episode = parse_file(path)
    assert any(line.speaker == "ANNA" for line in episode.shots[1].lines)
    assert any(line.text == "Go." for line in episode.shots[2].lines)
    assert next(line for line in episode.lines if line.text == "Always.").off_screen


def test_real_gap_shape_using_synthetic_fixture(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8")
    text = text.replace("Start: (00:04.00) | End: (00:08.00)", "Start: (00:33.70) | End: (00:35.30)")
    text = text.replace("Start: (00:08.00) | End: (00:10.00)", "Start: (00:36.50) | End: (00:39.00)")
    text = text.replace("Start: (00:05.00) | End: (00:09.50)", "Start: (00:35.40) | End: (00:37.70)")
    path = tmp_path / "gap.txt"
    path.write_text(text, encoding="utf-8")
    episode = parse_file(path)
    target = next(shot for shot in episode.shots if shot.start == 36.5)
    assert any(line.speaker == "ANNA" for line in target.lines)


def test_empty_file_and_missing_scene_metadata(tmp_path):
    path = tmp_path / "episode-7.txt"
    path.write_text("", encoding="utf-8")
    assert parse_file(path).warnings
    text = FIXTURE.read_text(encoding="utf-8").replace("Scene end: 00:20.00", "Scene end: nonsense")
    path.write_text(text, encoding="utf-8")
    episode = parse_file(path)
    assert episode.scenes[0].end == 16.5
    assert episode.warnings


def test_boundary_tolerance(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8")
    text = text.replace("Start: (00:05.00) | End: (00:09.50)", "Start: (00:07.90) | End: (00:08.10)")
    path = tmp_path / "tolerance.txt"
    path.write_text(text, encoding="utf-8")
    episode = parse_file(path)
    assert episode.shots[1].clean_end
    assert episode.shots[2].clean_start


def test_cli_numeric_order_outputs_and_render(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "parsed"
    text = FIXTURE.read_text(encoding="utf-8")
    (source / "alpha.txt").write_text(text.replace("Episode 99", "Episode 10"), encoding="utf-8")
    (source / "zeta.txt").write_text(text.replace("Episode 99", "Episode 2"), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "src.parser", "--input", str(source), "--out", str(output)],
        capture_output=True, text=True, check=True,
    )
    with (output / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["episode"] for row in rows] == ["2", "10"]
    assert rows[0]["n_unclean_boundaries"] == "1"
    assert "TOTAL: 2 episodes | 10 shots | 20 lines | 0 warnings" in result.stdout
    assert (output / "episode_010.json").is_file()
    rendered = subprocess.run(
        [sys.executable, "-m", "src.render", "--episode", "2", "--parsed-dir", str(output)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert "EPISODE 2" in rendered.stdout
    assert "[S002]" in rendered.stdout


def test_duplicate_episode_numbers_warn_and_continue(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "parsed"
    text = FIXTURE.read_text(encoding="utf-8")
    for name in ("alpha.txt", "beta.txt"):
        (source / name).write_text(text, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "src.parser", "--input", str(source), "--out", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert (output / "episode_099.json").is_file()
    assert (output / "episode_099_duplicate_002.json").is_file()
    assert "Duplicate episode number" in result.stdout


@pytest.mark.parametrize("second_camera,line_time", [
    ("Start: (00:00.30) | End: (00:00.50)", "Start: (00:00.20) | End: (00:00.40)"),
    ("Start: (00:00.50) | End: (00:00.70)", "Start: (00:00.40) | End: (00:00.40)"),
])
def test_centisecond_ties_choose_earlier_shot(tmp_path, second_camera, line_time):
    text = FIXTURE.read_text(encoding="utf-8")
    text = text.replace("Start: (00:00.00) | End: (00:04.00)", "Start: (00:00.10) | End: (00:00.30)")
    text = text.replace("Start: (00:04.00) | End: (00:08.00)", second_camera)
    text = text.replace("Start: (00:05.00) | End: (00:09.50)", line_time)
    path = tmp_path / "decimal_ties.txt"
    path.write_text(text, encoding="utf-8")
    episode = parse_file(path)
    assert any(line.speaker == "ANNA" for line in episode.shots[0].lines)


def test_effective_spans_and_clip_span(episode):
    assert episode.shots[1].effective_end == 9.5
    assert episode.shots[3].effective_start == 10.2
    assert episode.shots[3].effective_end == 16.5
    assert (episode.shots[4].effective_start, episode.shots[4].effective_end) == (20, 30)
    assert clip_span(episode, "E099_S002", "E099_S004") == (4, 16.5)
    assert clip_span(episode, "E099_S004", "E099_S004") == (10.2, 16.5)
    with pytest.raises(ValueError):
        clip_span(episode, "E099_S004", "E099_S002")
    with pytest.raises(ValueError):
        clip_span(episode, "E001_S002", "E099_S004")
    with pytest.raises(ValueError):
        clip_span(episode, "E099_S002", "missing")
