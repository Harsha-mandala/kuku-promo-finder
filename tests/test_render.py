from pathlib import Path

from src.parser import parse_file
from src.render import render_episode, render_episode_compact

FIXTURE = Path(__file__).parent / "fixtures" / "sample_episode.txt"


def test_render_episode():
    rendered = render_episode(parse_file(FIXTURE))
    assert rendered.startswith("EPISODE 99 | duration 00:30.00")
    assert "[S002]" in rendered
    assert "[!end]" not in rendered
    assert "[!start]" in rendered
    assert 'ANNA (Shocked, Breathless) 00:05.00: "You were here all along?"' in rendered
    assert "VICTOR (O.S.) (Cold)" in rendered
    assert "SOUND 00:28.00: Thunder crashes." in rendered
    assert "[L004] ANNA" in rendered
    assert "[L002] ACTION" in rendered
    assert "[L010] SOUND" in rendered
    assert "[S002] 00:04.00-00:09.50 (cam 00:04.00-00:08.00)" in rendered
    assert "[S004] 00:10.20-00:16.50 (cam 00:11.00-00:15.00)" in rendered


def test_camera_text_truncated_without_changing_source():
    episode = parse_file(FIXTURE)
    episode.shots[0].camera_text = "x" * 200
    rendered = render_episode(episode)
    assert "x" * 117 + "..." in rendered
    assert "x" * 121 not in rendered
    assert len(episode.shots[0].camera_text) == 200


def test_external_dialogue_marks_effective_end_unsafe(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8")
    text = text.replace("Start: (00:16.00) | End: (00:16.50)", "Start: (00:08.90) | End: (00:10.20)")
    path = tmp_path / "external_dialogue.txt"
    path.write_text(text, encoding="utf-8")
    episode = parse_file(path)
    assert not episode.shots[1].clean_end
    rendered = render_episode(episode)
    header = next(row for row in rendered.splitlines() if row.startswith("[S002]"))
    assert "00:04.00-00:09.50" in header
    assert "[!end]" in header


def test_compact_camera_limit_and_short_sound_last_shot_exception():
    from src.models import Line
    episode = parse_file(FIXTURE)
    episode.shots[0].camera_text = "x" * 200
    early = Line(id="E099_L011", episode=99, scene=1, type="sound", start=2, end=2.5, text="Brief click.")
    final = Line(id="E099_L012", episode=99, scene=2, type="sound", start=29, end=29.5, text="Final gasp.")
    episode.shots[0].lines.append(early)
    episode.shots[-1].lines.append(final)
    compact = render_episode_compact(episode)
    assert "x" * 67 + "..." in compact
    assert "x" * 71 not in compact
    assert "Brief click." not in compact
    assert "Final gasp." in compact
    assert "Thunder crashes." in compact  # Two-second sound remains.
    assert "Brief click." in render_episode(episode)
