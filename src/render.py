"""Render parsed episodes as compact, numbered LLM input."""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from .models import Episode
from .timeutils import format_timestamp


def render_episode(episode: Episode, *, compact: bool = False) -> str:
    """Show shot ranges, camera descriptions, and assigned content."""
    rows = [
        f"EPISODE {episode.number} | duration {format_timestamp(episode.duration)}",
        f"LOGLINE: {episode.logline}",
    ]
    for shot in episode.shots:
        camera = " ".join(shot.camera_text.split())
        camera_limit = 70 if compact else 120
        if len(camera) > camera_limit:
            camera = camera[:camera_limit - 3].rstrip() + "..."
        span = f"{format_timestamp(shot.effective_start)}-{format_timestamp(shot.effective_end)}"
        if shot.effective_start != shot.start or shot.effective_end != shot.end:
            span += f" (cam {format_timestamp(shot.start)}-{format_timestamp(shot.end)})"
        header = f"[S{shot.index:03d}] {span} | CAM: {camera}"
        if not shot.clean_start:
            header += " [!start]"
        if not shot.clean_end:
            header += " [!end]"
        rows.append(header)
        for line in shot.lines:
            if compact and line.type == "sound" and line.end - line.start < 1 and shot.id != episode.shots[-1].id:
                continue
            content = " ".join(line.text.split())
            if line.type == "dialogue":
                label = line.speaker or "UNKNOWN"
                if line.off_screen:
                    label += " (O.S.)"
                if line.emotion:
                    label += f" ({line.emotion})"
                content = f'"{content}"'
            else:
                label = line.type.upper()
            short_id = line.id.rsplit("_", 1)[-1]
            rows.append(f"       [{short_id}] {label} {format_timestamp(line.start)}: {content}")
    return "\n".join(rows)


def render_episode_compact(episode: Episode) -> str:
    """Render LLM input with short cameras and fewer incidental sounds."""
    return render_episode(episode, compact=True)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--episode", type=int, required=True)
    cli.add_argument("--parsed-dir", type=Path, default=Path("parsed"))
    args = cli.parse_args()
    path = args.parsed_dir / f"episode_{args.episode:03d}.json"
    try:
        episode = Episode.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError) as error:
        cli.error(f"Cannot load {path}: {error}")
    print(render_episode(episode))


if __name__ == "__main__":
    main()
