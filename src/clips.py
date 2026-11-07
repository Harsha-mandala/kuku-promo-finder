"""Resolve shot selections to safe content-inclusive timestamps."""

from .models import Episode


def clip_span(episode: Episode, start_shot_id: str, end_shot_id: str) -> tuple[float, float]:
    """Return effective bounds for an ordered, inclusive shot range."""
    positions = {shot.id: position for position, shot in enumerate(episode.shots)}
    if start_shot_id not in positions or end_shot_id not in positions:
        raise ValueError(f"Shot IDs must belong to episode {episode.number}")
    start = positions[start_shot_id]
    end = positions[end_shot_id]
    if start > end:
        raise ValueError("Start shot comes after end shot")
    return episode.shots[start].effective_start, episode.shots[end].effective_end
