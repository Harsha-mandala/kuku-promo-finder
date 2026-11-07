"""Deterministic, screenplay-only promo baselines, independent of gold labels."""

import random
import re
from pathlib import Path

import yaml

from .clips import clip_span
from .validate import ValidatedPromo, is_dead_shot


def default_rules():
    """Keep the keyword lists and weights in one authoritative config block."""
    path = Path(__file__).resolve().parents[1] / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config["evaluation"]["baseline_rules"]


DEFAULT_RULES = default_rules()


def windows(episode, minimum, maximum):
    """Enumerate clean whole-shot windows using the shared effective spans."""
    for first, shot in enumerate(episode.shots):
        if not shot.clean_start:
            continue
        for last in range(first, len(episode.shots)):
            end = episode.shots[last]
            start_s, end_s = clip_span(episode, shot.id, end.id)
            if (minimum <= end_s - start_s <= maximum and end.clean_end
                    and end_s <= episode.duration and not is_dead_shot(end)):
                yield first, last, start_s, end_s


def strong_emotion(line, rules):
    words = set(re.findall(r"[a-z]+", (line.emotion or "").lower()))
    return bool(words.intersection(rules["emotions"]))


def window_features(episode, window, rules):
    first, last, start, _ = window
    lines = sorted((line for shot in episode.shots[first:last + 1] for line in shot.lines),
                   key=lambda line: (line.start, line.end, line.id))
    pattern = r"\b(?:" + "|".join(re.escape(word) for word in rules["actions"]) + r")\w*\b"
    opening = [line for line in lines if 0 <= line.start - start <= 3]
    hook = any((line.type == "dialogue" and strong_emotion(line, rules)) or
               (line.type in ("action", "sound") and re.search(pattern, line.text, re.I))
               for line in opening)
    final_live = next((i for i in range(len(episode.shots) - 1, -1, -1)
                       if not is_dead_shot(episode.shots[i])), None)
    cliffhanger = last == final_live or bool(lines and (
        lines[-1].text.rstrip('"” ').endswith(("?", "!")) or strong_emotion(lines[-1], rules)))
    speakers = len({line.speaker for line in lines if line.speaker})
    words = sum(len(line.text.split()) for line in lines)
    density = sum(line.text.count("!") for line in lines) / max(1, words)
    return (rules["hook_weight"] * hook + rules["cliffhanger_weight"] * cliffhanger
            + rules["speaker_weight"] * min(speakers, 3) + rules["exclamation_weight"] * density)


def build_baseline(episode, system, *, seed=20261106, rules=None, window_cache=None):
    """Select up to three disjoint windows; never manufacture missing windows."""
    rules = DEFAULT_RULES | (rules or {})
    choices = list(window_cache) if window_cache is not None else list(windows(
        episode, 20 if system == "random" else 25, 60 if system == "random" else 45))
    if system == "random":
        random.Random(seed + episode.number).shuffle(choices)
    elif system == "rules_only":
        choices.sort(key=lambda window: (-window_features(episode, window, rules), window[2], window[3]))
    else:
        raise ValueError(f"Unknown baseline: {system}")
    selected = []
    for window in choices:
        if any(min(window[3], other[3]) > max(window[2], other[2]) for other in selected):
            continue
        selected.append(window)
        if len(selected) == 3:
            break
    promos = []
    for index, (first, last, start, end) in enumerate(selected, 1):
        lines = sorted((line for shot in episode.shots[first:last + 1] for line in shot.lines),
                       key=lambda line: (line.start, line.end, line.id))
        hook, cliff = (lines[0], lines[-1]) if lines else (None, None)
        promos.append(ValidatedPromo(
            episode=episode.number, id=f"E{episode.number:03d}_{system.upper()}{index:02d}", status="ok",
            orig_start_shot_id=episode.shots[first].id, orig_end_shot_id=episode.shots[last].id,
            start_shot_id=episode.shots[first].id, end_shot_id=episode.shots[last].id,
            orig_span=(start, end), final_span=(start, end), orig_duration=end-start, duration_s=end-start,
            start_clean=True, end_clean=True,
            hook_line_id=hook.id if hook else episode.shots[first].id,
            cliffhanger_line_id=cliff.id if cliff else episode.shots[last].id,
            hook_text=hook.text if hook else episode.shots[first].camera_text,
            cliffhanger_text=cliff.text if cliff else episode.shots[last].camera_text,
            hook_offset_s=hook.start-start if hook else 0,
            original_candidate={"baseline": system, "seed": seed},
        ))
    return promos
