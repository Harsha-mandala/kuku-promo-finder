"""Validate and fix continuous shot ranges without model calls."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .clips import clip_span
from .models import Episode


class ValidatedPromo(BaseModel):
    """Keep source proposals alongside the final shot selection and audit trail."""

    episode: int
    id: str
    status: Literal["ok", "fixed", "flagged", "rejected"] = "rejected"
    orig_start_shot_id: str = ""
    orig_end_shot_id: str = ""
    orig_span: tuple[float, float] | None = None
    orig_duration: float | None = None
    orig_hook_offset_s: float | None = None
    start_shot_id: str = ""
    end_shot_id: str = ""
    final_span: tuple[float, float] | None = None
    duration_s: float | None = None
    cliffhanger_end_gap_s: float | None = None
    hook_offset_s: float | None = None
    hook_line_id: str | None = None
    cliffhanger_line_id: str | None = None
    hook_text: str = ""
    cliffhanger_text: str = ""
    start_clean: bool = False
    end_clean: bool = False
    ends_on_episode_final_beat: bool = False
    why_it_works: str = ""
    self_score: int = 0
    fix_log: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    original_candidate: dict = Field(default_factory=dict)


def is_dead_shot(shot) -> bool:
    """Only remove pure transitions with no attached dialogue or action."""
    if any(line.type in ("dialogue", "action") for line in shot.lines):
        return False
    text = shot.camera_text.strip()
    return bool(re.fullmatch(
        r"(?:cut(?:s)?\s+to\s+black|fade(?:s)?\s+(?:to\s+black|out)|black(?:\s+(?:screen|frame))?|"
        r"(?:hard\s+)?cut\s+to(?:\s+.+)?|(?:cross[ -]?)?dissolve(?:\s+to(?:\s+.+)?)?|"
        r"(?:scene\s+)?transition(?:\s+to(?:\s+.+)?)?)[.!\s]*", text, re.IGNORECASE))


def canonical_id(episode: Episode, value: str) -> str:
    match = re.fullmatch(r"\[?([LS])(\d+)\]?", value.strip())
    if match:
        return f"E{episode.number:03d}_{match[1]}{int(match[2]):03d}"
    return value.strip()


def normalize_quote(value: str) -> str:
    """Normalize legacy formatting only; never approximate a quote."""
    value = re.sub(r"^(?:ACTION|SOUND(?: FX)?|SFX)\s+\d+:\d+(?:\.\d+)?:\s*", "", value.strip())
    return " ".join(value.strip('"“”').split())


def validate_candidate(episode: Episode, raw: dict, index: int = 1) -> ValidatedPromo:
    """Apply ordered repairs, preserving every original proposal."""
    promo = ValidatedPromo(
        episode=episode.number, id=str(raw.get("candidate_id", f"E{episode.number:03d}_C{index:02d}")),
        orig_start_shot_id=str(raw.get("start_shot_id", "")), orig_end_shot_id=str(raw.get("end_shot_id", "")),
        hook_text=str(raw.get("hook_text", raw.get("hook_line", ""))),
        cliffhanger_text=str(raw.get("cliffhanger_text", raw.get("cliffhanger_line", ""))),
        why_it_works=str(raw.get("why_it_works", "")), original_candidate=raw,
    )
    try:
        promo.self_score = max(0, min(5, int(raw.get("self_score", 0))))
    except (ValueError, TypeError):
        promo.fix_log.append("invalid self_score; treated as 0")
    positions = {shot.id: i for i, shot in enumerate(episode.shots)}
    sources = {}
    for i, shot in enumerate(episode.shots):
        for line in shot.lines:
            sources[line.id] = (i, line.start, line.text, line.type)
        if not shot.lines:
            sources[shot.id] = (i, shot.effective_start, shot.camera_text, "camera")
    start = canonical_id(episode, promo.orig_start_shot_id)
    end = canonical_id(episode, promo.orig_end_shot_id)
    promo.start_shot_id, promo.end_shot_id = start, end
    if start not in positions or end not in positions or positions[start] > positions[end]:
        promo.fix_log.append("rejected: unknown shot ID or start after end")
        return promo
    first, last = positions[start], positions[end]
    promo.orig_span = clip_span(episode, start, end)
    promo.orig_duration = round(promo.orig_span[1] - promo.orig_span[0], 3)

    def refresh():
        promo.start_shot_id = episode.shots[first].id
        promo.end_shot_id = episode.shots[last].id
        promo.final_span = clip_span(episode, promo.start_shot_id, promo.end_shot_id)
        promo.duration_s = round(promo.final_span[1] - promo.final_span[0], 3)
        promo.start_clean = episode.shots[first].clean_start
        promo.end_clean = episode.shots[last].clean_end
        if promo.hook_line_id in sources:
            promo.hook_offset_s = round(sources[promo.hook_line_id][1] - promo.final_span[0], 3)
        non_dead = [i for i, shot in enumerate(episode.shots) if not is_dead_shot(shot)]
        promo.ends_on_episode_final_beat = bool(non_dead and last == non_dead[-1])

    def flag(reason):
        if reason not in promo.flags:
            promo.flags.append(reason)
            promo.fix_log.append(reason)

    def resolve(kind):
        given = raw.get(f"{kind}_line_id")
        if given:
            source_id = canonical_id(episode, str(given))
            if source_id not in sources or not first <= sources[source_id][0] <= last:
                raise ValueError(f"{kind} ID unknown or outside clip")
            return source_id
        quote = normalize_quote(str(raw.get(f"{kind}_line", "")))
        matches = [key for key, source in sources.items() if first <= source[0] <= last and normalize_quote(source[2]) == quote]
        ordered = sorted(((key, source) for key, source in sources.items()
                          if first <= source[0] <= last and source[3] != "camera"),
                         key=lambda pair: (pair[1][1], pair[0]))
        for i, (key, _) in enumerate(ordered):
            for count in (2, 3):
                group = ordered[i:i + count]
                if len(group) == count and normalize_quote(" ".join(item[1][2] for item in group)) == quote:
                    matches.append(key)
        matches = list(dict.fromkeys(matches))
        if not quote or len(matches) != 1:
            raise ValueError(f"legacy {kind} quote has {len(matches)} exact source matches inside clip")
        promo.fix_log.append(f"legacy {kind} quote resolved exactly to {matches[0]}")
        return matches[0]

    try:
        promo.hook_line_id = resolve("hook")
        promo.hook_text = sources[promo.hook_line_id][2]
        promo.cliffhanger_line_id = resolve("cliffhanger")
        promo.cliffhanger_text = sources[promo.cliffhanger_line_id][2]
    except ValueError as error:
        refresh()
        promo.orig_hook_offset_s = promo.hook_offset_s
        promo.fix_log.append(f"rejected: {error}")
        return promo
    refresh()
    promo.orig_hook_offset_s = promo.hook_offset_s
    changed = False

    def move(new_first, new_last, reason):
        nonlocal first, last, changed
        first, last = new_first, new_last
        changed = True
        promo.fix_log.append(reason)
        refresh()

    def span(a, b):
        return clip_span(episode, episode.shots[a].id, episode.shots[b].id)

    def opening_hook(opening):
        start_time = episode.shots[opening].effective_start
        choices = [line for line in episode.shots[opening].lines
                   if line.type in ("dialogue", "action") and 0 <= round(line.start - start_time, 3) <= 3]
        return min(choices, key=lambda line: (line.type != "dialogue", line.start, line.id), default=None)

    def reassign_hook(line):
        promo.hook_line_id, promo.hook_text = line.id, line.text
        flag("hook_reassigned")
        refresh()

    # 1. Hook at the opening boundary.
    if promo.hook_offset_s > 3:
        move(sources[promo.hook_line_id][0], last, "start moved to hook shot")
    # 2. Recover the beginning of a sentence, at most two shots earlier.
    for _ in range(2):
        if promo.start_clean or first == 0:
            break
        move(first - 1, last, "unsafe start moved back one shot")
    if promo.hook_offset_s > 3:
        replacement = opening_hook(first)
        if replacement:
            reassign_hook(replacement)
    # 3. Remove trailing dead camera-only transitions.
    while last >= first and is_dead_shot(episode.shots[last]):
        if last == first:
            promo.fix_log.append("rejected: clip contains only dead transition shots")
            return promo
        move(first, last - 1, "dropped trailing dead shot")
    # 4. Prefer an adjacent clean end, then consider full duration adjustment.
    if not promo.end_clean:
        if last + 1 < len(episode.shots) and episode.shots[last + 1].clean_end and not is_dead_shot(episode.shots[last + 1]) and span(first, last + 1)[1] - span(first, last + 1)[0] <= 60:
            move(first, last + 1, "end extended to next clean shot")
        elif last > first and episode.shots[last - 1].clean_end and sources[promo.hook_line_id][0] <= last - 1:
            move(first, last - 1, "end pulled back to previous clean shot")
    # 5. Latest earlier clean boundary within the hard maximum.
    if promo.duration_s > 60:
        for new_last in range(last - 1, first - 1, -1):
            a, b = span(first, new_last)
            if episode.shots[new_last].clean_end and not is_dead_shot(episode.shots[new_last]) and 0 < b - a <= 60 and sources[promo.hook_line_id][0] <= new_last:
                move(first, new_last, "end trimmed to latest clean boundary within 60 s")
                break
    # 6. Extend until the first clean boundary meeting the minimum.
    extended_short = False
    if promo.duration_s < 20:
        for new_last in range(last + 1, len(episode.shots)):
            a, b = span(first, new_last)
            if 20 <= b - a <= 60 and b <= episode.duration and episode.shots[new_last].clean_end and not is_dead_shot(episode.shots[new_last]):
                move(first, new_last, "short clip extended to next clean boundary at least 20 s")
                extended_short = True
                break
        if promo.duration_s < 20:
            for new_first in range(first - 1, -1, -1):
                a, b = span(new_first, last)
                if 20 <= b - a <= 60 and episode.shots[new_first].clean_start:
                    move(new_first, last, "short clip extended backward to nearest clean boundary at least 20 s")
                    break

    # Changed endings must still cite a real beat in the final selection.
    cliff_outside = not first <= sources[promo.cliffhanger_line_id][0] <= last
    if cliff_outside or extended_short:
        lines = [line for shot in episode.shots[first:last + 1] for line in shot.lines]
        choices = [line for line in lines if line.type == "dialogue"]
        if not choices:
            choices = [line for line in lines if line.type == "action"]
        if choices:
            line = max(choices, key=lambda item: (item.end, item.start, item.id))
            promo.cliffhanger_line_id, promo.cliffhanger_text = line.id, line.text
            flag("cliffhanger_reassigned")
        else:
            flag("cliffhanger_missing")
    # Preserve the unresolved beat instead of running into a later resolution.
    cliff_id = promo.cliffhanger_line_id
    cliff_shot = sources[cliff_id][0]
    cliff_line = next((line for line in episode.shots[cliff_shot].lines if line.id == cliff_id), None)
    cliff_end = cliff_line.end if cliff_line else episode.shots[cliff_shot].effective_end
    if promo.final_span[1] - cliff_end > 3.0:
        trimmed_last = cliff_shot
        if cliff_shot + 1 <= last:
            reaction = episode.shots[cliff_shot + 1]
            if (reaction.effective_end - reaction.effective_start <= 3.0
                    and not any(line.type == "dialogue" for line in reaction.lines)
                    and not is_dead_shot(reaction)):
                trimmed_last += 1
        a, b = span(first, trimmed_last)
        if trimmed_last < last and b - a >= 20:
            move(first, trimmed_last, "trimmed after cliffhanger")
    refresh()
    if not 0 <= promo.hook_offset_s <= 3 or not first <= sources[promo.hook_line_id][0] <= last:
        # Check the finished span, including duration and cliffhanger repairs.
        for opening in range(first, min(last, cliff_shot) + 1):
            a, b = span(opening, last)
            if opening != first and (not episode.shots[opening].clean_start or not 20 <= round(b - a, 3) <= 60):
                continue
            replacement = opening_hook(opening)
            if replacement:
                if opening != first:
                    move(opening, last, "start advanced to first valid opening hook")
                reassign_hook(replacement)
                break
        else:
            flag("hook_reassignment_unavailable")
    promo.cliffhanger_end_gap_s = round(promo.final_span[1] - cliff_end, 3)
    if sources[promo.hook_line_id][3] in ("sound", "camera"):
        flag("weak_hook_kind")
    hard = []
    if not 20 <= promo.duration_s <= 60:
        hard.append("duration outside 20-60 s")
    if promo.hook_offset_s is None or not 0 <= promo.hook_offset_s <= 3:
        hard.append("hook offset outside 0-3 s")
    if not first <= sources[promo.hook_line_id][0] <= last:
        hard.append("hook outside final clip")
    if not first <= sources[promo.cliffhanger_line_id][0] <= last:
        hard.append("cliffhanger outside final clip")
    if promo.final_span[0] >= promo.final_span[1] or promo.final_span[1] > episode.duration:
        hard.append("invalid span or past episode end")
    if not promo.start_clean:
        flag("unsafe_start")
    if not promo.end_clean:
        flag("unsafe_end")
    if hard:
        promo.fix_log.extend(f"rejected: {reason}" for reason in hard)
    else:
        soft_flags = set(promo.flags) - {"hook_reassigned"}
        promo.status = "flagged" if soft_flags else "fixed" if changed else "ok"
    return promo


def dedupe_promos(promos: list[ValidatedPromo]) -> list[ValidatedPromo]:
    """Reject time-overlapping lower scores; ties prefer shorter clips."""
    eligible = [item for item in promos if item.status != "rejected"]
    eligible.sort(key=lambda item: (-item.self_score, item.duration_s, item.id))
    kept = []
    for item in eligible:
        duplicate = None
        for other in kept:
            if item.episode != other.episode:
                continue
            shared = max(0, min(item.final_span[1], other.final_span[1]) - max(item.final_span[0], other.final_span[0]))
            if shared / min(item.duration_s, other.duration_s) > 0.5:
                duplicate = other
                break
        if duplicate:
            item.status = "rejected"
            item.fix_log.append(f"rejected: duplicate of {duplicate.id}")
        else:
            kept.append(item)
    return promos


def run_validation(candidates: Path, tag: str, parsed: Path = Path("parsed"), output: Path = Path("outputs"), *, episode_numbers: set[int] | None = None) -> list[ValidatedPromo]:
    """Write separate audited outputs; never modify candidate inputs."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("tag must contain letters, numbers, underscore or hyphen")
    target = output / f"validated_{tag}"
    target.mkdir(parents=True, exist_ok=True)
    all_promos, errors = [], []
    files = sorted(candidates.glob("episode_*.json"))
    if episode_numbers is not None:
        files = [path for path in files if path.stem.split("_")[-1].isdigit() and int(path.stem.split("_")[-1]) in episode_numbers]
    if not files:
        raise ValueError(f"No candidate episode files found in {candidates}")
    for path in files:
        try:
            number = int(path.stem.split("_")[-1])
            episode = Episode.model_validate_json((parsed / path.name).read_text(encoding="utf-8"))
            if episode.number != number:
                raise ValueError("Parsed episode number differs from filename")
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Candidate file must contain a list")
            promos = [validate_candidate(episode, item, i) for i, item in enumerate(raw, 1) if isinstance(item, dict)]
            for i, item in enumerate(raw, 1):
                if not isinstance(item, dict):
                    errors.append(f"E{number:03d} item {i}: rejected: candidate must be an object")
            dedupe_promos(promos)
            (target / path.name).write_text(json.dumps([item.model_dump() for item in promos], ensure_ascii=False, indent=2), encoding="utf-8")
            all_promos.extend(promos)
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"{path.name}: rejected: {error}")
    fields = ["episode", "id", "status", "start_shot_id", "end_shot_id", "final_span", "duration_s", "hook_text", "cliffhanger_text", "fix_log", "why_it_works", "self_score"]
    with (output / f"validated_{tag}_all.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in all_promos:
            row = item.model_dump(include=set(fields))
            row["fix_log"] = " | ".join(item.fix_log)
            writer.writerow(row)
    errors.extend(f"E{item.episode:03d} {item.id}: " + " | ".join(item.fix_log) for item in all_promos if item.status == "rejected")
    (output / f"rejections_{tag}.log").write_text("\n".join(errors) + ("\n" if errors else ""), encoding="utf-8")
    counts = {status: sum(item.status == status for item in all_promos) for status in ("ok", "fixed", "flagged", "rejected")}
    print(f"{tag}: {counts} | file/item errors: {len(errors) - counts['rejected']}")
    return all_promos


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--candidates", type=Path, required=True)
    cli.add_argument("--tag", required=True)
    cli.add_argument("--parsed", type=Path, default=Path("parsed"))
    cli.add_argument("--out", type=Path, default=Path("outputs"))
    args = cli.parse_args()
    try:
        run_validation(args.candidates, args.tag, args.parsed, args.out)
    except (OSError, ValueError) as error:
        cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
