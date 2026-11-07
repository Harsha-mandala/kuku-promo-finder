"""Parse screenplays and build continuous camera shots."""

import argparse
import csv
import re
from pathlib import Path

from .models import Episode, Line, Scene, Shot
from .timeutils import parse_timestamp

SCENE_HEADER = re.compile(r"^SCENE\s+(\d+)\s*$", re.IGNORECASE)
BLOCK_HEADER = re.compile(r"^\*\*(.+?)\*\*\s*$")
TIMING = re.compile(r"^Start:\s*\(([^()]+)\)\s*\|\s*End:\s*\(([^()]+)\)\s*$", re.IGNORECASE)


def _filename_number(path: Path) -> int:
    match = re.search(r"episode[-_ ]*(\d+)", path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _speaker(header: str, warnings: list[str], location: str) -> tuple[str, str | None, bool]:
    modifiers = re.findall(r"\(([^()]*)\)", header)
    name = re.sub(r"\([^()]*\)", "", header).strip()
    emotions = []
    off_screen = False
    for modifier in modifiers:
        marker = re.sub(r"[.\s]", "", modifier).upper()
        if marker in {"OS", "VO"}:
            off_screen = True
        elif modifier.strip():
            emotions.append(modifier.strip())
    if not re.fullmatch(r"[A-Z][A-Z0-9 '\-’.]*", name) or "(" in name or ")" in name:
        warnings.append(f"{location}: unknown/odd header {header!r}; treated as dialogue")
    return name or header, ", ".join(emotions) or None, off_screen


def _interval(line: Line | Shot) -> tuple[float, float]:
    return min(line.start, line.end), max(line.start, line.end)


def build_shots(episode: Episode) -> list[Shot]:
    """Assign each non-camera line by overlap, then by temporal distance."""
    cameras = [line for line in episode.lines if line.type == "camera"]
    shots = []
    if cameras:
        for camera in cameras:
            shots.append(Shot(
                id="", index=0, episode=episode.number, scene=camera.scene,
                start=camera.start, end=camera.end, camera_text=camera.text,
            ))
    else:
        episode.warnings.append("No camera blocks; created one fallback shot per scene")
        for scene in episode.scenes:
            shots.append(Shot(
                id="", index=0, episode=episode.number, scene=scene.number,
                start=scene.start, end=scene.end, camera_text=scene.heading,
            ))
    shots.sort(key=lambda shot: shot.start)
    for index, shot in enumerate(shots, start=1):
        shot.index = index
        shot.id = f"E{episode.number:03d}_S{index:03d}"

    for line in episode.lines:
        if line.type == "camera":
            continue
        start, end = _interval(line)
        best_shot = None
        best_overlap = -1.0
        best_distance = float("inf")
        for shot in shots:
            shot_start, shot_end = _interval(shot)
            # Input times have centisecond precision; rounding preserves exact ties.
            overlap = round(max(0.0, min(end, shot_end) - max(start, shot_start)), 2)
            distance = round(max(0.0, shot_start - end, start - shot_end), 2)
            if overlap > best_overlap or (overlap == 0 == best_overlap and distance < best_distance):
                best_shot = shot
                best_overlap = overlap
                best_distance = distance
        if best_shot is not None:
            best_shot.lines.append(line)

    owners = {line.id: shot.id for shot in shots for line in shot.lines}
    dialogue = [line for line in episode.lines if line.type == "dialogue"]
    for shot in shots:
        shot.clean_start = not any(
            owners[line.id] != shot.id
            and line.start < shot.effective_start - 0.15
            and line.end > shot.effective_start + 0.15
            for line in dialogue
        )
        shot.clean_end = not any(
            owners[line.id] != shot.id
            and line.start < shot.effective_end - 0.15
            and line.end > shot.effective_end + 0.15
            for line in dialogue
        )
    return shots


def parse_file(path: str | Path) -> Episode:
    """Read one UTF-8 screenplay, retaining warnings for recoverable errors."""
    path = Path(path)
    episode = Episode(number=_filename_number(path), source_file=path.name)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        episode.warnings.append("Invalid UTF-8 bytes replaced while reading")
    except OSError as error:
        episode.warnings.append(f"Could not read file: {error}")
        return episode
    rows = text.splitlines()
    scene: Scene | None = None
    scene_bounds: dict[int, set[str]] = {}
    index = 0
    while index < len(rows):
        row = rows[index].strip()
        location = f"line {index + 1}"
        if row == "--- SCRIPT END ---":
            break
        scene_match = SCENE_HEADER.fullmatch(row)
        block_match = BLOCK_HEADER.fullmatch(row)
        if row.startswith("EPISODE NUMBER:"):
            match = re.fullmatch(r"EPISODE NUMBER:\s*Episode\s+(\d+)", row, re.IGNORECASE)
            if match:
                episode.number = int(match.group(1))
            else:
                episode.warnings.append(f"{location}: invalid episode number; using filename number")
        elif row.startswith("EPISODE LOGLINE:"):
            episode.logline = row.split(":", 1)[1].strip()
        elif row.startswith("CONTINUITY NOTES:"):
            episode.continuity_notes = row.split(":", 1)[1].strip()
        elif scene_match:
            scene = Scene(number=int(scene_match.group(1)), heading="", start=0, end=0)
            episode.scenes.append(scene)
            scene_bounds[id(scene)] = set()
        elif scene is not None and row.lower().startswith("scene heading:"):
            scene.heading = row.split(":", 1)[1].strip()
        elif scene is not None and row.lower().startswith(("scene start:", "scene end:")):
            field = "start" if row.lower().startswith("scene start:") else "end"
            try:
                setattr(scene, field, parse_timestamp(row.split(":", 1)[1].strip()))
                scene_bounds[id(scene)].add(field)
            except (ValueError, OverflowError):
                episode.warnings.append(f"{location}: malformed scene {field} timestamp")
        elif block_match:
            header = block_match.group(1).strip()
            if not header.endswith(":"):
                episode.warnings.append(f"{location}: odd block header missing colon")
            header = header.removesuffix(":").strip()
            next_index = index + 1
            while next_index < len(rows) and not rows[next_index].strip():
                next_index += 1
            timing = TIMING.fullmatch(rows[next_index].strip()) if next_index < len(rows) else None
            try:
                if timing is None:
                    raise ValueError("missing timing")
                start = parse_timestamp(timing.group(1))
                end = parse_timestamp(timing.group(2))
            except (ValueError, OverflowError):
                episode.warnings.append(f"{location}: missing/malformed Start/End; skipped {header!r} block")
                index += 1
                continue
            if scene is None:
                scene = Scene(number=1, heading="Unspecified scene", start=0, end=0)
                episode.scenes.append(scene)
                scene_bounds[id(scene)] = set()
                episode.warnings.append(f"{location}: block before SCENE; created implicit scene 1")
            body = []
            next_index += 1
            while next_index < len(rows):
                content = rows[next_index].strip()
                if BLOCK_HEADER.fullmatch(content) or SCENE_HEADER.fullmatch(content) or content == "--- SCRIPT END ---":
                    break
                if content:
                    body.append(content)
                next_index += 1
            block_type = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound", "SFX": "sound", "SOUND": "sound"}.get(header.upper(), "dialogue")
            speaker, emotion, off_screen = None, None, False
            content = " ".join(body)
            if block_type == "dialogue":
                speaker, emotion, off_screen = _speaker(header, episode.warnings, location)
                if len(content) >= 2 and content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
            episode.lines.append(Line(
                id="", episode=episode.number, scene=scene.number, type=block_type,
                speaker=speaker, emotion=emotion, off_screen=off_screen,
                start=start, end=end, text=content,
            ))
            if start > end:
                episode.warnings.append(f"{location}: {header!r} start {start:g} > end {end:g}")
            index = next_index
            continue
        index += 1

    if episode.number == 0:
        episode.warnings.append("Missing episode number; using 0")
    episode.lines.sort(key=lambda line: line.start)
    for index, line in enumerate(episode.lines, start=1):
        line.id = f"E{episode.number:03d}_L{index:03d}"
        line.episode = episode.number
    for scene in episode.scenes:
        scene_lines = [line for line in episode.lines if line.scene == scene.number]
        for field in ("start", "end"):
            if field not in scene_bounds[id(scene)]:
                values = [getattr(line, field) for line in scene_lines]
                inferred = (min(values) if field == "start" else max(values)) if values else 0.0
                setattr(scene, field, inferred)
                episode.warnings.append(f"Scene {scene.number}: missing/invalid {field}; inferred {inferred:g}")
        if scene.start > scene.end:
            episode.warnings.append(f"Scene {scene.number}: start {scene.start:g} > end {scene.end:g}")
        for line in scene_lines:
            if min(line.start, line.end) < scene.start or max(line.start, line.end) > scene.end:
                episode.warnings.append(f"{line.id}: block outside scene {scene.number} bounds ({scene.start:g}-{scene.end:g})")
    episode.duration = max([0.0] + [scene.end for scene in episode.scenes] + [line.end for line in episode.lines])
    episode.shots = build_shots(episode)
    return episode


def episode_summary(episode: Episode) -> dict:
    """Return one CSV/table summary row."""
    return {
        "episode": episode.number, "file": episode.source_file,
        "duration_s": episode.duration, "n_scenes": len(episode.scenes),
        "n_shots": len(episode.shots), "n_lines": len(episode.lines),
        "n_dialogue": sum(line.type == "dialogue" for line in episode.lines),
        "n_unclean_boundaries": sum(int(not shot.clean_start) + int(not shot.clean_end) for shot in episode.shots),
        "n_warnings": len(episode.warnings),
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, default=Path("data"))
    cli.add_argument("--out", type=Path, default=Path("parsed"))
    args = cli.parse_args()
    if not args.input.is_dir():
        cli.error(f"Input directory does not exist: {args.input}")
    paths = sorted(args.input.glob("*.txt"))
    if not paths:
        cli.error(f"No *.txt files found in {args.input}")
    episodes = sorted((parse_file(path) for path in paths), key=lambda episode: episode.number)
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    occurrences: dict[int, int] = {}
    for episode in episodes:
        occurrence = occurrences.get(episode.number, 0) + 1
        occurrences[episode.number] = occurrence
        stem = f"episode_{episode.number:03d}"
        if occurrence > 1:
            stem += f"_duplicate_{occurrence:03d}"
            episode.warnings.append(f"Duplicate episode number; saved as {stem}.json to avoid overwriting")
        target = args.out / f"{stem}.json"
        target.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
        summaries.append(episode_summary(episode))
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print("Episode  Duration  Scenes  Shots  Lines  Dialogue  Unclean  Warnings")
    for row in summaries:
        print(f"{row['episode']:7d}  {row['duration_s']:8.2f}  {row['n_scenes']:6d}  {row['n_shots']:5d}  {row['n_lines']:5d}  {row['n_dialogue']:8d}  {row['n_unclean_boundaries']:7d}  {row['n_warnings']:8d}")
    print(f"TOTAL: {len(episodes)} episodes | {sum(len(ep.shots) for ep in episodes)} shots | {sum(len(ep.lines) for ep in episodes)} lines | {sum(len(ep.warnings) for ep in episodes)} warnings")
    for episode in episodes:
        for warning in episode.warnings:
            print(f"WARNING E{episode.number:03d} ({episode.source_file}): {warning}")


if __name__ == "__main__":
    main()
