"""Render only the selected clip, without series or generator context."""

from .models import Episode
from .timeutils import format_timestamp
from .validate import ValidatedPromo


def render_clip(ep: Episode, promo: ValidatedPromo, *, mark_lines: bool = False) -> str:
    """Keep full camera and line text; optional markers are for final reports."""
    positions = {shot.id: i for i, shot in enumerate(ep.shots)}
    first, last = positions[promo.start_shot_id], positions[promo.end_shot_id]
    rows = [f"CLIP {promo.id} | duration {promo.duration_s:.2f} s"]
    marked = {promo.hook_line_id, promo.cliffhanger_line_id} if mark_lines else set()
    for shot in ep.shots[first:last + 1]:
        header = f"[{shot.id}] {format_timestamp(shot.effective_start)}-{format_timestamp(shot.effective_end)} | CAM: {shot.camera_text}"
        if shot.id in marked:
            header = f">> {header} <<"
        rows.append(header)
        for line in sorted(shot.lines, key=lambda item: (item.start, item.id)):
            label = line.speaker or line.type.upper()
            if line.off_screen:
                label += " (O.S.)"
            if line.emotion:
                label += f" ({line.emotion})"
            text = f'"{line.text}"' if line.type == "dialogue" else line.text
            row = f"[{line.id}] {label} {format_timestamp(line.start)}: {text}"
            if line.id in marked:
                row = f">> {row} <<"
            rows.append(row)
    return "\n".join(rows)


def render_judge_clip(ep: Episode, promo: ValidatedPromo) -> str:
    """Use lines as the primary script; keep only essential camera-only shots."""
    positions = {shot.id: i for i, shot in enumerate(ep.shots)}
    first, last = positions[promo.start_shot_id], positions[promo.end_shot_id]
    rows = [f"CLIP {promo.id} | duration {promo.duration_s:.2f} s"]
    shots = ep.shots[first:last + 1]
    lines = sorted((line for shot in shots for line in shot.lines), key=lambda line: (line.start, line.end, line.id))
    start = shots[0].effective_start
    opening = [line for line in lines if 0 <= line.start - start <= 3.0]

    def summary_line(line):
        label = line.speaker or line.type.upper()
        return f"[{line.id}] {label}: {line.text}"

    opening_text = " | ".join(summary_line(line) for line in opening) if opening else f"CAM: {shots[0].camera_text}"
    last_line = max(lines, key=lambda line: (line.start, line.end, line.id)) if lines else None
    final_text = (summary_line(last_line) + " | ") if last_line else ""
    final_text += f"CAM: {shots[-1].camera_text}"
    rows.extend([f"DURATION: {promo.duration_s:.2f} s", f"FIRST 3 SECONDS: {opening_text}", f"FINAL BEAT: {final_text}", "SCRIPT:"])
    for index in range(first, last + 1):
        shot = ep.shots[index]
        header = f"[{shot.id}] {format_timestamp(shot.effective_start)}-{format_timestamp(shot.effective_end)}"
        if not shot.lines:
            camera = " ".join(shot.camera_text.split())
            if len(camera) > 60:
                camera = camera[:57].rstrip() + "..."
            header += f" | CAM: {camera}"
        rows.append(header)
        for line in sorted(shot.lines, key=lambda item: (item.start, item.id)):
            if line.type == "sound" and index != last:
                continue
            label = line.speaker or line.type.upper()
            if line.off_screen:
                label += " (O.S.)"
            if line.emotion:
                label += f" ({line.emotion})"
            text = f'"{line.text}"' if line.type == "dialogue" else line.text
            rows.append(f"[{line.id}] {label} {format_timestamp(line.start)}: {text}")
    return "\n".join(rows)
