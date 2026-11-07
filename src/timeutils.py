"""Convert screenplay timestamps to and from seconds."""

import math
import re


def parse_timestamp(value: str) -> float:
    """Accept MM:SS[.cc] or HH:MM:SS[.cc]."""
    if not re.fullmatch(r"\d+:\d{2}(?::\d{2})?(?:\.\d{2})?", value.strip()):
        raise ValueError(f"Invalid timestamp: {value!r}")
    parts = value.strip().split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    if seconds >= 60 or (len(parts) == 3 and minutes >= 60):
        raise ValueError(f"Invalid timestamp: {value!r}")
    hours = int(parts[0]) if len(parts) == 3 else 0
    result = hours * 3600 + minutes * 60 + seconds
    if not math.isfinite(result):
        raise ValueError(f"Invalid timestamp: {value!r}")
    return result


def format_timestamp(seconds: float) -> str:
    """Format nonnegative seconds as MM:SS.cc, rounding to centiseconds."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Seconds must be finite and nonnegative")
    centiseconds = round(seconds * 100)
    minutes, remainder = divmod(centiseconds, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"
