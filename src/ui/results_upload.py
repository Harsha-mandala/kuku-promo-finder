"""Extract the supported view of a results archive into a session workspace."""

import io
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

from .data import load_bundle

MODE = "Load results bundle (.zip)"
MAX_BYTES = 50_000_000
TAG = r"[A-Za-z0-9_-]+"
ROOT_FILES = {"story_bible.json", "story_bible.md", "llm_usage.csv"}
RUN_FILES = {
    "final": {"promos_ranked.json", "trailers_ranked.json", "editor_inputs.json", "editor_order.json",
              "exclusions.json", "safety_review.json", "report.md"},
    "judged": {"scores.json"},
    "trailers": {"manifest.json", "validated.json"},
    "eval": {"metrics.json", "eval_report.md"},
}
ROW_LISTS = {"promos_ranked.json", "trailers_ranked.json", "editor_order.json",
             "safety_review.json", "validated.json", "scores.json"}


def member_path(info):
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or ".." in path.parts or
            any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part.endswith(".") for part in path.parts)):
        raise ValueError("Results ZIP contains an unsafe path.")
    kind = stat.S_IFMT(info.external_attr >> 16)
    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError("Results ZIP cannot contain links or special files.")
    return path


def supported_file(path, tag):
    if len(path.parts) == 1:
        return path.name in ROOT_FILES or path.name == f"pipeline_{tag}_events.jsonl"
    if len(path.parts) != 2:
        return False
    folder, name = path.parts
    if folder in {"parsed", f"validated_{tag}", f"candidates_{tag}"}:
        return bool(re.fullmatch(r"episode_\d+\.json", name))
    return any(folder == f"{group}_{tag}" and name in names for group, names in RUN_FILES.items())


def result_bytes(archive, info, path):
    data = archive.read(info)
    if path.suffix != ".json":
        return data
    rows = path.name in ROW_LISTS or path.parts[0].startswith(("candidates_", "validated_"))
    try:
        document = json.loads(data)
        valid = (isinstance(document, list) and all(isinstance(row, dict) for row in document)
                 if rows else isinstance(document, dict))
        if not valid:
            raise ValueError("wrong JSON shape")
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError(f"Results ZIP contains malformed JSON in {path.as_posix()}.") from error
    return data


def load_results_zip(data, workspace):
    """Validate the archive before extracting only files used by the UI."""
    if len(data) > MAX_BYTES:
        raise ValueError("Results ZIP exceeds the 50 MB limit.")
    root = Path(workspace)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > 2500 or sum(info.file_size for info in infos) > MAX_BYTES:
                raise ValueError("Results ZIP exceeds 50 MB unpacked or contains too many entries.")
            members, seen = {}, set()
            for info in infos:
                path = member_path(info)
                if info.is_dir():
                    continue
                if path.parts[0] == "raw_outputs":
                    path = PurePosixPath(*path.parts[1:])
                key = path.as_posix().casefold()
                if key in seen:
                    raise ValueError("Results ZIP contains duplicate filenames.")
                seen.add(key)
                members[path] = info
            tags = {path.parts[0][6:] for path in members if len(path.parts) == 2
                    and re.fullmatch(rf"final_{TAG}", path.parts[0]) and path.name == "promos_ranked.json"}
            if len(tags) != 1:
                raise ValueError("Results ZIP must contain one final_<tag>/promos_ranked.json.")
            tag = tags.pop()
            required = PurePosixPath(f"final_{tag}/report.md")
            if required not in members:
                raise ValueError(f"Results ZIP is missing {required.as_posix()}.")
            if not any(path.parts[0] == "parsed" and supported_file(path, tag) for path in members if path.parts):
                raise ValueError("Results ZIP is missing parsed/episode_*.json. Use the refreshed submission ZIP.")
            for path, info in members.items():
                if supported_file(path, tag):
                    target = root.joinpath(*path.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(result_bytes(archive, info, path))
        bundle = load_bundle(root / "parsed", root, tag)
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as error:
        raise ValueError("Could not read the results ZIP. Upload an unencrypted ZIP archive.") from error
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        if type(error) is ValueError:
            raise
        raise ValueError("Results ZIP contains malformed result data.") from error
    return bundle
