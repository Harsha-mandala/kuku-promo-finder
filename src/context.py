"""Build series context from metadata and sampled screenplay evidence."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .llm import LLMClient, LLMOutputError, LLMRequestError, LLMBudgetError, get_llm, step_model_config, estimate_json_tokens
from .models import Episode, StoryBible, StoryArc, Character

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_hints(path: str | Path | None = None) -> dict[str, str]:
    """Hints are opt-in; config.yaml never implicitly supplies identities."""
    if path is None:
        return {}
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    hints = value.get("alias_hints", value)
    if not isinstance(hints, dict) or not all(isinstance(key, str) and isinstance(name, str) for key, name in hints.items()):
        raise ValueError("Hints file must contain an alias_hints mapping of strings")
    return hints


def speaker_evidence(episodes: list[Episode]) -> list[dict]:
    """Sample first and last available utterances with full neighbouring context."""
    occurrences = defaultdict(list)
    for episode in sorted(episodes, key=lambda item: item.number):
        for index, shot in enumerate(episode.shots):
            for line in shot.lines:
                if line.type == "dialogue" and line.speaker:
                    occurrences[line.speaker].append((episode, index, line))
    records = []
    for label, values in sorted(occurrences.items()):
        selected = [values[0]]
        if len(values) > 1:
            selected.append(values[-1])
        samples = []
        for episode, index, line in selected:
            neighbours = []
            for shot in episode.shots[max(0, index - 1):index + 2]:
                neighbours.append({
                    "shot_id": shot.id, "camera": shot.camera_text,
                    "actions": [{"id": item.id, "text": item.text} for item in shot.lines if item.type == "action"],
                })
            samples.append({"line_id": line.id, "text": line.text, "episode": episode.number, "shots": neighbours})
        records.append({"speaker": label, "episodes": sorted({value[0].number for value in values}), "samples": samples})
    return records


def load_episodes(parsed_dir: str | Path) -> list[Episode]:
    """Load all parsed episodes once in numeric order."""
    episodes = [
        Episode.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(Path(parsed_dir).glob("episode_*.json"))
    ]
    episodes.sort(key=lambda episode: episode.number)
    if not episodes:
        raise ValueError(f"No parsed episodes found in {parsed_dir}; run src.parser first")
    if len({episode.number for episode in episodes}) != len(episodes):
        raise ValueError("Parsed data has duplicate episode numbers; resolve them before generating context")
    return episodes


def episode_metadata(episodes: list[Episode], alias_hints: dict[str, str] | None = None) -> str:
    """Include metadata plus sampled dialogue and neighbouring shot context."""
    records = []
    for episode in sorted(episodes, key=lambda item: item.number):
        speakers = sorted({line.speaker for line in episode.lines if line.speaker})
        records.append({
            "episode": episode.number, "logline": episode.logline,
            "continuity_notes": episode.continuity_notes, "speakers": speakers,
        })
    labels = sorted({speaker for record in records for speaker in record["speakers"]})
    return json.dumps({
        "identity_hints": alias_hints or {}, "all_speaker_labels": labels, "episodes": records,
        "speaker_evidence": speaker_evidence(episodes),
    }, ensure_ascii=False, separators=(",", ":"))


def bible_source_fingerprint(episodes, alias_hints=None):
    """Bind a saved bible to this show's metadata and source evidence."""
    return hashlib.sha256(episode_metadata(episodes, alias_hints).encode("utf-8")).hexdigest()


def validate_bible(
    bible: StoryBible, episodes: list[Episode], alias_hints: dict[str, str] | None = None,
) -> list[str]:
    labels = {line.speaker for episode in episodes for line in episode.lines if line.speaker}
    aliases = {alias for character in bible.characters for alias in character.aliases}
    missing = sorted(labels - aliases - set(bible.minor_characters))
    errors = []
    if missing:
        errors.append("[missing_aliases] Missing exact speaker labels in character aliases: " + ", ".join(missing))
    by_label = {record["speaker"]: record for record in speaker_evidence(episodes)}
    for label in sorted(set(bible.minor_characters) & labels):
        named = not label.isupper() and any(re.search(r"\b" + re.escape(label) + r"\b", episode.logline) for episode in episodes)
        if label not in aliases and (len(by_label[label]["episodes"]) >= 2 or named):
            errors.append(f"[main_character_required] {label!r} recurs or is named in a logline; include it in main character aliases")
    corpus = "\n".join(
        [episode.logline + " " + episode.continuity_notes for episode in episodes]
        + [line.text for episode in episodes for line in episode.lines]
        + list(labels)
    ).casefold()
    for character in bible.characters:
        if not re.search(r"\b" + re.escape(character.name.casefold()) + r"\b", corpus):
            errors.append(f"[unsupported_name] Full name/title {character.name!r} is not a literal source name; use an observed shorter name or exact speaker label")
        for word in re.findall(r"[\w]+", character.name.casefold()):
            if word not in {"unknown", "unidentified"} and not re.search(r"\b" + re.escape(word) + r"\b", corpus):
                errors.append(f"[unsupported_name] {character.name!r}: name/title component {word!r} is not in the source; use an exact label if uncertain")
        evidence_ids = re.findall(r"E\d+_[LS]\d+", character.evidence)
        valid_ids = {line.id for episode in episodes for line in episode.lines} | {shot.id for episode in episodes for shot in episode.shots}
        if evidence_ids and any(identifier not in valid_ids for identifier in evidence_ids):
            errors.append(f"[unknown_evidence] {character.name}: cite only existing source IDs")
    for alias, expected in (alias_hints or {}).items():
        if alias not in labels:
            continue
        owners = [character.name for character in bible.characters if alias in character.aliases]
        if not owners or any(not re.search(r"\b" + re.escape(expected) + r"\b", name, re.IGNORECASE) for name in owners):
            errors.append(f"[alias_identity] Author-provided identity: {alias!r} is {expected}, not {owners}. Correct the alias mapping.")
    return errors


def render_bible_markdown(bible: StoryBible) -> str:
    rows = [f"# {bible.title_guess} (title guess)", "", bible.premise, "", "## Characters", ""]
    for character in bible.characters:
        rows.extend([
            f"### {character.name}", f"Aliases: {', '.join(character.aliases)}",
            f"Role: {character.role}", f"Relationships: {character.relationships}",
            f"Confidence: {character.confidence}", f"Evidence: {character.evidence}", "",
        ])
    rows.extend(["## Minor characters", "", ", ".join(bible.minor_characters), ""])
    rows.extend(["## Main conflicts", ""])
    rows.extend(f"- {conflict}" for conflict in bible.main_conflicts)
    rows.extend(["", "## Arcs", ""])
    rows.extend(f"- Episodes {arc.episodes}: {arc.summary}" for arc in bible.arcs)
    return "\n".join(rows) + "\n"


def _short(text: str, length: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length - 3].rstrip() + "..."


def render_bible_compact(bible: StoryBible) -> str:
    """Keep core context under 1,800 characters (roughly under 600 tokens)."""
    rows = [f"TITLE GUESS: {_short(bible.title_guess, 80)}", f"PREMISE: {_short(bible.premise, 420)}"]
    conflicts = "Conflicts: " + _short("; ".join(bible.main_conflicts), 260)
    for character in bible.characters[:6]:
        aliases = ", ".join(character.aliases[:4])
        row = (
            f"{character.name} [{aliases}; {character.confidence} confidence]: {_short(character.role, 80)}; "
            f"{_short(character.relationships, 130)}"
        )
        if len("\n".join(rows + [row, conflicts])) > 1800:
            break
        rows.append(row)
    rows.append(conflicts)
    return "\n".join(rows)


class StoryOverview(BaseModel):
    title_guess: str
    premise: str
    main_conflicts: list[str]
    arcs: list[StoryArc]


class SpeakerIdentity(BaseModel):
    label: str
    name: str = Field(max_length=80)
    confidence: Literal["high", "medium", "low"]
    evidence: str = Field(max_length=140)


class IdentityBatch(BaseModel):
    speakers: list[SpeakerIdentity]


def _build_batched_bible(payload: dict, system: str, client: LLMClient, output: Path) -> StoryBible:
    """Fit the evidence into provider token limits, retaining all source context."""
    metadata = [{key: value for key, value in item.items() if key != "speakers"} for item in payload["episodes"]]
    overview = client.generate_json(
        "Summarize only these episode loglines/notes into a factual overview with episode-range arcs. Never invent names or titles.",
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), StoryOverview, "story_overview",
    )
    identities = []
    def checkpoint(done, total):
        progress = {"overview": overview.model_dump(), "identities": [item.model_dump() for item in identities],
                    "completed_evidence_batches": done, "total_evidence_batches": total,
                    "episode_numbers": [item["episode"] for item in payload["episodes"]]}
        (output / "story_bible_progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    batches, batch = [], []
    # Reserve room for system/schema instructions and a validation repair.
    budget = min(14000, max(2000, (client.limiter.tokens_per_minute or 8000) * 4 - 12000))
    for record in payload["speaker_evidence"]:
        if batch and len(json.dumps(batch + [record], ensure_ascii=False)) > budget:
            batches.append(batch)
            batch = []
        batch.append(record)
    if batch:
        batches.append(batch)
    checkpoint(0, len(batches))
    for index, batch in enumerate(batches, start=1):
        labels = {item["speaker"] for item in batch}
        def check(result):
            actual = [item.label for item in result.speakers]
            return [] if set(actual) == labels and len(actual) == len(labels) else ["Return exactly one identity record for every supplied speaker label."]
        result = client.generate_json(
            "Infer each speaker's identity from the two samples and full same/neighbouring shot camera/actions. "
            "Never invent a surname or title. If uncertain use the exact speaker label with low confidence. "
            "Return one record per label with high/medium/low confidence and a short supporting quote or canonical line ID.",
            json.dumps({"overview": overview.model_dump(), "identity_hints": payload["identity_hints"], "speakers": batch}, ensure_ascii=False, separators=(",", ":")),
            IdentityBatch, f"story_evidence_{index:02d}", extra_validator=check,
        )
        identities.extend(result.speakers)
        checkpoint(index, len(batches))
        print(f"Story evidence {index}/{len(batches)}: {len(batch)} speaker labels", flush=True)
    evidence_output = {"identity_hints": payload["identity_hints"], "overview": overview.model_dump(), "identities": [item.model_dump() for item in identities]}
    (output / "story_evidence.json").write_text(json.dumps(evidence_output, ensure_ascii=False, indent=2), encoding="utf-8")
    return assemble_story_bible(payload, overview, identities)


def assemble_story_bible(payload: dict, overview: StoryOverview, identities: list[SpeakerIdentity]) -> StoryBible:
    """Assemble a complete roster from saved model evidence without API calls."""
    # Every identity has already been inferred from its source samples. Assemble
    # those complete records in code rather than asking the model to repeat a
    # large roster and risk dropping labels at its completion limit.
    occurrences = {item["speaker"]: set(item["episodes"]) for item in payload["speaker_evidence"]}
    source_words = " ".join(json.dumps(payload, ensure_ascii=False).split()).casefold()
    groups = defaultdict(list)
    for identity in identities:
        name = identity.name
        if not re.search(r"\b" + re.escape(name.casefold()) + r"\b", source_words):
            # A place/house name in context does not establish a surname.
            observed = None
            titles = {"the", "lady", "lord", "sir", "duke", "duchess", "king", "queen", "father", "unknown", "unidentified"}
            for word in re.findall(r"\w+", name):
                if word.casefold() not in titles and re.search(r"\b" + re.escape(word.casefold()) + r"\b", source_words):
                    observed = word
                    break
            identity = identity.model_copy(update={"name": observed or identity.label, "confidence": identity.confidence if observed else "low"})
        groups[identity.name.casefold()].append(identity)
    characters, minor = [], []
    confidence_order = {"low": 0, "medium": 1, "high": 2}
    for members in groups.values():
        name = members[0].name
        aliases = sorted({item.label for item in members})
        episode_numbers = set().union(*(occurrences[label] for label in aliases))
        named_loglines = [item["logline"] for item in payload["episodes"] if not name.isupper() and re.search(r"\b" + re.escape(name) + r"\b", item["logline"])]
        if len(episode_numbers) < 2 and not named_loglines:
            minor.extend(aliases)
            continue
        notes = [item["continuity_notes"] for item in payload["episodes"] if re.search(r"\b" + re.escape(name) + r"\b", item["continuity_notes"], re.IGNORECASE)]
        characters.append(Character(
            name=name, aliases=aliases,
            role="Source logline: " + named_loglines[0] if named_loglines else f"Recurring speaker in {len(episode_numbers)} episodes.",
            relationships="Source continuity: " + notes[0] if notes else "Not established in the supplied continuity notes.",
            confidence=min((item.confidence for item in members), key=confidence_order.get),
            evidence="; ".join(dict.fromkeys(item.evidence for item in members)),
        ))
    characters.sort(key=lambda item: (-sum(len(occurrences[label]) for label in item.aliases), item.name))
    return StoryBible(**overview.model_dump(), characters=characters, minor_characters=sorted(minor))


def build_story_bible(
    episodes: list[Episode], client: LLMClient, output_dir: str | Path,
    alias_hints: dict[str, str] | None = None,
) -> StoryBible:
    """Synthesize all episode metadata through one shared JSON generation call."""
    system = (PROMPTS_DIR / "story_bible.md").read_text(encoding="utf-8")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    user = episode_metadata(episodes, alias_hints)
    payload = json.loads(user)
    limits = [value for value in (client.max_request_tokens, max(1, client.limiter.tokens_per_minute - 1500) if client.limiter.tokens_per_minute else None) if value]
    if limits and estimate_json_tokens(system, user, StoryBible) > min(limits):
        bible = _build_batched_bible(payload, system, client, output)
        errors = validate_bible(bible, episodes, alias_hints)
        if errors:
            raise ValueError("Story evidence assembly failed: " + "; ".join(errors))
    else:
        bible = client.generate_json(
            system, user, StoryBible, "story_bible",
            extra_validator=lambda result: validate_bible(result, episodes, alias_hints),
        )
    (output / "story_bible.json").write_text(bible.model_dump_json(indent=2), encoding="utf-8")
    (output / "story_bible.md").write_text(render_bible_markdown(bible), encoding="utf-8")
    (output / "story_bible_build.json").write_text(json.dumps({"source_fingerprint": bible_source_fingerprint(episodes, alias_hints), "hints_used": bool(alias_hints), "identity_hints": alias_hints or {}, "samples_per_label": {item["speaker"]: len(item["samples"]) for item in payload["speaker_evidence"]}},indent=2), encoding="utf-8")
    return bible


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=Path("config.yaml"))
    cli.add_argument("--provider")
    cli.add_argument("--model")
    cli.add_argument("--hints", type=Path, help="Optional identity hints YAML; omitted by default")
    cli.add_argument("--max-api-calls", type=int, help="Maximum real attempts, including retries and repairs")
    args = cli.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        if args.max_api_calls is not None:
            config.setdefault("llm", {})["max_api_calls_per_run"] = args.max_api_calls
        episodes = load_episodes(config.get("parsed_dir", "parsed"))
        provider, model = step_model_config(config, "bible", args.provider, args.model)
        client = get_llm(config, provider, model)
        bible = build_story_bible(
            episodes, client, config.get("output_dir", "outputs"),
            load_hints(args.hints),
        )
        print(render_bible_markdown(bible))
        print(f"cached={client.last_call['cached']} tokens_in={client.last_call['in_tokens']} "
              f"tokens_out={client.last_call['out_tokens']} latency_s={client.last_call['latency_s']:.6f}")
    except LLMBudgetError as error:
        numbers = [episode.number for episode in episodes]
        print(f"Bible completed episodes: [] | Pending episodes: {numbers}")
        print("Any completed evidence is checkpointed; raw logs/cache are retained for resuming.")
        cli.exit(1, f"{error}\n")
    except (OSError, ValueError, LLMRequestError, LLMOutputError) as error:
        cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
