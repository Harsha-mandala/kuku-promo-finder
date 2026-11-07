"""Structured screenplay records."""

from typing import Literal

from pydantic import BaseModel, Field, RootModel, computed_field


class TrailerSegment(BaseModel):
    start_shot_id: str
    end_shot_id: str


class Trailer(BaseModel):
    segments: list[TrailerSegment] = Field(min_length=2, max_length=3)
    hook_line_id: str
    cliffhanger_line_id: str
    story: str
    why_it_works: str
    spoiler_risk: Literal["low", "medium", "high"]
    self_score: int = Field(ge=1, le=5)


class TrailerBatch(RootModel[list[Trailer]]):
    pass


class Line(BaseModel):
    """A timed screenplay block, including camera blocks."""

    id: str
    episode: int
    scene: int
    type: Literal["camera", "action", "dialogue", "sound"]
    speaker: str | None = None
    emotion: str | None = None
    off_screen: bool = False
    start: float
    end: float
    text: str


class Shot(BaseModel):
    """A camera block and the lines assigned to it."""

    id: str
    index: int
    episode: int
    scene: int
    start: float
    end: float
    camera_text: str
    lines: list[Line] = Field(default_factory=list)
    clean_start: bool = True
    clean_end: bool = True

    @computed_field
    @property
    def effective_start(self) -> float:
        """Include content attached before the camera starts."""
        return min([self.start] + [line.start for line in self.lines])

    @computed_field
    @property
    def effective_end(self) -> float:
        """Include content attached after the camera ends."""
        return max([self.end] + [line.end for line in self.lines])


class Scene(BaseModel):
    """Scene metadata using episode-relative seconds."""

    number: int
    heading: str
    start: float
    end: float


class Episode(BaseModel):
    """A parsed episode and its recoverable input warnings."""

    number: int
    source_file: str
    logline: str = ""
    continuity_notes: str = ""
    duration: float = 0.0
    scenes: list[Scene] = Field(default_factory=list)
    lines: list[Line] = Field(default_factory=list)
    shots: list[Shot] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Character(BaseModel):
    """A grounded character identity and its exact script labels."""

    name: str
    aliases: list[str]
    role: str
    relationships: str
    confidence: Literal["high", "medium", "low"]
    evidence: str = Field(min_length=1)


class StoryArc(BaseModel):
    episodes: str
    summary: str


class StoryBible(BaseModel):
    title_guess: str
    premise: str
    characters: list[Character]
    minor_characters: list[str] = Field(default_factory=list)
    main_conflicts: list[str]
    arcs: list[StoryArc]


class Candidate(BaseModel):
    """An LLM-proposed continuous shot range with source IDs."""

    start_shot_id: str = Field(description="Canonical ID E001_S015 for episode 1 [S015]; use this episode's number.")
    end_shot_id: str = Field(description="Canonical episode-specific ID, e.g. E001_S030 for episode 1 [S030].")
    hook_line_id: str = Field(min_length=1, description="Source line ID, e.g. L027 or E001_L027. Only a shot with no attached lines may use its shot ID as camera fallback.")
    hook_type: Literal[
        "slap_or_physical", "shocking_reveal", "cruel_or_shocking_line",
        "confrontation", "public_humiliation", "threat", "romance_tension", "other",
    ]
    cliffhanger_line_id: str = Field(min_length=1, description="Source line ID inside the selected shots; use the shot ID itself only when that shot has no attached lines.")
    cliffhanger_type: Literal[
        "unanswered_question", "threat", "pending_reveal", "decision_moment",
        "shocked_reaction", "danger", "emotional_turn", "other",
    ]
    context_needed: str
    why_it_works: str
    self_score: int = Field(ge=1, le=5)


class CandidateList(BaseModel):
    episode: int
    candidates: list[Candidate]


class CandidateBatch(RootModel[list[CandidateList]]):
    """One separate episode result per item in a batch response."""


class ScoredCandidate(Candidate):
    """A proposal with source-derived timing and boundary measurements."""

    candidate_id: str
    hook_text: str
    cliffhanger_text: str
    span_start: float
    span_end: float
    duration_s: float
    hook_offset_s: float
    cliffhanger_end_gap_s: float
    start_clean: bool
    end_clean: bool
    n_shots: int
