"""Validated teaching modes and their shot-level obligations."""

from enum import Enum
from typing import Literal

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString


class PedagogyMode(str, Enum):
    """Supported topic-specific explanatory rhetorics."""

    MECHANISM_FIRST = "mechanism_first"
    CONCEPT_OVERVIEW = "concept_overview"
    CONCEPT_SET = "concept_set"
    WORKED_EXAMPLE = "worked_example"
    MISCONCEPTION_CORRECTION = "misconception_correction"
    ANALOGY = "analogy"
    PROOF_DERIVATION = "proof_derivation"
    CHRONOLOGICAL = "chronological"
    COMPARISON = "comparison"
    SIMULATION = "simulation"
    SPATIAL_ANATOMY = "spatial_anatomy"
    CODE_EXECUTION = "code_execution"


class PedagogyShot(BaseModel):
    """Define one required visual and rhetorical move."""

    shot_id: NonEmptyString
    purpose: Literal[
        "introduce",
        "demonstrate",
        "compare",
        "transform",
        "connect",
        "emphasize",
        "summarize",
    ]
    visual_obligation: NonEmptyString
    narration_obligation: NonEmptyString
    camera_operation: Literal["fit", "focus", "hold"] = "hold"


class PedagogyPlan(BaseModel):
    """Represent the authoritative teaching grammar for one lesson."""

    mode: PedagogyMode
    rationale: NonEmptyString
    shots: list[PedagogyShot] = Field(min_length=3, max_length=6)

    @property
    def narration_obligations(self) -> list[str]:
        """Return ordered narration requirements for prompt and compiler use."""

        return [shot.narration_obligation for shot in self.shots]
