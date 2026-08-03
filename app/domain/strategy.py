"""Visual-strategy and template-selection contracts."""

from typing import Literal

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString
from app.domain.pedagogy import PedagogyMode
from app.domain.storyboard import Storyboard, VisualObjectSpec


class VisualStrategy(BaseModel):
    """Describe an evidence-backed teaching visualization strategy."""

    strategy_id: NonEmptyString
    concept_ids: list[NonEmptyString] = Field(min_length=1)
    preferred_template: str | None = None
    teaching_strategy: NonEmptyString
    animation_hints: list[NonEmptyString] = Field(default_factory=list)
    visual_complexity: int = Field(default=3, ge=1, le=10)
    audience_levels: list[Literal["beginner", "intermediate", "advanced"]] = Field(
        default_factory=lambda: ["beginner", "intermediate", "advanced"]
    )
    evidence_score: float = Field(default=0.5, ge=0, le=1)


class TemplateMatch(BaseModel):
    """Represent a ranked semantic-template match."""

    template_id: NonEmptyString
    concept_ids: list[NonEmptyString] = Field(min_length=1)
    parameters: dict[str, object] = Field(default_factory=dict)
    prototype: VisualObjectSpec | None = None
    score: float = Field(ge=0, le=1)
    reason: NonEmptyString


class CompiledTemplateProgram(BaseModel):
    """Record an authoritative reviewed-template visual program."""

    template_id: NonEmptyString
    parameters: dict[str, object] = Field(default_factory=dict)
    pedagogy_mode: PedagogyMode
    storyboard: Storyboard
