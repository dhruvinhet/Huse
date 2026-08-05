"""Visual-strategy and template-selection contracts."""

from typing import Literal

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString
from app.domain.pedagogy import PedagogyMode
from app.domain.lesson import ConceptRelation
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


class TemplateCapabilities(BaseModel):
    """Machine-readable obligations a reviewed template can satisfy."""

    relation_types: list[ConceptRelation] = Field(
        default_factory=lambda: list(ConceptRelation)
    )
    semantic_actions: list[NonEmptyString] = Field(default_factory=lambda: [
        "transfer", "route", "split", "merge", "group", "compare",
        "consume", "produce", "transform", "substitute", "accumulate",
        "trace",
    ])
    minimum_operands: int = Field(default=1, ge=1, le=64)
    maximum_operands: int = Field(default=12, ge=1, le=64)
    pedagogy_roles: list[NonEmptyString] = Field(default_factory=lambda: [
        "introduce", "demonstrate", "compare", "transform", "connect",
        "emphasize", "summarize",
    ])
    layout_constraints: list[NonEmptyString] = Field(default_factory=list)
    required_parameters: list[NonEmptyString] = Field(default_factory=list)


class ParameterProvenance(BaseModel):
    """Explain where a template parameter originated."""

    source: Literal["extracted", "derived", "default"]
    source_field: NonEmptyString
    confidence: float = Field(default=1.0, ge=0, le=1)


class TemplateMatch(BaseModel):
    """Represent a ranked semantic-template match."""

    template_id: NonEmptyString
    concept_ids: list[NonEmptyString] = Field(min_length=1)
    parameters: dict[str, object] = Field(default_factory=dict)
    prototype: VisualObjectSpec | None = None
    score: float = Field(ge=0, le=1)
    reason: NonEmptyString
    capabilities: TemplateCapabilities = Field(default_factory=TemplateCapabilities)
    capability_evidence: list[NonEmptyString] = Field(default_factory=list)
    match_confidence: float = Field(default=0.5, ge=0, le=1)
    parameter_provenance: dict[NonEmptyString, ParameterProvenance] = Field(
        default_factory=dict
    )
    default_usage: list[NonEmptyString] = Field(default_factory=list)
    novelty_penalty: float = Field(default=0.0, ge=0, le=1)
    novelty_evidence: list[NonEmptyString] = Field(default_factory=list)
    layout_variant: Literal["canonical", "horizontal", "vertical", "grid"] = (
        "canonical"
    )


class CompiledTemplateProgram(BaseModel):
    """Record an authoritative reviewed-template visual program."""

    template_id: NonEmptyString
    template_ids: list[NonEmptyString] = Field(default_factory=list)
    shot_template_ids: dict[NonEmptyString, NonEmptyString] = Field(
        default_factory=dict
    )
    parameters: dict[str, object] = Field(default_factory=dict)
    parameter_provenance: dict[NonEmptyString, ParameterProvenance] = Field(
        default_factory=dict
    )
    default_usage: list[NonEmptyString] = Field(default_factory=list)
    match_confidence: float = Field(default=0.5, ge=0, le=1)
    capability_evidence: list[NonEmptyString] = Field(default_factory=list)
    pedagogy_mode: PedagogyMode
    storyboard: Storyboard
    layout_variants: dict[NonEmptyString, NonEmptyString] = Field(
        default_factory=dict
    )
