"""Structural fingerprints and controlled novelty contracts."""

from typing import Literal

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString


class StructuralFingerprint(BaseModel):
    """Describe output structure without depending on lesson wording."""

    schema_version: Literal["1.0"] = "1.0"
    digest: NonEmptyString
    operator_sequence: list[NonEmptyString] = Field(default_factory=list)
    template_families: list[NonEmptyString] = Field(default_factory=list)
    layout_topology: list[NonEmptyString] = Field(default_factory=list)
    camera_operations: list[NonEmptyString] = Field(default_factory=list)
    action_sequence: list[NonEmptyString] = Field(default_factory=list)
    style_tokens: list[NonEmptyString] = Field(default_factory=list)


class NoveltyAssessment(BaseModel):
    """Report similarity without turning novelty into a correctness gate."""

    fingerprint: StructuralFingerprint
    maximum_recent_similarity: float = Field(default=0.0, ge=0, le=1)
    closest_recent_digest: str | None = None
    threshold: float = Field(default=0.82, ge=0, le=1)
    variant_requested: bool = False
    status: Literal["novel", "similar", "no_history"] = "no_history"
