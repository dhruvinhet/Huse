"""Constraint-based layout contracts for semantic visual documents."""

from enum import Enum
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from app.models.base import BaseModel, NonEmptyString


class ConstraintType(str, Enum):
    """Supported relationships understood by the layout engine."""

    ALIGN = "align"
    DISTRIBUTE = "distribute"
    CONTAIN = "contain"
    ANCHOR = "anchor"
    CONNECT = "connect"
    AVOID_OVERLAP = "avoid_overlap"
    MIN_GAP = "min_gap"
    SAME_SIZE = "same_size"
    ASPECT_RATIO = "aspect_ratio"
    ORDER = "order"
    NEAR = "near"
    FAR_FROM = "far_from"


class ConstraintStrength(str, Enum):
    """Priority used when resolving competing layout constraints."""

    REQUIRED = "required"
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


class LayoutConstraint(BaseModel):
    """Express a geometric relationship without pixel coordinates."""

    constraint_id: NonEmptyString
    type: ConstraintType
    subject_ids: list[NonEmptyString] = Field(min_length=1)
    reference_id: str | None = None
    strength: ConstraintStrength = ConstraintStrength.STRONG
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class Viewport(BaseModel):
    """Describe the available canvas and safe margins."""

    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    margin: int = Field(default=64, ge=0)

    @model_validator(mode="after")
    def validate_margin(self) -> Self:
        """Ensure margins leave positive drawable space."""

        if self.margin * 2 >= min(self.width, self.height):
            raise ValueError("viewport margin leaves no drawable area")
        return self


class LayoutBox(BaseModel):
    """Represent renderer-ready geometry after solving constraints."""

    x: float
    y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    rotation: float = 0.0


class LaidOutNode(BaseModel):
    """Represent one hierarchical node after layout."""

    object_id: NonEmptyString
    kind: NonEmptyString
    box: LayoutBox
    z_index: int = 0
    children: list["LaidOutNode"] = Field(default_factory=list)


class LayoutPlan(BaseModel):
    """Represent deterministic geometry for each visual state."""

    schema_version: Literal["2.0"] = "2.0"
    viewport: Viewport
    state_roots: dict[NonEmptyString, LaidOutNode]
    diagnostics: list[NonEmptyString] = Field(default_factory=list)
