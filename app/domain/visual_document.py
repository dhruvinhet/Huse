"""Persistent visual-document and immutable state contracts."""

from enum import Enum
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from app.domain.storyboard import VisualObjectSpec
from app.models.base import BaseModel, NonEmptyString


class ObjectLifecycle(str, Enum):
    """Observable lifecycle states for a persistent visual object."""

    VISIBLE = "visible"
    EMPHASIZED = "emphasized"
    DIMMED = "dimmed"
    HIDDEN = "hidden"
    REMOVED = "removed"


class ObjectState(BaseModel):
    """Represent mutable presentation state for one semantic object."""

    object_id: NonEmptyString
    kind: NonEmptyString
    lifecycle: ObjectLifecycle = ObjectLifecycle.VISIBLE
    content: dict[str, JsonValue] = Field(default_factory=dict)
    style_token: NonEmptyString = "concept.primary"
    parent_id: str | None = None
    child_ids: list[NonEmptyString] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SemanticStateTransition(BaseModel):
    """Preserve one explicit before/action/after semantic transition triple."""

    action_id: NonEmptyString
    action: NonEmptyString
    operator: NonEmptyString
    operand_ids: list[NonEmptyString] = Field(min_length=1)
    previous_object_states: dict[NonEmptyString, ObjectState]
    next_object_states: dict[NonEmptyString, ObjectState]


class VisualState(BaseModel):
    """Represent an immutable checkpoint after applying one visual beat."""

    state_id: NonEmptyString
    beat_id: NonEmptyString
    parent_state_id: str | None = None
    object_states: dict[NonEmptyString, ObjectState]
    transitions: list[SemanticStateTransition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_hierarchy(self) -> Self:
        """Require object-map keys and hierarchy references to agree."""

        known_ids = set(self.object_states)
        for key, state in self.object_states.items():
            if key != state.object_id:
                raise ValueError("object-state keys must match object IDs")
            if state.parent_id is not None and state.parent_id not in known_ids:
                raise ValueError("object parents must exist in the state")
            if not set(state.child_ids).issubset(known_ids):
                raise ValueError("object children must exist in the state")
        return self


class VisualDocument(BaseModel):
    """Represent initial semantic objects and materialized state history."""

    schema_version: Literal["2.0"] = "2.0"
    document_id: NonEmptyString
    initial_objects: list[VisualObjectSpec] = Field(default_factory=list)
    states: list[VisualState] = Field(min_length=1)
