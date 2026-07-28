"""Persistent visual-document operations."""

from enum import Enum

from pydantic import Field, JsonValue

from app.models.base import BaseModel, NonEmptyString


class OperationType(str, Enum):
    """Supported state transitions for persistent objects."""

    CREATE = "create"
    UPDATE = "update"
    MOVE = "move"
    RESIZE = "resize"
    HIGHLIGHT = "highlight"
    DIM = "dim"
    MORPH = "morph"
    DUPLICATE = "duplicate"
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    ERASE = "erase"
    SHOW = "show"
    HIDE = "hide"
    GROUP = "group"
    UNGROUP = "ungroup"


class VisualOperation(BaseModel):
    """Describe one semantic mutation of the visual document."""

    operation_id: NonEmptyString
    operation: OperationType
    target_ids: list[NonEmptyString] = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    reversible: bool = True
