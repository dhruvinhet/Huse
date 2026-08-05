"""Persistent visual-document operations."""

from enum import Enum
from typing import Annotated, Literal, Self, TypeAlias, Union

from pydantic import Field, JsonValue, model_validator

from app.models.base import BaseModel, NonEmptyString
from app.domain.lesson import ConceptRelation
from app.domain.visual_intent import RendererOperator


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


class ActionCondition(BaseModel):
    """One deterministic semantic-state precondition or postcondition."""

    object_id: NonEmptyString
    field: NonEmptyString
    expected: JsonValue


class SemanticActionBase(BaseModel):
    """Shared timing, ownership, and validation fields for semantic actions."""

    action_id: NonEmptyString
    operator: RendererOperator
    operand_ids: list[NonEmptyString] = Field(min_length=1, max_length=16)
    relation: ConceptRelation | None = None
    direction: Literal[
        "forward", "reverse", "bidirectional", "inward", "outward"
    ] = "forward"
    preconditions: list[ActionCondition] = Field(default_factory=list, max_length=12)
    postconditions: list[ActionCondition] = Field(default_factory=list, max_length=12)
    duration_hint: float = Field(default=0.8, gt=0, le=12)
    easing: Literal["linear", "ease_in", "ease_out", "ease_in_out"] = (
        "ease_in_out"
    )
    reversible: bool = True

    @model_validator(mode="after")
    def unique_operands(self) -> Self:
        if len(self.operand_ids) != len(set(self.operand_ids)):
            raise ValueError("semantic action operands must be unique")
        return self


class TransferAction(SemanticActionBase):
    action: Literal["transfer"] = "transfer"
    source_id: NonEmptyString
    target_id: NonEmptyString
    payload_ids: list[NonEmptyString] = Field(min_length=1, max_length=12)


class RouteAction(SemanticActionBase):
    action: Literal["route"] = "route"
    source_id: NonEmptyString
    target_id: NonEmptyString
    path_ids: list[NonEmptyString] = Field(default_factory=list, max_length=12)


class SplitAction(SemanticActionBase):
    action: Literal["split"] = "split"
    source_id: NonEmptyString
    output_ids: list[NonEmptyString] = Field(min_length=2, max_length=8)


class MergeAction(SemanticActionBase):
    action: Literal["merge"] = "merge"
    input_ids: list[NonEmptyString] = Field(min_length=2, max_length=8)
    target_id: NonEmptyString


class GroupAction(SemanticActionBase):
    action: Literal["group"] = "group"
    member_ids: list[NonEmptyString] = Field(min_length=1, max_length=12)
    group_id: NonEmptyString


class CompareAction(SemanticActionBase):
    action: Literal["compare"] = "compare"
    left_id: NonEmptyString
    right_id: NonEmptyString


class ConsumeAction(SemanticActionBase):
    action: Literal["consume"] = "consume"
    consumer_id: NonEmptyString
    item_ids: list[NonEmptyString] = Field(min_length=1, max_length=12)


class ProduceAction(SemanticActionBase):
    action: Literal["produce"] = "produce"
    producer_id: NonEmptyString
    item_ids: list[NonEmptyString] = Field(min_length=1, max_length=12)


class TransformAction(SemanticActionBase):
    action: Literal["transform"] = "transform"
    source_id: NonEmptyString
    target_id: NonEmptyString


class SubstituteAction(SemanticActionBase):
    action: Literal["substitute"] = "substitute"
    source_id: NonEmptyString
    replacement_id: NonEmptyString


class AccumulateAction(SemanticActionBase):
    action: Literal["accumulate"] = "accumulate"
    accumulator_id: NonEmptyString
    item_ids: list[NonEmptyString] = Field(min_length=1, max_length=12)


class TraceAction(SemanticActionBase):
    action: Literal["trace"] = "trace"
    path_ids: list[NonEmptyString] = Field(min_length=2, max_length=16)


SemanticAction: TypeAlias = Annotated[
    Union[
        TransferAction,
        RouteAction,
        SplitAction,
        MergeAction,
        GroupAction,
        CompareAction,
        ConsumeAction,
        ProduceAction,
        TransformAction,
        SubstituteAction,
        AccumulateAction,
        TraceAction,
    ],
    Field(discriminator="action"),
]
