"""Typed semantic-action validation, lowering, state, and motion coverage."""

from pathlib import Path

import pytest

from app.domain.layout import LayoutBox, LayoutPlan, LaidOutNode, Viewport
from app.domain.narration import AlignedAudio, PhraseTiming
from app.domain.operations import (
    ActionCondition,
    ConsumeAction,
    TransferAction,
    VisualOperation,
    OperationType,
)
from app.domain.storyboard import Storyboard, VisualBeat, VisualObjectSpec
from app.domain.visual_document import ObjectLifecycle
from app.domain.visual_intent import RendererOperator
from app.motion import SemanticAnimationPlanner
from app.rendering import SemanticFrameRenderer
from app.state import (
    SemanticActionCompiler,
    UnsupportedOperatorActionError,
    VisualStateTransitionEngine,
)


def _root() -> VisualObjectSpec:
    return VisualObjectSpec(
        object_id="workspace",
        kind="nested_group",
        semantic_role="action_workspace",
        content={"operator": "process", "layout": "horizontal"},
        accessibility_label="Action workspace",
        children=[
            VisualObjectSpec(
                object_id=object_id,
                kind="component",
                semantic_role=object_id,
                content={"label": object_id.title()},
                accessibility_label=object_id.title(),
            )
            for object_id in ("left", "right", "token", "fuel")
        ],
    )


def _create_beat() -> VisualBeat:
    root = _root()
    return VisualBeat(
        beat_id="create",
        section_id="actions",
        concept_ids=["transfer"],
        teaching_intent="Create the action workspace.",
        phrase_intent="First, create the source, target, and token.",
        estimated_duration=1,
        operations=[VisualOperation(
            operation_id="create_workspace",
            operation=OperationType.CREATE,
            target_ids=[root.object_id],
            arguments={"objects": [root.model_dump(mode="json")]},
        )],
    )


def _transfer(**updates: object) -> TransferAction:
    payload: dict[str, object] = {
        "action_id": "move_token",
        "action": "transfer",
        "operator": RendererOperator.PROCESS,
        "operand_ids": ["left", "right", "token"],
        "source_id": "left",
        "target_id": "right",
        "payload_ids": ["token"],
        "preconditions": [ActionCondition(
            object_id="token",
            field="parent_id",
            expected="workspace",
        )],
        "postconditions": [ActionCondition(
            object_id="token",
            field="parent_id",
            expected="right",
        )],
    }
    payload.update(updates)
    return TransferAction.model_validate(payload)


def _action_beat(action: object, beat_id: str = "transfer") -> VisualBeat:
    return VisualBeat(
        beat_id=beat_id,
        section_id="actions",
        concept_ids=["transfer"],
        teaching_intent="Transfer the token to the target.",
        phrase_intent="The token travels from left to right.",
        purpose="transform",
        estimated_duration=1,
        operations=[VisualOperation(
            operation_id=f"hold_{beat_id}",
            operation=OperationType.SHOW,
            target_ids=["workspace"],
        )],
        semantic_actions=[action],
    )


def test_transfer_lowers_with_preconditions_and_semantic_state_delta() -> None:
    board = Storyboard(
        document_id="actions",
        title="Actions",
        beats=[_create_beat(), _action_beat(_transfer())],
    )

    document = VisualStateTransitionEngine().materialize(board)
    final = document.states[-1].object_states

    assert final["token"].parent_id == "right"
    assert "token" in final["right"].child_ids
    assert "token" not in final["workspace"].child_ids


def test_unsupported_operator_action_and_missing_operand_are_precise() -> None:
    initial = VisualStateTransitionEngine().materialize(
        Storyboard(document_id="base", title="Base", beats=[_create_beat()])
    ).states[0].object_states
    compiler = SemanticActionCompiler()

    with pytest.raises(UnsupportedOperatorActionError, match="equation.*transfer"):
        compiler.lower(
            _transfer(operator=RendererOperator.EQUATION),
            initial,
        )
    with pytest.raises(ValueError, match="omits referenced operands.*token"):
        compiler.lower(
            _transfer(operand_ids=["left", "right"]),
            initial,
        )


def test_failed_precondition_stops_before_mutation() -> None:
    action = _transfer(preconditions=[ActionCondition(
        object_id="token",
        field="parent_id",
        expected="left",
    )])
    board = Storyboard(
        document_id="bad_condition",
        title="Bad condition",
        beats=[_create_beat(), _action_beat(action)],
    )

    with pytest.raises(ValueError, match="precondition failed"):
        VisualStateTransitionEngine().materialize(board)


def test_consume_hides_items_and_reverse_transfer_restores_parent() -> None:
    transfer = _transfer(preconditions=[], postconditions=[])
    reverse = SemanticActionCompiler.reverse(transfer)
    consume = ConsumeAction(
        action_id="consume_fuel",
        action="consume",
        operator=RendererOperator.PROCESS,
        operand_ids=["right", "fuel"],
        consumer_id="right",
        item_ids=["fuel"],
    )
    board = Storyboard(
        document_id="reversible",
        title="Reversible",
        beats=[
            _create_beat(),
            _action_beat(transfer),
            _action_beat(reverse, "reverse"),
            _action_beat(consume, "consume"),
        ],
    )

    document = VisualStateTransitionEngine().materialize(board)
    final = document.states[-1].object_states

    assert final["token"].parent_id == "left"
    assert final["fuel"].lifecycle is ObjectLifecycle.HIDDEN
    assert final["right"].content["consumed"] == ["fuel"]


def test_motion_trajectory_uses_semantic_endpoints_at_intermediate_times() -> None:
    action = _transfer(preconditions=[], postconditions=[])
    board = Storyboard(
        document_id="motion_action",
        title="Motion action",
        beats=[_create_beat(), _action_beat(action)],
    )
    root = LaidOutNode(
        object_id="layout_root",
        kind="root",
        box=LayoutBox(x=0, y=0, width=640, height=360),
        children=[
            LaidOutNode(
                object_id="left",
                kind="component",
                box=LayoutBox(x=80, y=120, width=100, height=80),
            ),
            LaidOutNode(
                object_id="right",
                kind="component",
                box=LayoutBox(x=460, y=120, width=100, height=80),
            ),
            LaidOutNode(
                object_id="token",
                kind="component",
                box=LayoutBox(x=470, y=130, width=40, height=40),
            ),
        ],
    )
    layout = LayoutPlan(
        viewport=Viewport(width=640, height=360, margin=20),
        state_roots={"create": root, "transfer": root},
    )
    alignment = AlignedAudio(
        audio_path=Path("actions.wav").as_posix(),
        duration=2,
        sample_rate=24000,
        phrases=[
            PhraseTiming(
                phrase_id="create_phrase",
                beat_id="create",
                audio_start=0,
                audio_end=1,
            ),
            PhraseTiming(
                phrase_id="transfer_phrase",
                beat_id="transfer",
                audio_start=1,
                audio_end=2,
            ),
        ],
    )

    motion = SemanticAnimationPlanner().plan(board, layout, alignment)
    event = next(item for item in motion.events if item.event_id == "motion_semantic_move_token")
    final_box = LayoutBox(x=470, y=130, width=40, height=40)
    start_box = SemanticFrameRenderer._semantic_action_box(final_box, event, 0)
    middle_box = SemanticFrameRenderer._semantic_action_box(final_box, event, 0.5)
    end_box = SemanticFrameRenderer._semantic_action_box(final_box, event, 1)

    assert event.strategy == "semantic_transfer"
    assert start_box.x < middle_box.x < end_box.x
    assert round(start_box.x) == 110
    assert round(end_box.x) == 490
