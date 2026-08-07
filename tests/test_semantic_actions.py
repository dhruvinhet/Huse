"""Typed semantic-action validation, lowering, state, and motion coverage."""

from pathlib import Path

import pytest

from app.domain.layout import LayoutBox, LayoutPlan, LaidOutNode, Viewport
from app.domain.narration import AlignedAudio, PhraseTiming
from app.domain.motion import MotionEvent
from app.domain.operations import (
    ActionCondition,
    ConsumeAction,
    AccumulateAction,
    CompareAction,
    GroupAction,
    MergeAction,
    ProduceAction,
    RouteAction,
    SplitAction,
    TransformAction,
    SubstituteAction,
    TraceAction,
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
    OPERATOR_STATE_MODELS,
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


def test_destructive_actions_preserve_explicit_transition_snapshots() -> None:
    """Merge and consume retain visible source pixels until interpolation ends."""

    merge = MergeAction(
        action_id="merge_inputs",
        action="merge",
        operator=RendererOperator.PROCESS,
        operand_ids=["left", "token", "right"],
        input_ids=["left", "token"],
        target_id="right",
    )
    consume = ConsumeAction(
        action_id="consume_fuel_after_merge",
        action="consume",
        operator=RendererOperator.PROCESS,
        operand_ids=["right", "fuel"],
        consumer_id="right",
        item_ids=["fuel"],
    )
    board = Storyboard(
        document_id="transition_snapshots",
        title="Transition snapshots",
        beats=[
            _create_beat(),
            _action_beat(merge, "merge"),
            _action_beat(consume, "consume"),
        ],
    )

    document = VisualStateTransitionEngine().materialize(board)
    merge_transition = document.states[1].transitions[0]
    consume_transition = document.states[2].transitions[0]

    assert merge_transition.previous_object_states["left"].lifecycle is ObjectLifecycle.VISIBLE
    assert merge_transition.next_object_states["left"].lifecycle is ObjectLifecycle.HIDDEN
    assert consume_transition.previous_object_states["fuel"].lifecycle is ObjectLifecycle.VISIBLE
    assert consume_transition.next_object_states["fuel"].lifecycle is ObjectLifecycle.HIDDEN


@pytest.mark.parametrize(
    ("action", "strategy", "moving_id"),
    [
        (
            SplitAction(
                action_id="split_token",
                action="split",
                operator=RendererOperator.PROCESS,
                operand_ids=["left", "token", "fuel"],
                source_id="left",
                output_ids=["token", "fuel"],
            ),
            "semantic_split",
            "token",
        ),
        (
            MergeAction(
                action_id="merge_token",
                action="merge",
                operator=RendererOperator.PROCESS,
                operand_ids=["token", "fuel", "right"],
                input_ids=["token", "fuel"],
                target_id="right",
            ),
            "semantic_merge",
            "token",
        ),
        (
            ConsumeAction(
                action_id="consume_token",
                action="consume",
                operator=RendererOperator.PROCESS,
                operand_ids=["right", "token"],
                consumer_id="right",
                item_ids=["token"],
            ),
            "semantic_consume",
            "token",
        ),
        (
            ProduceAction(
                action_id="produce_token",
                action="produce",
                operator=RendererOperator.PROCESS,
                operand_ids=["left", "token"],
                producer_id="left",
                item_ids=["token"],
            ),
            "semantic_produce",
            "token",
        ),
        (
            TransformAction(
                action_id="transform_left",
                action="transform",
                operator=RendererOperator.PROCESS,
                operand_ids=["left", "right"],
                source_id="left",
                target_id="right",
            ),
            "semantic_transform",
            "left",
        ),
    ],
)
def test_action_families_have_distinct_pixel_trajectories(
    action: object,
    strategy: str,
    moving_id: str,
) -> None:
    """Every state-changing action emits concrete interpolated geometry."""

    board = Storyboard(
        document_id=f"motion_{strategy}",
        title="Action motion",
        beats=[_create_beat(), _action_beat(action)],
    )
    root = LaidOutNode(
        object_id="layout_root",
        kind="root",
        box=LayoutBox(x=0, y=0, width=640, height=360),
        children=[
            LaidOutNode(object_id="left", kind="component", box=LayoutBox(x=60, y=120, width=80, height=60)),
            LaidOutNode(object_id="right", kind="component", box=LayoutBox(x=500, y=120, width=80, height=60)),
            LaidOutNode(object_id="token", kind="component", box=LayoutBox(x=220, y=120, width=50, height=50)),
            LaidOutNode(object_id="fuel", kind="component", box=LayoutBox(x=340, y=120, width=50, height=50)),
        ],
    )
    layout = LayoutPlan(
        viewport=Viewport(width=640, height=360, margin=20),
        state_roots={"create": root, "transfer": root},
    )
    alignment = AlignedAudio(
        audio_path="actions.wav",
        duration=2,
        sample_rate=24000,
        phrases=[
            PhraseTiming(phrase_id="create", beat_id="create", audio_start=0, audio_end=1),
            PhraseTiming(phrase_id="action", beat_id="transfer", audio_start=1, audio_end=2),
        ],
    )

    event = next(
        item
        for item in SemanticAnimationPlanner().plan(board, layout, alignment).events
        if item.parameters.get("semantic_action") == action.action
    )
    trajectories = event.parameters["trajectories"]

    assert event.strategy == strategy
    assert moving_id in trajectories
    assert len(trajectories[moving_id]) >= 2


def test_compare_has_visible_pixel_scale_interpolation() -> None:
    final = LayoutBox(x=100, y=80, width=120, height=70)

    start = SemanticFrameRenderer._semantic_compare_box(final, 0)
    middle = SemanticFrameRenderer._semantic_compare_box(final, 0.5)
    end = SemanticFrameRenderer._semantic_compare_box(final, 1)

    assert start == final
    assert end == final
    assert middle.width > final.width
    assert middle.height > final.height


def test_transform_interpolates_position_and_geometry() -> None:
    event = MotionEvent(
        event_id="transform_geometry",
        beat_id="transform",
        operation_id="semantic_transform",
        object_ids=["source", "target"],
        strategy="semantic_transform",
        start_time=0,
        duration=1,
        parameters={
            "geometry_transitions": {
                "source": {
                    "start": [20, 40, 80, 50],
                    "end": [300, 120, 160, 100],
                }
            }
        },
    )
    final = LayoutBox(x=20, y=40, width=80, height=50)

    middle = SemanticFrameRenderer._semantic_transform_box(
        final, event, 0.5, "source"
    )

    assert middle == LayoutBox(x=160, y=80, width=120, height=75)


def _supported_action(action: str, operator: RendererOperator) -> object:
    common = {
        "action_id": f"{operator.value}_{action}",
        "action": action,
        "operator": operator,
    }
    if action == "transfer":
        return TransferAction(
            **common,
            operand_ids=["left", "right", "token"],
            source_id="left",
            target_id="right",
            payload_ids=["token"],
        )
    if action == "route":
        return RouteAction(
            **common,
            operand_ids=["left", "token", "right"],
            source_id="left",
            path_ids=["token"],
            target_id="right",
        )
    if action == "split":
        return SplitAction(
            **common,
            operand_ids=["left", "split_a", "split_b"],
            source_id="left",
            output_ids=["split_a", "split_b"],
        )
    if action == "merge":
        return MergeAction(
            **common,
            operand_ids=["left", "token", "right"],
            input_ids=["left", "token"],
            target_id="right",
        )
    if action == "group":
        return GroupAction(
            **common,
            operand_ids=["right", "token", "fuel"],
            member_ids=["token", "fuel"],
            group_id="right",
        )
    if action == "compare":
        return CompareAction(
            **common,
            operand_ids=["left", "right"],
            left_id="left",
            right_id="right",
        )
    if action == "consume":
        return ConsumeAction(
            **common,
            operand_ids=["right", "token"],
            consumer_id="right",
            item_ids=["token"],
        )
    if action == "produce":
        return ProduceAction(
            **common,
            operand_ids=["left", "token"],
            producer_id="left",
            item_ids=["token"],
        )
    if action == "transform":
        return TransformAction(
            **common,
            operand_ids=["left", "right"],
            source_id="left",
            target_id="right",
        )
    if action == "substitute":
        return SubstituteAction(
            **common,
            operand_ids=["left", "right"],
            source_id="left",
            replacement_id="right",
        )
    if action == "accumulate":
        return AccumulateAction(
            **common,
            operand_ids=["right", "token", "fuel"],
            accumulator_id="right",
            item_ids=["token", "fuel"],
        )
    if action == "trace":
        return TraceAction(
            **common,
            operand_ids=["left", "token", "right"],
            path_ids=["left", "token", "right"],
        )
    raise AssertionError(action)


@pytest.mark.parametrize(
    ("operator", "action_name"),
    [
        (operator, action_name)
        for operator, model in OPERATOR_STATE_MODELS.items()
        for action_name in sorted(model.supported_actions)
    ],
)
def test_every_supported_operator_action_pair_has_state_and_motion_adapters(
    operator: RendererOperator,
    action_name: str,
) -> None:
    """No advertised operator/action pair can reach rendering unimplemented."""

    states = VisualStateTransitionEngine().materialize(
        Storyboard(document_id="pair_base", title="Pair base", beats=[_create_beat()])
    ).states[0].object_states
    action = _supported_action(action_name, operator)

    operations = SemanticActionCompiler().lower(action, states)

    assert operations
    assert action_name in SemanticAnimationPlanner._ACTION_STRATEGIES
