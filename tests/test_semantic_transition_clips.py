"""Rendered golden-transition sequences for typed semantic actions."""

from pathlib import Path

import pytest
from PIL import Image, ImageChops

from app.camera import SemanticCameraPlanner
from app.domain.assets import ResolvedAssetSet
from app.domain.generation import AudienceProfile
from app.domain.layout import Viewport
from app.domain.narration import AlignedAudio, PhraseTiming
from app.domain.operations import (
    ConsumeAction,
    MergeAction,
    ProduceAction,
    RouteAction,
    SplitAction,
    TransformAction,
    VisualOperation,
    OperationType,
)
from app.domain.rendering import RenderJob
from app.domain.storyboard import Storyboard, VisualBeat, VisualObjectSpec
from app.domain.visual_intent import RendererOperator
from app.layout import HierarchicalLayoutEngine
from app.motion import SemanticAnimationPlanner
from app.rendering import SemanticFrameRenderer
from app.state import VisualStateTransitionEngine
from app.timeline import PhraseManifestBuilder


def _root() -> VisualObjectSpec:
    return VisualObjectSpec(
        object_id="workspace",
        kind="nested_group",
        semantic_role="transition_workspace",
        content={"operator": "process", "layout": "horizontal"},
        accessibility_label="Transition workspace",
        children=[
            VisualObjectSpec(
                object_id=object_id,
                kind=("tree_node" if object_id == "waypoint" else "component"),
                semantic_role=object_id,
                content={"label": object_id.replace("_", " ").title()},
                accessibility_label=object_id,
            )
            for object_id in ("source", "waypoint", "target", "item_a", "item_b")
        ],
    )


def _action(name: str) -> object:
    common = {
        "action_id": f"golden_{name}",
        "action": name,
        "operator": RendererOperator.PROCESS,
        "duration_hint": 0.75,
    }
    if name == "route":
        return RouteAction(
            **common,
            operand_ids=["source", "waypoint", "target"],
            source_id="source",
            path_ids=["waypoint"],
            target_id="target",
        )
    if name == "split":
        return SplitAction(
            **common,
            operand_ids=["source", "item_a", "item_b"],
            source_id="source",
            output_ids=["item_a", "item_b"],
        )
    if name == "merge":
        return MergeAction(
            **common,
            operand_ids=["item_a", "item_b", "target"],
            input_ids=["item_a", "item_b"],
            target_id="target",
        )
    if name == "consume":
        return ConsumeAction(
            **common,
            operand_ids=["target", "item_a"],
            consumer_id="target",
            item_ids=["item_a"],
        )
    if name == "produce":
        return ProduceAction(
            **common,
            operand_ids=["source", "item_a"],
            producer_id="source",
            item_ids=["item_a"],
        )
    if name == "transform":
        return TransformAction(
            **common,
            operand_ids=["source", "target"],
            source_id="source",
            target_id="target",
        )
    raise AssertionError(name)


def _job(action: object, output: Path) -> tuple[RenderJob, Storyboard]:
    root = _root()
    board = Storyboard(
        document_id=f"golden_{action.action}",
        title=f"Golden {action.action}",
        beats=[
            VisualBeat(
                beat_id="setup",
                section_id="golden",
                concept_ids=["setup"],
                teaching_intent="Show the initial state.",
                phrase_intent="The initial state is visible.",
                estimated_duration=1,
                operations=[VisualOperation(
                    operation_id="create_workspace",
                    operation=OperationType.CREATE,
                    target_ids=[root.object_id],
                    arguments={"objects": [root.model_dump(mode="json")]},
                )],
            ),
            VisualBeat(
                beat_id="action",
                section_id="golden",
                concept_ids=[action.action],
                teaching_intent=f"Animate {action.action}.",
                phrase_intent=f"The objects visibly {action.action}.",
                purpose="transform",
                estimated_duration=1,
                operations=[VisualOperation(
                    operation_id="show_workspace",
                    operation=OperationType.SHOW,
                    target_ids=[root.object_id],
                )],
                semantic_actions=[action],
            ),
        ],
    )
    alignment = AlignedAudio(
        audio_path="golden.wav",
        duration=2,
        sample_rate=24000,
        phrases=[
            PhraseTiming(
                phrase_id="setup_phrase",
                beat_id="setup",
                audio_start=0,
                audio_end=1,
            ),
            PhraseTiming(
                phrase_id="action_phrase",
                beat_id="action",
                audio_start=1,
                audio_end=2,
            ),
        ],
    )
    document = VisualStateTransitionEngine().materialize(board)
    assets = ResolvedAssetSet()
    layout = HierarchicalLayoutEngine().layout(
        document, assets, Viewport(width=640, height=360, margin=20)
    )
    motion = SemanticAnimationPlanner().plan(board, layout, alignment)
    camera = SemanticCameraPlanner().plan(board, layout, alignment)
    manifest = PhraseManifestBuilder().build(board, alignment, fps=4)
    return RenderJob(
        run_id=f"golden_{action.action}",
        document=document,
        assets=assets,
        layout=layout,
        motion=motion,
        camera=camera,
        manifest=manifest,
        output_folder=output.as_posix(),
    ), board


@pytest.mark.parametrize(
    "action_name",
    ["route", "split", "merge", "consume", "produce", "transform"],
)
def test_golden_transition_clip_contains_multiple_pixel_states(
    tmp_path: Path,
    action_name: str,
) -> None:
    """The real frame renderer must produce a visible transition sequence."""

    job, _board = _job(_action(action_name), tmp_path / action_name)
    SemanticFrameRenderer().render(job)
    frames = [
        Image.open(job.output_folder + f"/frame_{number:06d}.png").convert("RGB")
        for number in (5, 6, 7, 8)
    ]
    white = Image.new("RGB", frames[0].size, "white")

    assert all(ImageChops.difference(frame, white).getbbox() for frame in frames)
    assert len({frame.tobytes() for frame in frames}) >= 3
    assert ImageChops.difference(frames[0], frames[1]).getbbox() is not None
    assert ImageChops.difference(frames[1], frames[-1]).getbbox() is not None


@pytest.mark.parametrize("action_name", ["merge", "consume"])
def test_destructive_golden_clip_uses_source_ghost_until_completion(
    tmp_path: Path,
    action_name: str,
) -> None:
    """Source pixels exist mid-transition and disappear only at completion."""

    job, _board = _job(_action(action_name), tmp_path / action_name)
    renderer = SemanticFrameRenderer()
    state = job.document.states[1]
    previous = job.document.states[0]
    root = job.layout.state_roots[state.state_id]
    ordered = sorted(
        (
            node for node in renderer._flatten(root)
            if node.object_id in state.object_states
        ),
        key=lambda node: (
            renderer._render_priority(
                node.kind,
                bool(state.object_states.get(node.object_id).child_ids)
                if node.object_id in state.object_states
                else False,
            ),
            node.z_index,
        ),
    )
    boxes = {node.object_id: node.box for node in ordered}
    previous_boxes = {
        node.object_id: node.box
        for node in renderer._flatten(
            job.layout.state_roots[previous.state_id]
        )
    }
    events = [event for event in job.motion.events if event.beat_id == "action"]
    with_ghost = renderer._render_state(
        job, state, previous, ordered, boxes, previous_boxes, events, 1.25
    )
    without_ghost_state = state.model_copy(update={"transitions": []}, deep=True)
    without_ghost = renderer._render_state(
        job,
        without_ghost_state,
        previous,
        ordered,
        boxes,
        previous_boxes,
        events,
        1.25,
    )

    assert ImageChops.difference(
        with_ghost.convert("RGB"), without_ghost.convert("RGB")
    ).getbbox() is not None
    completed = renderer._render_state(
        job, state, previous, ordered, boxes, previous_boxes, events, 1.75
    )
    completed_without_ghost = renderer._render_state(
        job,
        without_ghost_state,
        previous,
        ordered,
        boxes,
        previous_boxes,
        events,
        1.75,
    )
    assert ImageChops.difference(
        completed.convert("RGB"), completed_without_ghost.convert("RGB")
    ).getbbox() is None
