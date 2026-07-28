"""Smoke test for real V2 semantic PNG rendering."""

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image, ImageChops, ImageDraw

from app.camera import SemanticCameraPlanner
from app.domain.assets import ResolvedAssetSet
from app.domain.camera import CameraCue, CameraOperation, CameraPlan
from app.domain.layout import LayoutBox, Viewport
from app.domain.motion import MotionPlan
from app.domain.rendering import RenderJob
from app.domain.visual_document import ObjectState, VisualState
from app.layout import HierarchicalLayoutEngine
from app.motion import SemanticAnimationPlanner
from app.rendering import SemanticFrameRenderer
from app.state import VisualStateTransitionEngine
from app.timeline import PhraseManifestBuilder
from tests.test_domain_v2 import storyboard
from tests.test_v2_layout_motion_quality import aligned_audio


def test_semantic_renderer_creates_nonblank_continuous_frames(tmp_path: Path) -> None:
    """Persistent semantic states render into correctly named PNG files."""

    board = storyboard()
    alignment = aligned_audio()
    document = VisualStateTransitionEngine().materialize(board)
    assets = ResolvedAssetSet()
    layout = HierarchicalLayoutEngine().layout(
        document,
        assets,
        Viewport(width=640, height=360, margin=24),
    )
    motion = SemanticAnimationPlanner().plan(board, layout, alignment)
    camera = SemanticCameraPlanner().plan(board, layout, alignment)
    manifest = PhraseManifestBuilder().build(board, alignment, fps=2)
    result = SemanticFrameRenderer().render(
        RenderJob(
            run_id="render_test",
            document=document,
            assets=assets,
            layout=layout,
            motion=motion,
            camera=camera,
            manifest=manifest,
            output_folder=tmp_path.as_posix(),
        )
    )

    files = sorted(tmp_path.glob("frame_*.png"))
    assert result.total_frames == 8
    assert [path.name for path in files] == [
        f"frame_{index:06d}.png" for index in range(1, 9)
    ]
    image = Image.open(files[-1]).convert("RGB")
    white = Image.new("RGB", image.size, "white")
    assert ImageChops.difference(image, white).getbbox() is not None


def test_semantic_renderer_reuses_pixel_identical_hold_frames(
    tmp_path: Path,
) -> None:
    """Static beat intervals reuse encoded frames instead of redrawing them."""

    board = storyboard()
    alignment = aligned_audio()
    document = VisualStateTransitionEngine().materialize(board)
    assets = ResolvedAssetSet()
    layout = HierarchicalLayoutEngine().layout(
        document,
        assets,
        Viewport(width=640, height=360, margin=24),
    )
    camera = CameraPlan(
        duration=4,
        cues=[
            CameraCue(
                cue_id="cue_1",
                beat_id="beat_1",
                start_time=0,
                duration=2,
                operation=CameraOperation.FIT,
            ),
            CameraCue(
                cue_id="cue_2",
                beat_id="beat_2",
                start_time=2,
                duration=2,
                operation=CameraOperation.FIT,
            ),
        ],
    )
    manifest = PhraseManifestBuilder().build(board, alignment, fps=2)
    renderer = SemanticFrameRenderer()
    original_reuse = renderer._link_or_copy
    renderer._link_or_copy = MagicMock(wraps=original_reuse)

    renderer.render(
        RenderJob(
            run_id="reuse_test",
            document=document,
            assets=assets,
            layout=layout,
            motion=MotionPlan(duration=4),
            camera=camera,
            manifest=manifest,
            output_folder=tmp_path.as_posix(),
        )
    )

    assert renderer._link_or_copy.call_count == 6
    assert len(list(tmp_path.glob("frame_*.png"))) == 8


def test_connector_anchors_use_facing_box_boundaries() -> None:
    """Vertically separated nodes connect bottom-to-top, never center-to-center."""

    renderer = SemanticFrameRenderer()
    source = LayoutBox(x=100, y=50, width=200, height=100)
    target = LayoutBox(x=400, y=300, width=200, height=100)

    start, end, horizontal = renderer._connector_anchors(source, target)

    assert not horizontal
    assert start == (200, 150)
    assert end == (500, 300)


def test_connector_route_avoids_intermediate_nodes() -> None:
    """Orthogonal connector routing selects a clear path around obstacles."""

    renderer = SemanticFrameRenderer()
    obstacle = LayoutBox(x=250, y=50, width=100, height=100)
    route = renderer._orthogonal_route(
        start=(100, 100),
        end=(500, 100),
        horizontal=True,
        obstacles=[obstacle],
        boxes={
            "source": LayoutBox(x=20, y=60, width=80, height=80),
            "obstacle": obstacle,
            "target": LayoutBox(x=500, y=60, width=80, height=80),
        },
    )

    assert not any(
        renderer._segment_intersects_box(first, second, obstacle)
        for first, second in zip(route, route[1:])
    )


def test_active_view_excludes_unrelated_persistent_roots() -> None:
    """Camera context keeps connected groups and removes unrelated history."""

    state = VisualState(
        state_id="state",
        beat_id="beat",
        object_states={
            "group_a": ObjectState(
                object_id="group_a",
                kind="pipeline",
                child_ids=["a"],
            ),
            "a": ObjectState(
                object_id="a",
                kind="component",
                parent_id="group_a",
            ),
            "group_b": ObjectState(
                object_id="group_b",
                kind="pipeline",
                child_ids=["b"],
            ),
            "b": ObjectState(
                object_id="b",
                kind="component",
                parent_id="group_b",
            ),
            "unrelated": ObjectState(
                object_id="unrelated",
                kind="component",
            ),
            "a_to_b": ObjectState(
                object_id="a_to_b",
                kind="connector",
                content={"source_id": "a", "target_id": "b"},
            ),
        },
    )
    cue = CameraCue(
        cue_id="cue",
        beat_id="beat",
        start_time=0,
        duration=1,
        operation=CameraOperation.FIT,
        target_ids=["a_to_b"],
    )

    selected = SemanticFrameRenderer()._visible_context_ids(state, cue)

    assert selected == {"group_a", "a", "group_b", "b", "a_to_b"}


def test_adaptive_text_stays_inside_its_layout_box() -> None:
    """Long labels shrink and wrap instead of painting beyond their box."""

    canvas = Image.new("RGBA", (500, 180), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    box = (40, 30, 460, 150)

    SemanticFrameRenderer()._draw_centered_text(
        draw,
        box,
        "A deliberately long semantic label that must wrap cleanly",
        42,
        (0, 0, 0, 255),
    )

    painted = canvas.getbbox()
    assert painted is not None
    assert painted[0] >= box[0]
    assert painted[1] >= box[1]
    assert painted[2] <= box[2]
    assert painted[3] <= box[3]
