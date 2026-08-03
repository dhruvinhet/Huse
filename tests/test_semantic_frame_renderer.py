"""Smoke test for real V2 semantic PNG rendering."""

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image, ImageChops, ImageDraw

from app.camera import SemanticCameraPlanner
from app.domain.assets import (
    AssetSource,
    ResolvedAssetSet,
    ResolvedSemanticAsset,
)
from app.domain.camera import CameraCue, CameraOperation, CameraPlan
from app.domain.layout import LayoutBox, Viewport
from app.domain.motion import MotionEvent, MotionPlan
from app.domain.rendering import FrameSequence, RenderJob
from app.domain.visual_document import ObjectState, VisualState
from app.layout import HierarchicalLayoutEngine
from app.motion import SemanticAnimationPlanner
from app.rendering import SemanticFrameRenderer
from app.rendering.operator_plugins import OperatorRendererRegistry
from app.quality import RenderedFrameQualityEvaluator
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


def test_future_highlight_does_not_hide_an_already_created_object() -> None:
    """Renderer event selection respects create time instead of list order."""

    events = [
        MotionEvent(
            event_id="create",
            beat_id="beat",
            operation_id="create_root",
            object_ids=["root"],
            strategy="stage_flow",
            start_time=0,
            duration=1,
            parameters={"operation_type": "create"},
        ),
        MotionEvent(
            event_id="highlight",
            beat_id="beat",
            operation_id="highlight_root",
            object_ids=["root"],
            strategy="pulse_highlight",
            start_time=5,
            duration=1,
            parameters={"operation_type": "highlight"},
        ),
    ]

    renderer = SemanticFrameRenderer()

    assert renderer._event_at(events, 2).event_id == "create"
    assert renderer._reveal_time(events) == 0


def test_component_kind_resolves_to_evidence_card_plugin() -> None:
    """Leaf components must not be overwritten by a container registration."""

    registry = OperatorRendererRegistry()

    assert registry._kinds["component"].__class__.__name__ == "CardPlugin"


def test_component_card_renders_its_resolved_semantic_asset(tmp_path: Path) -> None:
    """Asset queries enrich concept cards instead of remaining unused metadata."""

    icon = tmp_path / "icon.svg"
    icon.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
        '<circle cx="32" cy="32" r="25" fill="none" stroke="#1261a0" '
        'stroke-width="8"/></svg>',
        encoding="utf-8",
    )
    job = MagicMock()
    job.assets = ResolvedAssetSet(assets=[
        ResolvedSemanticAsset(
            asset_id="asset_concept",
            query_digest="digest",
            source=AssetSource.GENERATED,
            path=icon.as_posix(),
            mime_type="image/svg+xml",
            license_id="generated-internal",
            content_hash="hash",
            editable=True,
            ready=True,
        )
    ])
    canvas = Image.new("RGBA", (500, 240), (255, 255, 255, 255))
    state = ObjectState(
        object_id="concept",
        kind="component",
        content={
            "label": "General concept",
            "detail": "Grounded explanatory evidence.",
            "asset_slot": "left",
        },
    )

    SemanticFrameRenderer()._draw_object(
        canvas,
        state,
        LayoutBox(x=40, y=30, width=420, height=180),
        {"concept": LayoutBox(x=40, y=30, width=420, height=180)},
        job,
    )

    left_crop = canvas.crop((50, 70, 140, 170)).convert("RGB")
    white = Image.new("RGB", left_crop.size, "white")
    assert ImageChops.difference(left_crop, white).getbbox() is not None


def test_rendered_pixel_gate_rejects_a_blank_opening(tmp_path: Path) -> None:
    """Pixel QA catches renderer defects even when multimodal QA is disabled."""

    for number in range(1, 13):
        image = Image.new("RGB", (320, 180), "white")
        if number >= 10:
            ImageDraw.Draw(image).rectangle((40, 30, 280, 150), fill="black")
        image.save(tmp_path / f"frame_{number:06d}.png")
    frames = FrameSequence(
        folder=tmp_path.as_posix(),
        total_frames=12,
        fps=4,
        sample_paths=[
            (tmp_path / "frame_000001.png").as_posix(),
            (tmp_path / "frame_000012.png").as_posix(),
        ],
    )

    report = RenderedFrameQualityEvaluator().evaluate(frames)

    assert report.decision.value == "repair"
    assert "rendered_opening_blank" in {
        finding.code for finding in report.findings
    }
