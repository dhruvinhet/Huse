"""Regression tests for the richer semantic whiteboard feature set."""

from app.camera import SemanticCameraPlanner
from app.design import WhiteboardDesignSystem
from app.domain.layout import LaidOutNode, LayoutBox, LayoutPlan, Viewport
from app.domain.narration import AlignedAudio, PhraseTiming, WordTiming
from app.motion import SemanticAnimationPlanner
from app.quality import VisualQualityEvaluator
from app.rendering import SemanticFrameRenderer
from app.templates import builtin_templates
from app.state import VisualStateTransitionEngine
from app.layout import HierarchicalLayoutEngine
from app.domain.assets import ResolvedAssetSet
from tests.test_domain_v2 import storyboard


def word_aligned_audio() -> AlignedAudio:
    """Build phrase and word timing for the standard two-beat fixture."""

    return AlignedAudio(
        audio_path="outputs/audio/test.mp3",
        duration=4,
        sample_rate=24000,
        phrases=[
            PhraseTiming(
                phrase_id="phrase_1",
                beat_id="beat_1",
                audio_start=0,
                audio_end=2,
            ),
            PhraseTiming(
                phrase_id="phrase_2",
                beat_id="beat_2",
                audio_start=2,
                audio_end=4,
            ),
        ],
        words=[
            WordTiming(
                phrase_id="phrase_1",
                beat_id="beat_1",
                text="Introduce",
                audio_start=0.15,
                audio_end=0.55,
            ),
            WordTiming(
                phrase_id="phrase_1",
                beat_id="beat_1",
                text="input",
                audio_start=1.1,
                audio_end=1.4,
            ),
            WordTiming(
                phrase_id="phrase_2",
                beat_id="beat_2",
                text="Emphasize",
                audio_start=2.2,
                audio_end=2.7,
            ),
            WordTiming(
                phrase_id="phrase_2",
                beat_id="beat_2",
                text="result",
                audio_start=3.1,
                audio_end=3.5,
            ),
        ],
    )


def compiled_layout() -> tuple[object, LayoutPlan]:
    """Materialize and lay out the standard storyboard."""

    board = storyboard()
    document = VisualStateTransitionEngine().materialize(board)
    layout = HierarchicalLayoutEngine().layout(
        document,
        ResolvedAssetSet(),
        Viewport(width=1280, height=720, margin=40),
    )
    return board, layout


def test_design_system_resolves_semantic_accents() -> None:
    """Semantic style tokens resolve to a consistent accessible palette."""

    design = WhiteboardDesignSystem()
    input_style = design.resolve("component", "data.input")
    output_style = design.resolve("component", "data.output")

    assert input_style.accent != output_style.accent
    assert design.resolve("label", "concept.primary", dimmed=True).ink == design.muted


def test_expanded_template_library_contains_general_visual_grammars() -> None:
    """Reusable comparisons, cycles, charts, and traces are registered."""

    identifiers = {template.template_id for template in builtin_templates()}

    assert {
        "comparison.v1",
        "cycle.v1",
        "cause_effect.v1",
        "flowchart.v1",
        "bar_chart.v1",
        "equation_derivation.v1",
        "code_trace.v1",
    }.issubset(identifiers)


def test_motion_events_anchor_to_spoken_word_boundaries() -> None:
    """Animation events begin at measured word onsets when available."""

    board, layout = compiled_layout()
    plan = SemanticAnimationPlanner().plan(board, layout, word_aligned_audio())
    onsets = {0.15, 1.1, 2.2, 3.1}

    assert all(event.parameters["sync"] == "word" for event in plan.events)
    assert all(event.start_time in onsets for event in plan.events)


def test_camera_choreography_uses_beat_purpose() -> None:
    """An emphasis beat automatically receives a restrained zoom cue."""

    original, layout = compiled_layout()
    board = original.model_copy(
        update={
            "beats": [
                original.beats[0].model_copy(update={"purpose": "introduce"}),
                original.beats[1].model_copy(update={"purpose": "emphasize"}),
            ]
        }
    )
    plan = SemanticCameraPlanner().plan(board, layout, word_aligned_audio())

    assert plan.cues[0].operation.value == "fit"
    assert plan.cues[1].operation.value == "zoom"
    assert plan.cues[1].parameters["purpose"] == "emphasize"


def test_camera_focus_recovers_visible_content_from_connector_only_attention() -> None:
    """Connector-only attention must not create a close-up of connector glyphs."""

    geometry = {
        "state_root": LayoutBox(x=0, y=0, width=1920, height=1080),
        "lesson_scene": LayoutBox(x=100, y=420, width=1720, height=240),
        "scene_connector_000": LayoutBox(x=900, y=520, width=120, height=24),
        "scene_connector_001": LayoutBox(x=900, y=520, width=120, height=24),
        "scene_evidence": LayoutBox(x=1200, y=430, width=420, height=180),
    }

    targets = SemanticCameraPlanner._focus_targets(
        ["scene_connector_000", "scene_connector_001"],
        geometry,
        viewport_width=1920,
    )

    assert targets == ["scene_evidence"]


def test_connector_path_reveal_follows_route_distance() -> None:
    """A drawing animation reveals a routed path by distance, not a box mask."""

    points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]

    assert SemanticFrameRenderer._partial_polyline(points, 0.25) == [
        (0.0, 0.0),
        (50.0, 0.0),
    ]
    assert SemanticFrameRenderer._partial_polyline(points, 0.75) == [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 50.0),
    ]


def test_visual_quality_gate_rejects_sibling_overlap() -> None:
    """Renderer preflight requests repair for materially colliding siblings."""

    first = LaidOutNode(
        object_id="first",
        kind="component",
        box=LayoutBox(x=40, y=40, width=100, height=80),
    )
    second = LaidOutNode(
        object_id="second",
        kind="component",
        box=LayoutBox(x=80, y=60, width=100, height=80),
    )
    root = LaidOutNode(
        object_id="root",
        kind="pipeline",
        box=LayoutBox(x=20, y=20, width=220, height=160),
        children=[first, second],
    )
    layout = LayoutPlan(
        viewport=Viewport(width=300, height=220, margin=10),
        state_roots={"state": root},
    )

    report = VisualQualityEvaluator().evaluate("layout", layout, {})

    assert report.decision.value == "repair"
    assert any(finding.code == "sibling_overlap" for finding in report.findings)
