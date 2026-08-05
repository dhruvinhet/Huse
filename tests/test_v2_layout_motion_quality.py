"""Layout, animation, camera, assets, and quality tests for V2."""

from pathlib import Path

from app.camera import SemanticCameraPlanner
from app.domain.assets import (
    AssetPresentation,
    AssetSource,
    ResolvedAssetSet,
    ResolvedSemanticAsset,
)
from app.domain.layout import Viewport
from app.domain.narration import AlignedAudio, PhraseTiming
from app.domain.operations import OperationType, VisualOperation
from app.domain.storyboard import ShotPlan, VisualObjectSpec
from app.domain.visual_document import (
    ObjectLifecycle,
    ObjectState,
    VisualDocument,
    VisualState,
)
from app.layout import HierarchicalLayoutEngine
from app.motion import SemanticAnimationPlanner
from app.quality import DeterministicQualityEvaluator, VisualQualityEvaluator
from app.semantic_assets import CatalogSemanticAssetResolver
from app.state import VisualStateTransitionEngine
from tests.test_domain_v2 import concept_graph, storyboard


def aligned_audio() -> AlignedAudio:
    """Return continuous timing for the two fixture beats."""

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
    )


def test_layout_motion_camera_and_quality_form_valid_plan() -> None:
    """The semantic compiler produces viewport-safe, phrase-spanning plans."""

    board = storyboard()
    document = VisualStateTransitionEngine().materialize(board)
    layout = HierarchicalLayoutEngine().layout(
        document,
        ResolvedAssetSet(),
        Viewport(width=1280, height=720, margin=40),
    )
    motion = SemanticAnimationPlanner().plan(board, layout, aligned_audio())
    camera = SemanticCameraPlanner().plan(board, layout, aligned_audio())
    report = DeterministicQualityEvaluator().evaluate(
        "compiled",
        motion,
        {
            # This fixture intentionally contains no connector; keep this test
            # focused on layout/motion by evaluating a relation-free graph.
            "concept_graph": concept_graph().model_copy(update={"edges": []}),
            "storyboard": board,
            "assets": ResolvedAssetSet(),
            "document": document,
            "layout": layout,
            "motion": motion,
        },
    )

    assert not layout.diagnostics
    assert {event.beat_id for event in motion.events} == {"beat_1", "beat_2"}
    assert len(camera.cues) == 2
    assert report.decision.value == "pass"
    assert report.scores["semantic_coverage"] == 1


def test_tree_operator_overrides_generic_horizontal_layout_hint() -> None:
    """A semantic tree must use depth instead of collapsing into a flat rail."""

    tree = VisualObjectSpec(
        object_id="concept_tree",
        kind="nested_group",
        semantic_role="compiled_visual_intent",
        content={
            "label": "Search concepts",
            "layout": "horizontal",
            "operator": "tree",
        },
        children=[
            VisualObjectSpec(
                object_id=f"concept_{index}",
                kind="component",
                semantic_role="concept",
                content={
                    "label": f"Concept {index}",
                    "detail": "A complete explanation of this search concept.",
                },
                accessibility_label=f"Concept {index}",
            )
            for index in range(7)
        ] + [
            VisualObjectSpec(
                object_id="evidence",
                kind="callout",
                semantic_role="evidence",
                content={"text": "A concise piece of supporting evidence."},
                accessibility_label="Supporting evidence",
            ),
        ],
        accessibility_label="Search concept tree",
    )
    board = storyboard()
    first = board.beats[0].model_copy(update={
        "operations": [
            VisualOperation(
                operation_id="create_tree",
                operation=OperationType.CREATE,
                target_ids=[tree.object_id],
                arguments={"objects": [tree.model_dump(mode="json")]},
            )
        ],
    })
    board = board.model_copy(update={"beats": [first]})

    document = VisualStateTransitionEngine().materialize(board)
    layout = HierarchicalLayoutEngine().layout(
        document,
        ResolvedAssetSet(),
        Viewport(width=1920, height=1080),
    )
    report = VisualQualityEvaluator().evaluate(
        "layout",
        layout,
        {"layout": layout},
    )
    compiled_report = DeterministicQualityEvaluator().evaluate(
        "layout",
        layout,
        {"layout": layout, "document": document},
    )
    content_root = next(iter(layout.state_roots.values())).children[0]

    assert content_root.box.height / layout.viewport.height >= 0.20
    assert "canvas_underused" not in {
        finding.code for finding in report.findings
    }
    assert "semantic_text_geometry_unreadable" not in {
        finding.code for finding in compiled_report.findings
    }


def test_semantic_asset_diagnostic_fallback_creates_ready_svg(tmp_path: Path) -> None:
    """Unknown semantic queries resolve to editable line art, not placeholders."""

    from app.domain.assets import AssetKind, AssetQuery
    from app.domain.storyboard import VisualObjectSpec
    item = VisualObjectSpec(
        object_id="quasar",
        kind="semantic_asset",
        semantic_role="actor",
        content={"label": "Quasar luminosity"},
        asset_query=AssetQuery(
            concept="purple quasar luminosity",
            asset_kind=AssetKind.LINE_ART,
            style_id="whiteboard.default",
        ),
        accessibility_label="Quasar luminosity",
    )
    board = storyboard().model_copy(
        update={"initial_objects": [item]},
        deep=True,
    )
    generated_dir = tmp_path / "v2" / "semantic_assets"
    result = CatalogSemanticAssetResolver(generated_dir=generated_dir).resolve(board)

    assert len(result.assets) == 1
    assert result.assets[0].ready
    assert "placeholder" not in result.assets[0].path
    assert (generated_dir / f"{result.assets[0].query_digest}.svg").is_file()


def test_generated_diagram_layout_reserves_aspect_correct_space() -> None:
    """A composition receives diagram geometry rather than an icon-sized slot."""

    document = VisualDocument(
        document_id="diagram_document",
        states=[VisualState(
            state_id="diagram_state",
            beat_id="diagram_beat",
            object_states={
                "unknown_art": ObjectState(
                    object_id="unknown_art",
                    kind="semantic_asset",
                )
            },
        )],
    )
    assets = ResolvedAssetSet(assets=[ResolvedSemanticAsset(
        asset_id="asset_unknown_art",
        query_digest="diagram-digest",
        source=AssetSource.GENERATED,
        presentation=AssetPresentation.DIAGRAM,
        path="generated.svg",
        mime_type="image/svg+xml",
        content_hash="diagram-hash",
        editable=True,
        ready=True,
        intrinsic_width=800,
        intrinsic_height=400,
        aspect_ratio=2,
    )])

    layout = HierarchicalLayoutEngine().layout(
        document,
        assets,
        Viewport(width=1280, height=720, margin=40),
    )
    diagram = layout.state_roots["diagram_state"].children[0]

    assert diagram.box.width >= 500
    assert diagram.box.height >= 250
    assert abs(diagram.box.width / diagram.box.height - 2.0) < 0.01


def test_coverage_finding_names_missing_concept_ids() -> None:
    """Repair diagnostics tell providers and deterministic repair what is absent."""

    board = storyboard()
    incomplete = board.model_copy(
        update={
            "beats": [
                beat.model_copy(update={"concept_ids": ["input"]})
                for beat in board.beats
            ]
        }
    )

    report = DeterministicQualityEvaluator().evaluate(
        "storyboard",
        incomplete,
        {"concept_graph": concept_graph()},
    )

    finding = next(
        item for item in report.findings
        if item.code == "semantic_coverage_low"
    )
    assert "output" in finding.message


def test_shot_plan_cleans_history_and_camera_uses_source_geometry() -> None:
    """Focused shots hide history and emit an authoritative crop rectangle."""

    original = storyboard()
    board = original.model_copy(
        update={
            "beats": [
                beat.model_copy(
                    update={"shot_plan": ShotPlan.for_purpose(beat.purpose)}
                )
                for beat in original.beats
            ]
        }
    )
    document = VisualStateTransitionEngine().materialize(board)
    second = document.states[1].object_states
    assert second["input_box"].lifecycle is ObjectLifecycle.HIDDEN
    assert second["output_box"].lifecycle is not ObjectLifecycle.HIDDEN

    layout = HierarchicalLayoutEngine().layout(
        document,
        ResolvedAssetSet(),
        Viewport(width=1280, height=720, margin=40),
    )
    camera = SemanticCameraPlanner().plan(board, layout, aligned_audio())
    assert all("source_left" in cue.parameters for cue in camera.cues)
    assert "output_box" in camera.cues[1].target_ids
