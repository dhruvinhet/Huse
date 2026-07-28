"""Layout, animation, camera, assets, and quality tests for V2."""

from pathlib import Path

from app.camera import SemanticCameraPlanner
from app.domain.assets import ResolvedAssetSet
from app.domain.layout import Viewport
from app.domain.narration import AlignedAudio, PhraseTiming
from app.layout import HierarchicalLayoutEngine
from app.motion import SemanticAnimationPlanner
from app.quality import DeterministicQualityEvaluator
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
            "concept_graph": concept_graph(),
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


def test_semantic_asset_fallback_creates_ready_svg(tmp_path: Path) -> None:
    """Unknown semantic queries resolve to editable line art, not placeholders."""

    from app.domain.assets import AssetKind, AssetQuery
    from app.domain.storyboard import VisualObjectSpec
    item = VisualObjectSpec(
        object_id="robot",
        kind="semantic_asset",
        semantic_role="actor",
        content={"label": "Robot"},
        asset_query=AssetQuery(
            concept="robot learning",
            asset_kind=AssetKind.LINE_ART,
            style_id="whiteboard.default",
        ),
        accessibility_label="Robot learning",
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
