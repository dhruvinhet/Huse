"""Tests for structural fingerprints and correctness-safe novelty selection."""

from pathlib import Path

from app.camera import SemanticCameraPlanner
from app.domain.assets import ResolvedAssetSet
from app.domain.camera import CameraOperation
from app.domain.layout import Viewport
from app.domain.lesson import ConceptRelation
from app.domain.strategy import TemplateCapabilities, TemplateMatch
from app.layout import HierarchicalLayoutEngine
from app.novelty import (
    NoveltyManager,
    StructuralFingerprintBuilder,
    structural_similarity,
)
from app.planning import TemplateCompiler
from app.domain.storyboard import VisualObjectSpec
from app.state import VisualStateTransitionEngine
from tests.test_domain_v2 import concept_graph, storyboard
from tests.test_v2_layout_motion_quality import aligned_audio


def _fingerprint():
    board = storyboard()
    document = VisualStateTransitionEngine().materialize(board)
    layout = HierarchicalLayoutEngine().layout(
        document,
        ResolvedAssetSet(),
        Viewport(width=640, height=360, margin=20),
    )
    camera = SemanticCameraPlanner().plan(board, layout, aligned_audio())
    return (
        StructuralFingerprintBuilder.build(board, layout, camera, document),
        board,
        document,
        layout,
        camera,
    )


def test_fingerprint_is_deterministic_and_camera_sensitive() -> None:
    """The same structure hashes identically while choreography changes do not."""

    first, board, document, layout, camera = _fingerprint()
    repeated = StructuralFingerprintBuilder.build(
        board, layout, camera, document
    )
    changed_camera = camera.model_copy(deep=True)
    changed_camera.cues[0].operation = CameraOperation.HOLD
    changed = StructuralFingerprintBuilder.build(
        board, layout, changed_camera, document
    )

    assert first.digest == repeated.digest
    assert structural_similarity(first, repeated) == 1
    assert changed.digest != first.digest
    assert structural_similarity(first, changed) < 1


def test_recent_similarity_is_reported_and_history_is_bounded(tmp_path: Path) -> None:
    """Successful fingerprints persist and become an explicit reported signal."""

    fingerprint, *_rest = _fingerprint()
    manager = NoveltyManager(tmp_path / "history.json")

    assert manager.assess(fingerprint).status == "no_history"
    manager.record(fingerprint)
    assessment = manager.assess(fingerprint)

    assert assessment.status == "similar"
    assert assessment.maximum_recent_similarity == 1
    assert assessment.variant_requested is True


def test_variant_penalty_requires_a_capability_compatible_alternative(
    tmp_path: Path,
) -> None:
    """Novelty cannot authorize an alternative that violates relations."""

    fingerprint, *_rest = _fingerprint()
    fingerprint.template_families = ["pipeline"]
    manager = NoveltyManager(tmp_path / "history.json")
    manager.record(fingerprint)
    primary = TemplateMatch(
        template_id="pipeline.v1",
        concept_ids=["input", "output"],
        score=0.80,
        reason="primary",
    )
    incompatible = TemplateMatch(
        template_id="comparison.v1",
        concept_ids=["input", "output"],
        score=0.76,
        reason="wrong relation",
        capabilities=TemplateCapabilities(
            relation_types=[ConceptRelation.CONTRASTS_WITH]
        ),
    )

    unchanged = manager.adjust_matches(
        [primary, incompatible], concept_graph()
    )
    assert unchanged[0].novelty_penalty == 0

    compatible = incompatible.model_copy(update={
        "template_id": "input_output.v1",
        "capabilities": TemplateCapabilities(
            relation_types=[ConceptRelation.TRANSFORMS_TO]
        ),
    })
    adjusted = manager.adjust_matches([primary, compatible], concept_graph())

    assert adjusted[0].novelty_penalty == 0.10
    assert adjusted[1].novelty_penalty == 0

    layout_only = manager.adjust_matches([primary], concept_graph())
    assert layout_only[0].layout_variant == "vertical"
    assert "reviewed_layout_variant" in layout_only[0].novelty_evidence

    root = VisualObjectSpec(
        object_id="root",
        kind="pipeline",
        semantic_role="process",
        content={"layout": "horizontal"},
        accessibility_label="Pipeline",
    )
    varied = TemplateCompiler._apply_layout_variant(root, layout_only[0])
    assert varied.content["layout"] == "vertical"
