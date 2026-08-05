"""Evidence-based quality gates and risk-routing regression tests."""

from pathlib import Path

from PIL import Image, ImageDraw

from app.domain.generation import AudienceProfile
from app.domain.layout import LaidOutNode, LayoutBox, LayoutPlan, Viewport
from app.domain.motion import MotionEvent, MotionPlan
from app.domain.narration import AlignedAudio, PhraseTiming
from app.domain.operations import OperationType, VisualOperation
from app.domain.quality import EvaluationDecision, QualityReport
from app.domain.quality import FindingSeverity, QualityFinding
from app.domain.repair import RepairStage
from app.domain.rendering import FrameSequence
from app.domain.strategy import TemplateMatch
from app.domain.storyboard import VisualBeat
from app.domain.visual_document import ObjectState, VisualDocument, VisualState
from app.quality import (
    DeterministicQualityEvaluator,
    EducationalQualityEvaluator,
    QualityReviewPolicy,
    QualityRepairPlanner,
    RenderedFrameQualityEvaluator,
    VisualQualityEvaluator,
)
from app.state import VisualStateTransitionEngine
from tests.test_domain_v2 import concept_graph, storyboard


def test_diagram_correctness_rejects_a_reversed_grounded_connector() -> None:
    document = VisualDocument(
        document_id="reversed",
        states=[VisualState(
            state_id="state",
            beat_id="beat_1",
            object_states={
                "input_box": ObjectState(
                    object_id="input_box",
                    kind="component",
                    metadata={"concept_ids": ["input"]},
                ),
                "output_box": ObjectState(
                    object_id="output_box",
                    kind="component",
                    metadata={"concept_ids": ["output"]},
                ),
                "reversed_edge": ObjectState(
                    object_id="reversed_edge",
                    kind="connector",
                    content={
                        "source_id": "output_box",
                        "target_id": "input_box",
                    },
                ),
            },
        )],
    )

    report = DeterministicQualityEvaluator().evaluate(
        "compiled",
        document,
        {"concept_graph": concept_graph(), "document": document},
    )

    finding = next(
        item for item in report.findings
        if item.code == "connector_relation_reversed"
    )
    assert report.decision is EvaluationDecision.REPAIR
    assert report.scores["diagram_correctness"] == 0
    assert finding.object_ids == ["reversed_edge", "output_box", "input_box"]
    assert finding.patch_paths == ["/beats"]


def test_alignment_score_measures_event_windows_instead_of_presence() -> None:
    board = storyboard()
    alignment = AlignedAudio(
        audio_path="test.mp3",
        duration=4,
        sample_rate=24_000,
        phrases=[
            PhraseTiming(
                phrase_id="p1",
                beat_id="beat_1",
                audio_start=0,
                audio_end=2,
            ),
            PhraseTiming(
                phrase_id="p2",
                beat_id="beat_2",
                audio_start=2,
                audio_end=4,
            ),
        ],
    )
    motion = MotionPlan(
        duration=4,
        events=[MotionEvent(
            event_id="late_event",
            beat_id="beat_1",
            operation_id="create_input",
            object_ids=["input_box"],
            strategy="fade_in",
            start_time=2.1,
            duration=0.5,
        )],
    )

    report = DeterministicQualityEvaluator().evaluate(
        "compiled",
        motion,
        {
            "storyboard": board,
            "motion": motion,
            "alignment": alignment,
            "audience": AudienceProfile(learning_goal="Understand"),
        },
    )

    assert report.scores["alignment"] == 0
    assert "narration_visual_alignment_invalid" in {
        finding.code for finding in report.findings
    }


def test_transform_requires_semantic_not_only_lifecycle_delta() -> None:
    base = storyboard()
    transform = base.beats[1].model_copy(update={"purpose": "transform"})
    board = base.model_copy(update={"beats": [base.beats[0], transform]})
    document = VisualStateTransitionEngine().materialize(board)

    report = DeterministicQualityEvaluator().evaluate(
        "compiled",
        document,
        {"storyboard": board, "document": document},
    )

    finding = next(
        item for item in report.findings
        if item.code == "semantic_state_delta_missing"
    )
    assert report.scores["semantic_state_delta"] == 0
    assert finding.beat_id == "beat_2"
    assert finding.repair_scope == "beat"


def test_visual_obligation_requires_a_visible_concept_and_action() -> None:
    missing = VisualBeat(
        beat_id="transform",
        section_id="section",
        concept_ids=["missing_concept"],
        teaching_intent="Transform the missing concept.",
        phrase_intent="Show the missing concept changing.",
        purpose="transform",
        estimated_duration=2,
        operations=[VisualOperation(
            operation_id="highlight_only",
            operation=OperationType.HIGHLIGHT,
            target_ids=["input_box"],
        )],
    )
    board = storyboard().model_copy(update={"beats": [storyboard().beats[0], missing]})
    document = VisualStateTransitionEngine().materialize(board)

    report = EducationalQualityEvaluator().evaluate(
        "compiled",
        board,
        {"storyboard": board, "document": document},
    )

    codes = {finding.code for finding in report.findings}
    assert "visual_obligation_unrepresented" in codes
    assert "visual_obligation_action_missing" in codes


def test_rendered_gate_samples_transitions_and_reports_safe_area_pixels(
    tmp_path: Path,
) -> None:
    for number in range(1, 5):
        image = Image.new("RGB", (320, 180), "white")
        ImageDraw.Draw(image).rectangle((0, 20, 180, 160), fill="black")
        image.save(tmp_path / f"frame_{number:06d}.png")
    frames = FrameSequence(
        folder=tmp_path.as_posix(),
        total_frames=4,
        fps=2,
        sample_paths=[(tmp_path / "frame_000004.png").as_posix()],
    )

    report = RenderedFrameQualityEvaluator().evaluate(frames)

    finding = next(
        item for item in report.findings
        if item.code == "rendered_safe_area_clipped"
    )
    assert report.decision is EvaluationDecision.REPAIR
    assert finding.frame_numbers == [4]
    assert finding.repair_target == "layout"


def test_review_policy_is_conditional_and_records_stable_reasons() -> None:
    policy = QualityReviewPolicy()
    report = QualityReport(
        overall_score=1,
        scores={"deterministic": 1},
        decision=EvaluationDecision.PASS,
    )
    high_confidence = TemplateMatch(
        template_id="pipeline.v1",
        concept_ids=["input"],
        score=0.9,
        reason="Strong structural and lexical match.",
    )

    assert policy.storyboard_reasons(report, []) == [
        "no_reviewed_template_match"
    ]
    assert policy.storyboard_reasons(report, [high_confidence]) == []


def test_visual_preflight_detects_crossing_relation_paths() -> None:
    content = LaidOutNode(
        object_id="content",
        kind="graph",
        box=LayoutBox(x=0, y=0, width=400, height=400),
        children=[
            LaidOutNode(
                object_id=object_id,
                kind="component",
                box=LayoutBox(x=x, y=y, width=40, height=40),
            )
            for object_id, x, y in (
                ("a", 20, 20),
                ("b", 340, 340),
                ("c", 340, 20),
                ("d", 20, 340),
            )
        ],
    )
    layout = LayoutPlan(
        viewport=Viewport(width=400, height=400, margin=0),
        state_roots={
            "state": LaidOutNode(
                object_id="root",
                kind="state",
                box=LayoutBox(x=0, y=0, width=400, height=400),
                children=[content],
            )
        },
    )
    document = VisualDocument(
        document_id="crossing",
        states=[VisualState(
            state_id="state",
            beat_id="beat",
            object_states={
                **{
                    object_id: ObjectState(object_id=object_id, kind="component")
                    for object_id in ("a", "b", "c", "d")
                },
                "edge_ab": ObjectState(
                    object_id="edge_ab",
                    kind="connector",
                    content={"source_id": "a", "target_id": "b"},
                ),
                "edge_cd": ObjectState(
                    object_id="edge_cd",
                    kind="connector",
                    content={"source_id": "c", "target_id": "d"},
                ),
            },
        )],
    )

    report = VisualQualityEvaluator().evaluate(
        "layout",
        layout,
        {"layout": layout, "document": document},
    )

    crossing = next(
        finding for finding in report.findings
        if finding.code == "connector_crossing"
    )
    assert crossing.object_ids == ["edge_ab", "edge_cd"]
    assert crossing.measured_value == 1


def test_repair_planner_maps_owner_dependencies_and_patch_scope() -> None:
    report = QualityReport(
        overall_score=0.5,
        scores={"layout": 0.5},
        findings=[QualityFinding(
            code="rendered_safe_area_clipped",
            severity=FindingSeverity.ERROR,
            artifact_id="frames",
            message="Ink touches the frame edge.",
            repair_target="layout",
            repair_scope="frame",
            beat_id="beat_2",
            object_ids=["node_2"],
            frame_numbers=[11, 12],
            measured_value=2,
            required_value=0,
            patch_paths=["/layout", "/beats/1"],
        )],
        decision=EvaluationDecision.REPAIR,
    )

    plan = QualityRepairPlanner().plan(report)

    assert plan.owner_stage is RepairStage.LAYOUT
    assert RepairStage.NARRATION not in plan.invalidated_stages
    assert RepairStage.RENDERER in plan.invalidated_stages
    assert plan.allowed_patch_paths == ["/layout"]
    assert plan.beat_ids == ["beat_2"]
    assert plan.frame_numbers == [11, 12]
    assert plan.fingerprint == QualityRepairPlanner().plan(report).fingerprint
