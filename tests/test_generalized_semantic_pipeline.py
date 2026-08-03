"""Cross-topic regressions for relation-aware visual compilation."""

import pytest

from app.agents.lesson_planner import GeminiLessonPlanner
from app.domain.generation import AudienceProfile, GenerationRequest
from app.domain.lesson import (
    ConceptEdge,
    ConceptGraph,
    ConceptNode,
    ConceptRelation,
    LessonPlan,
)
from app.domain.operations import OperationType, VisualOperation
from app.domain.assets import ResolvedAssetSet
from app.domain.layout import Viewport
from app.domain.quality import EvaluationDecision
from app.domain.storyboard import Storyboard, VisualBeat, VisualObjectSpec
from app.domain.strategy import TemplateMatch
from app.planning import (
    PedagogyRouter,
    SemanticAssetQueryPlanner,
    TemplateCompiler,
)
from app.planning.graph_semantics import normalize_lesson_structure
from app.layout import HierarchicalLayoutEngine
from app.quality import EducationalQualityEvaluator, VisualQualityEvaluator
from app.state import VisualStateTransitionEngine
from app.templates import TemplateRegistry, builtin_templates


def _mechanism_lesson() -> LessonPlan:
    nodes = [
        ConceptNode(
            concept_id="sensor",
            label="Sensor",
            definition="Measures the current state.",
            importance=1,
            teaching_order=0,
            visual_affordances=["signal", "graph node"],
        ),
        ConceptNode(
            concept_id="decision",
            label="Decision Logic",
            definition="Compares the measured state with the goal.",
            importance=1,
            prerequisites=["sensor"],
            teaching_order=1,
            visual_affordances=["logic", "component"],
        ),
        ConceptNode(
            concept_id="actuator",
            label="Actuator",
            definition="Changes the system using the decision.",
            importance=1,
            prerequisites=["decision"],
            teaching_order=2,
            visual_affordances=["component"],
        ),
        ConceptNode(
            concept_id="controller",
            label="Controller",
            definition="Contains the parts that measure, decide, and act.",
            importance=1,
            teaching_order=3,
            visual_affordances=["system", "hierarchy"],
        ),
    ]
    edges = [
        ConceptEdge(
            edge_id="sensor_part",
            source_id="sensor",
            target_id="controller",
            relation=ConceptRelation.PART_OF,
        ),
        ConceptEdge(
            edge_id="decision_part",
            source_id="decision",
            target_id="controller",
            relation=ConceptRelation.PART_OF,
        ),
        ConceptEdge(
            edge_id="actuator_part",
            source_id="actuator",
            target_id="controller",
            relation=ConceptRelation.PART_OF,
        ),
        ConceptEdge(
            edge_id="measure_flow",
            source_id="sensor",
            target_id="decision",
            relation=ConceptRelation.FLOWS_TO,
            label="measurement flows to",
        ),
        ConceptEdge(
            edge_id="command_flow",
            source_id="decision",
            target_id="actuator",
            relation=ConceptRelation.FLOWS_TO,
            label="command flows to",
        ),
    ]
    return LessonPlan(
        title="How a feedback controller works",
        summary="A controller measures state, decides, and acts.",
        concept_graph=ConceptGraph(
            objectives=["Explain the controller's measurement-to-action mechanism"],
            nodes=nodes,
            edges=edges,
            teaching_sequence=["sensor", "decision", "actuator", "controller"],
        ),
    )


def test_relation_compiler_nests_parts_and_only_connects_declared_flows() -> None:
    """New topics compile from typed semantics, not guessed label order."""

    lesson = _mechanism_lesson()
    audience = AudienceProfile(learning_goal="Understand how it works")
    pedagogy = PedagogyRouter().route(lesson, audience)
    registry = TemplateRegistry(builtin_templates())
    program = TemplateCompiler().compile(
        lesson,
        [],
        [
            TemplateMatch(
                template_id="system_design.v1",
                concept_ids=list(lesson.concept_graph.teaching_sequence),
                score=1,
                reason="Cross-topic structural regression",
            )
        ],
        registry,
        pedagogy,
    )

    assert program is not None
    root = VisualObjectSpec.model_validate(
        program.storyboard.beats[0].operations[0].arguments["objects"][0]
    )
    assert root.content["operator"] == "semantic_structure"
    objects = {item.object_id: item for item in root.flatten()}
    controller = next(
        item for item in objects.values() if item.concept_ids == ["controller"]
    )
    assert {item.concept_ids[0] for item in controller.children} == {
        "sensor", "decision", "actuator"
    }
    connectors = [item for item in root.flatten() if item.kind == "connector"]
    assert len(connectors) == 2
    assert {item.content["relation"] for item in connectors} == {"flows_to"}
    assert all(
        operation.operation in {
            OperationType.CREATE,
            OperationType.SHOW,
            OperationType.DIM,
            OperationType.HIGHLIGHT,
        }
        for beat in program.storyboard.beats
        for operation in beat.operations
    )
    mechanism_beat = next(
        beat for beat in program.storyboard.beats if beat.purpose == "transform"
    )
    assert mechanism_beat.concept_ids == list(
        lesson.concept_graph.teaching_sequence[1:]
    )
    assert not any(
        sentence.strip().casefold().startswith(("explain ", "describe "))
        for beat in program.storyboard.beats
        for sentence in beat.phrase_intent.split(".")
    )
    report = EducationalQualityEvaluator().evaluate(
        "storyboard",
        program.storyboard,
        {
            "concept_graph": lesson.concept_graph,
            "audience": audience,
            "pedagogy": pedagogy,
        },
    )
    assert report.decision is EvaluationDecision.PASS
    assert not {
        "concept_object_mismatch",
        "semantic_relation_missing",
        "visual_progression_highlight_only",
    }.intersection(item.code for item in report.findings)
    document = VisualStateTransitionEngine().materialize(program.storyboard)
    layout = HierarchicalLayoutEngine().layout(
        document,
        ResolvedAssetSet(),
        Viewport(width=1920, height=1080),
    )
    visual_report = VisualQualityEvaluator().evaluate(
        "layout", layout, {"layout": layout}
    )
    assert visual_report.decision is EvaluationDecision.PASS


def test_quality_gate_rejects_flat_containment_and_highlight_only_shots() -> None:
    """A structurally valid but semantically false diagram cannot score a pass."""

    lesson = _mechanism_lesson()
    children = [
        VisualObjectSpec(
            object_id=concept_id,
            kind="component",
            semantic_role="concept",
            concept_ids=[concept_id],
            content={"label": label},
            accessibility_label=label,
        )
        for concept_id, label in (
            ("sensor", "Sensor"),
            ("decision", "Decision Logic"),
            ("actuator", "Actuator"),
            ("controller", "Controller"),
        )
    ]
    root = VisualObjectSpec(
        object_id="flat",
        kind="pipeline",
        semantic_role="incorrect_flat_chain",
        concept_ids=list(lesson.concept_graph.teaching_sequence),
        content={"label": lesson.title, "layout": "horizontal"},
        children=children,
        accessibility_label="Incorrect flat mechanism",
    )
    beats = []
    for index in range(4):
        operations = [
            VisualOperation(
                operation_id="create" if index == 0 else f"highlight_{index}",
                operation=(
                    OperationType.CREATE
                    if index == 0
                    else OperationType.HIGHLIGHT
                ),
                target_ids=["flat"] if index == 0 else [children[index].object_id],
                arguments=(
                    {"objects": [root.model_dump(mode="json")]}
                    if index == 0 else {}
                ),
            )
        ]
        beats.append(VisualBeat(
            beat_id=f"beat_{index}",
            section_id="mechanism",
            concept_ids=[lesson.concept_graph.teaching_sequence[index]],
            teaching_intent="Explain one step",
            phrase_intent="Explain the visible step clearly.",
            purpose="introduce" if index == 0 else "transform",
            estimated_duration=2,
            operations=operations,
        ))
    board = Storyboard(document_id="bad", title=lesson.title, beats=beats)
    pedagogy = PedagogyRouter().route(
        lesson,
        AudienceProfile(learning_goal="Understand how it works"),
    )

    report = EducationalQualityEvaluator().evaluate(
        "storyboard",
        board,
        {"concept_graph": lesson.concept_graph, "pedagogy": pedagogy},
    )

    assert report.decision is EvaluationDecision.REPAIR
    codes = {item.code for item in report.findings}
    assert "semantic_relation_missing" in codes
    assert "visual_progression_highlight_only" in codes


def test_mechanism_validation_rejects_a_parts_only_graph() -> None:
    """How-it-works requests require behavior, not merely named components."""

    lesson = _mechanism_lesson()
    parts_only = lesson.model_copy(update={
        "concept_graph": lesson.concept_graph.model_copy(update={
            "edges": [
                edge
                for edge in lesson.concept_graph.edges
                if edge.relation is ConceptRelation.PART_OF
            ]
        })
    })
    request = GenerationRequest(
        run_id="parts_only",
        topic="How does this controller work?",
        audience=AudienceProfile(learning_goal="Understand how it works"),
    )

    with pytest.raises(ValueError, match="containment-only"):
        GeminiLessonPlanner._validate_semantics(request, parts_only)


def test_router_distinguishes_mechanism_from_static_anatomy() -> None:
    """General question intent wins over incidental architecture vocabulary."""

    mechanism = _mechanism_lesson()
    audience = AudienceProfile(learning_goal="Understand the system")
    assert PedagogyRouter().route(mechanism, audience).mode.value == "mechanism_first"

    anatomy = mechanism.model_copy(update={
        "title": "Controller structure and parts",
        "concept_graph": mechanism.concept_graph.model_copy(update={
            "edges": [
                edge
                for edge in mechanism.concept_graph.edges
                if edge.relation is ConceptRelation.PART_OF
            ]
        }),
    })
    assert PedagogyRouter().route(anatomy, audience).mode.value == "spatial_anatomy"


def test_peer_collection_cannot_be_compiled_as_a_false_pipeline() -> None:
    """Numbered peers with vague model edges remain independent concepts."""

    nodes = [
        ConceptNode(
            concept_id=f"rule_{index}",
            label=f"Operating Rule {ordinal}",
            definition=f"A distinct condition handled by rule {index}.",
            importance=1,
            prerequisites=[] if index == 1 else [f"rule_{index - 1}"],
            teaching_order=index - 1,
            visual_affordances=["principle", "condition"],
        )
        for index, ordinal in enumerate(
            ("First", "Second", "Third"),
            start=1,
        )
    ]
    nodes.append(ConceptNode(
        concept_id="support",
        label="Supporting Interaction",
        definition="An application that helps demonstrate the collection.",
        importance=0.8,
        prerequisites=["rule_1"],
        teaching_order=3,
        visual_affordances=["interaction"],
    ))
    lesson = LessonPlan(
        title="Three operating rules",
        summary="Three co-equal rules describe distinct conditions.",
        concept_graph=ConceptGraph(
            objectives=["Explain three operating rules"],
            nodes=nodes,
            edges=[
                ConceptEdge(
                    edge_id="bad_dependency",
                    source_id="rule_1",
                    target_id="rule_2",
                    relation=ConceptRelation.DEPENDS_ON,
                    label="involves",
                ),
                ConceptEdge(
                    edge_id="bad_flow",
                    source_id="rule_3",
                    target_id="rule_2",
                    relation=ConceptRelation.FLOWS_TO,
                    label="requires",
                ),
                ConceptEdge(
                    edge_id="bad_support_cause",
                    source_id="support",
                    target_id="rule_3",
                    relation=ConceptRelation.CAUSES,
                    label="produces an outcome",
                ),
            ],
            teaching_sequence=["rule_1", "rule_2", "rule_3", "support"],
        ),
    )
    normalized = normalize_lesson_structure(lesson)
    audience = AudienceProfile(learning_goal="Understand all three rules")
    pedagogy = PedagogyRouter().route(normalized, audience)
    registry = TemplateRegistry(builtin_templates())
    matches = registry.match(normalized.concept_graph, audience)

    assert normalized.concept_graph.edges == []
    assert all(not node.prerequisites for node in normalized.concept_graph.nodes)
    assert pedagogy.mode.value == "concept_set"
    assert matches[0].template_id == "concept_set.v1"
    assert "pipeline.v1" not in {match.template_id for match in matches}

    program = TemplateCompiler().compile(
        normalized,
        [],
        matches,
        registry,
        pedagogy,
    )
    assert program is not None
    board = SemanticAssetQueryPlanner().enrich(
        program.storyboard,
        normalized,
    )
    created = VisualObjectSpec.model_validate(
        board.beats[0].operations[0].arguments["objects"][0]
    )
    queried = [item for item in created.flatten() if item.asset_query is not None]
    assert len(queried) == 4
    focused = [
        tuple(operation.target_ids)
        for beat in board.beats
        for operation in beat.operations
        if operation.operation is OperationType.HIGHLIGHT
    ]
    assert len(set(focused)) == len(focused)
