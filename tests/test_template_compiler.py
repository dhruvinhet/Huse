"""Tests for authoritative deterministic template compilation."""

from app.domain.generation import AudienceProfile
from app.domain.lesson import (
    ConceptEdge,
    ConceptGraph,
    ConceptNode,
    ConceptRelation,
    LessonPlan,
)
from app.domain.operations import OperationType
from app.domain.operations import TraceAction
from app.domain.storyboard import VisualObjectSpec
from app.domain.strategy import TemplateCapabilities, TemplateMatch
from app.domain.visual_document import ObjectLifecycle
from app.planning import PedagogyRouter, TemplateCompiler
from app.templates import TemplateRegistry
from app.templates.builtins import builtin_templates
from app.state import VisualStateTransitionEngine


def _binary_search_lesson() -> LessonPlan:
    """Return a lesson that matches reviewed binary-search templates."""

    nodes = [
        ConceptNode(
            concept_id="sorted_array",
            label="Sorted Array",
            definition="Values are ordered before searching.",
            importance=1,
            teaching_order=0,
            visual_affordances=["array", "binary search"],
        ),
        ConceptNode(
            concept_id="midpoint",
            label="Midpoint",
            definition="Compare the target with the middle value.",
            importance=1,
            prerequisites=["sorted_array"],
            teaching_order=1,
            visual_affordances=["binary search"],
        ),
        ConceptNode(
            concept_id="half",
            label="Discard Half",
            definition="Remove the impossible half of the search range.",
            importance=1,
            prerequisites=["midpoint"],
            teaching_order=2,
            visual_affordances=["binary search"],
        ),
    ]
    return LessonPlan(
        title="Binary Search",
        summary="Search a sorted array by testing the midpoint.",
        concept_graph=ConceptGraph(
            objectives=["Execute binary search on a sorted array"],
            nodes=nodes,
            teaching_sequence=[node.concept_id for node in nodes],
        ),
    )


class PoisonedPrototypeLibrary:
    """Return a trusted instance that differs from advisory prompt data."""

    def __init__(self) -> None:
        self.parameters = None

    def instantiate(self, template_id, parameters):
        assert template_id == "trusted.v1"
        self.parameters = parameters
        return VisualObjectSpec(
            object_id=str(parameters["object_id"]),
            kind="trusted_diagram",
            semantic_role="reviewed_structure",
            content={"layout": "horizontal"},
            accessibility_label="Trusted reviewed structure",
            children=[
                VisualObjectSpec(
                    object_id="trusted_child",
                    kind="component",
                    semantic_role="reviewed_component",
                    content={"label": "Trusted"},
                    accessibility_label="Trusted component",
                )
            ],
        )


class SectionTemplateLibrary:
    """Create deterministic shot-scoped roots for multi-template tests."""

    def instantiate(self, template_id, parameters):
        root_id = str(parameters["object_id"])
        left_id = f"{root_id}_input"
        right_id = f"{root_id}_output"
        return VisualObjectSpec(
            object_id=root_id,
            kind="nested_group",
            semantic_role="reviewed_section",
            content={"operator": "flow", "layout": "horizontal"},
            accessibility_label=f"{template_id} reviewed section",
            children=[
                VisualObjectSpec(
                    object_id=left_id,
                    kind="component",
                    semantic_role="source",
                    content={"label": "Input"},
                    accessibility_label="Input",
                ),
                VisualObjectSpec(
                    object_id=right_id,
                    kind="component",
                    semantic_role="target",
                    content={"label": "Output"},
                    accessibility_label="Output",
                ),
                VisualObjectSpec(
                    object_id=f"{root_id}_connector",
                    kind="connector",
                    semantic_role="relation",
                    content={
                        "source_id": left_id,
                        "target_id": right_id,
                        "relation": "transforms_to",
                    },
                    accessibility_label="Input transforms to output",
                ),
            ],
        )


def test_compiler_ignores_advisory_prototype() -> None:
    """The registry instance, never TemplateMatch.prototype, is authoritative."""

    lesson = _binary_search_lesson()
    route = PedagogyRouter().route(
        lesson,
        AudienceProfile(learning_goal="Run the algorithm"),
    )
    poisoned = VisualObjectSpec(
        object_id="untrusted_prompt_object",
        kind="generic_box",
        semantic_role="advisory_only",
        content={"label": "Ignore me"},
        accessibility_label="Untrusted model context",
    )
    match = TemplateMatch(
        template_id="trusted.v1",
        concept_ids=["sorted_array"],
        parameters={
            "object_id": "attacker_controlled_id",
            "unknown_parameter": "must be removed",
        },
        prototype=poisoned,
        score=1,
        reason="Test match",
    )
    library = PoisonedPrototypeLibrary()

    program = TemplateCompiler().compile(
        lesson,
        [],
        [match],
        library,
        route,
    )

    assert program is not None
    created = program.storyboard.beats[0].operations[0].arguments["objects"][0]
    assert created["kind"] == "trusted_diagram"
    assert created["object_id"] != "untrusted_prompt_object"
    assert created["object_id"] != "attacker_controlled_id"
    assert "unknown_parameter" not in library.parameters


def test_builtin_match_compiles_complete_visual_program() -> None:
    """A reviewed match determines structure, transitions, and shot bounds."""

    lesson = _binary_search_lesson()
    audience = AudienceProfile(learning_goal="Run binary search")
    route = PedagogyRouter().route(lesson, audience)
    registry = TemplateRegistry(builtin_templates())
    matches = registry.match(lesson.concept_graph)

    program = TemplateCompiler().compile(
        lesson,
        [],
        matches,
        registry,
        route,
    )

    assert program is not None
    assert program.template_id == "binary_search.v1"
    assert len(program.storyboard.beats) == len(route.shots)
    assert program.storyboard.document_id.startswith("compiled_")
    operations = [
        operation.operation
        for beat in program.storyboard.beats
        for operation in beat.operations
    ]
    assert operations.count(OperationType.CREATE) == 1
    assert OperationType.SHOW in operations
    assert OperationType.DIM in operations
    assert OperationType.HIGHLIGHT in operations
    assert {
        concept_id
        for beat in program.storyboard.beats
        for concept_id in beat.concept_ids
    } == set(lesson.concept_graph.teaching_sequence)
    assert [beat.teaching_intent for beat in program.storyboard.beats] == [
        shot.visual_obligation for shot in route.shots
    ]
    assert program.match_confidence > 0
    assert program.capability_evidence
    assert program.parameter_provenance["concepts"].source == "extracted"
    assert isinstance(program.default_usage, list)


def test_required_template_parameter_cannot_use_generic_default() -> None:
    """A reviewed required operand fails clearly when extraction has no value."""

    lesson = _binary_search_lesson()
    route = PedagogyRouter().route(
        lesson,
        AudienceProfile(learning_goal="Run the algorithm"),
    )
    match = TemplateMatch(
        template_id="trusted.v1",
        concept_ids=["sorted_array"],
        score=1,
        reason="Requires a factual measurement",
        capabilities=TemplateCapabilities(required_parameters=["values"]),
    )

    import pytest
    with pytest.raises(ValueError, match="requires extracted parameter 'values'"):
        TemplateCompiler().compile(
            lesson,
            [],
            [match],
            PoisonedPrototypeLibrary(),
            route,
        )


def test_templates_can_be_owned_by_individual_shots() -> None:
    """Concept-specific matches compose without duplicate IDs or stale sections."""

    lesson = LessonPlan(
        title="Section-owned templates",
        summary="Input transforms to output.",
        concept_graph=ConceptGraph(
            objectives=["Explain the transformation"],
            nodes=[
                ConceptNode(
                    concept_id="input",
                    label="Input",
                    definition="The initial state.",
                    importance=1,
                    teaching_order=0,
                ),
                ConceptNode(
                    concept_id="output",
                    label="Output",
                    definition="The resulting state.",
                    importance=1,
                    teaching_order=1,
                    prerequisites=["input"],
                ),
            ],
            edges=[ConceptEdge(
                edge_id="transform",
                source_id="input",
                target_id="output",
                relation=ConceptRelation.TRANSFORMS_TO,
            )],
            teaching_sequence=["input", "output"],
        ),
    )
    route = PedagogyRouter().route(
        lesson,
        AudienceProfile(learning_goal="Understand the transformation"),
    )
    matches = [
        TemplateMatch(
            template_id="input_section.v1",
            concept_ids=["input"],
            score=0.91,
            reason="Input introduction",
        ),
        TemplateMatch(
            template_id="output_section.v1",
            concept_ids=["output"],
            score=0.95,
            reason="Output transformation",
        ),
    ]

    program = TemplateCompiler().compile(
        lesson,
        [],
        matches,
        SectionTemplateLibrary(),
        route,
    )

    assert program is not None
    assert set(program.template_ids) == {
        "input_section.v1",
        "output_section.v1",
    }
    assert len(program.shot_template_ids) == len(route.shots)
    created = [
        VisualObjectSpec.model_validate(root)
        for beat in program.storyboard.beats
        for operation in beat.operations
        if operation.operation is OperationType.CREATE
        for root in operation.arguments["objects"]
    ]
    object_ids = [item.object_id for root in created for item in root.flatten()]
    assert len(object_ids) == len(set(object_ids))
    document = VisualStateTransitionEngine().materialize(program.storyboard)
    assert (
        document.states[1].object_states[created[0].object_id].lifecycle
        is ObjectLifecycle.HIDDEN
    )
    for state in document.states:
        for item in state.object_states.values():
            if item.kind == "connector":
                assert item.content["source_id"] in state.object_states
                assert item.content["target_id"] in state.object_states


def test_compiler_bounds_oversized_semantic_graph_before_validation() -> None:
    """Large model graphs become a bounded, connected visual program."""

    nodes = [
        ConceptNode(
            concept_id=str(index),
            label=f"Concept {index}",
            definition=f"Definition {index}.",
            importance=(index % 5) / 4,
            teaching_order=index,
        )
        for index in range(15)
    ]
    edges = [
        ConceptEdge(
            edge_id=f"edge_{index}",
            source_id=str(source),
            target_id=str(target),
            relation=ConceptRelation.CAUSES,
            label="leads to",
        )
        for index, (source, target) in enumerate(
            (pair for pair in ((source, target) for source in range(15) for target in range(15)) if pair[0] != pair[1])
        )
        if index < 30
    ]
    lesson = LessonPlan(
        title="Oversized graph",
        summary="A graph larger than a single visual shot.",
        concept_graph=ConceptGraph(
            objectives=["Explain the graph"],
            nodes=nodes,
            edges=edges,
            teaching_sequence=[node.concept_id for node in nodes],
        ),
    )
    registry = TemplateRegistry(builtin_templates())
    pedagogy = PedagogyRouter().route(
        lesson,
        AudienceProfile(learning_goal="Understand the graph"),
    )

    program = TemplateCompiler().compile(
        lesson,
        [],
        [
            TemplateMatch(
                template_id="cause_effect.v1",
                concept_ids=["0", "1"],
                score=1,
                reason="Oversized graph regression",
            )
        ],
        registry,
        pedagogy,
    )

    assert program is not None
    assert len(program.parameters["concepts"]) == 12
    assert len(program.parameters["relations"]) <= 24
    selected_ids = {
        item["concept_id"] for item in program.parameters["concepts"]
    }
    assert {"0", "1"}.issubset(selected_ids)
    assert all(
        relation["source_id"] in selected_ids
        and relation["target_id"] in selected_ids
        for relation in program.parameters["relations"]
    )


def test_transformation_beats_use_reviewed_non_trace_action_recipes() -> None:
    """Matched templates must bind typed actions instead of generic traces."""

    lesson = _binary_search_lesson()
    route = PedagogyRouter().route(
        lesson,
        AudienceProfile(learning_goal="Run binary search"),
    )
    registry = TemplateRegistry(builtin_templates())
    program = TemplateCompiler().compile(
        lesson,
        [],
        registry.match(lesson.concept_graph),
        registry,
        route,
    )

    assert program is not None
    transformation_beats = [
        beat
        for beat in program.storyboard.beats
        if beat.purpose in {"transform", "demonstrate", "connect", "compare"}
    ]
    assert transformation_beats
    assert all(beat.semantic_actions for beat in transformation_beats)
    assert all(
        not isinstance(action, TraceAction)
        for beat in transformation_beats
        for action in beat.semantic_actions
    )
    VisualStateTransitionEngine().materialize(program.storyboard)
