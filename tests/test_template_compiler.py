"""Tests for authoritative deterministic template compilation."""

from app.domain.generation import AudienceProfile
from app.domain.lesson import ConceptGraph, ConceptNode, LessonPlan
from app.domain.operations import OperationType
from app.domain.storyboard import VisualObjectSpec
from app.domain.strategy import TemplateMatch
from app.planning import PedagogyRouter, TemplateCompiler
from app.templates import TemplateRegistry
from app.templates.builtins import builtin_templates


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
