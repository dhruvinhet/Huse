"""Regression tests for provider-independent concept-graph storyboards."""

import pytest

from app.domain.generation import AudienceProfile
from app.domain.lesson import (
    ConceptEdge,
    ConceptGraph,
    ConceptNode,
    ConceptRelation,
    LessonPlan,
)
from app.domain.operations import OperationType, VisualOperation
from app.domain.storyboard import Storyboard, VisualBeat, VisualObjectSpec
from app.planning import ConceptGraphStoryboardBuilder
from app.quality import DeterministicQualityEvaluator


def _lesson(
    title: str,
    labels: list[str],
    relations: list[tuple[int, int]],
    affordance: str,
) -> LessonPlan:
    nodes = [
        ConceptNode(
            concept_id=f"c{index}",
            label=label,
            definition=f"Explain {label} in {title}.",
            importance=1.0,
            teaching_order=index,
            visual_affordances=[affordance],
        )
        for index, label in enumerate(labels, start=1)
    ]
    return LessonPlan(
        title=title,
        summary=f"A lesson about {title}",
        concept_graph=ConceptGraph(
            objectives=[f"Understand {title}"],
            nodes=nodes,
            edges=[
                ConceptEdge(
                    edge_id=f"e{index}",
                    source_id=f"c{source}",
                    target_id=f"c{target}",
                    relation=ConceptRelation.FLOWS_TO,
                )
                for index, (source, target) in enumerate(relations, start=1)
            ],
            teaching_sequence=[node.concept_id for node in nodes],
        ),
    )


TOPICS = [
    _lesson(
        "Binary Search",
        [
            "Sorted List",
            "Divide and Conquer",
            "Comparison",
            "Halve Search Space",
            "Binary Search Algorithm",
        ],
        [(1, 2), (2, 3), (3, 4), (4, 5)],
        "array",
    ),
    _lesson(
        "Heap",
        [
            "Heap",
            "Min Heap",
            "Max Heap",
            "Heapify",
            "Insert",
            "Extract",
            "Applications",
        ],
        [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6)],
        "tree",
    ),
    _lesson(
        "REST API Request",
        ["Client", "Request", "Server", "Response"],
        [(1, 2), (2, 3), (3, 4)],
        "pipeline",
    ),
]


@pytest.mark.parametrize("lesson", TOPICS, ids=lambda item: item.title)
def test_fallback_covers_every_concept_with_valid_lifecycle(
    lesson: LessonPlan,
) -> None:
    """Unrelated subjects compile from the same graph-driven algorithm."""

    board = ConceptGraphStoryboardBuilder().build(lesson)
    represented = {
        concept_id
        for beat in board.beats
        for concept_id in beat.concept_ids
    }
    report = DeterministicQualityEvaluator().evaluate(
        "storyboard",
        board,
        {
            "concept_graph": lesson.concept_graph,
            "audience": AudienceProfile(learning_goal="Learn the topic"),
        },
    )

    assert represented.issuperset(lesson.concept_graph.teaching_sequence)
    assert report.scores["semantic_coverage"] == 1
    assert report.decision.value == "pass"
    assert all(beat.camera_intent.operation == "hold" for beat in board.beats)


def test_ground_replaces_schema_valid_but_incomplete_storyboard() -> None:
    """A model cannot pass by mentioning only the umbrella concept."""

    lesson = TOPICS[0]
    model_object = VisualObjectSpec(
        object_id="algorithm",
        kind="component",
        semantic_role="concept",
        content={"label": "Binary Search Algorithm"},
        accessibility_label="Binary Search Algorithm",
    )
    incomplete = Storyboard(
        document_id="model_storyboard",
        title=lesson.title,
        beats=[
            VisualBeat(
                beat_id="model_beat",
                section_id="model",
                concept_ids=["c5"],
                teaching_intent="Explain Binary Search Algorithm",
                phrase_intent="Show the binary search algorithm at a high level.",
                estimated_duration=5,
                operations=[
                    VisualOperation(
                        operation_id="create_algorithm",
                        operation=OperationType.CREATE,
                        target_ids=["algorithm"],
                        arguments={
                            "objects": [model_object.model_dump(mode="json")]
                        },
                    )
                ],
            )
        ],
    )

    grounded = ConceptGraphStoryboardBuilder().ground(incomplete, lesson)

    assert grounded.document_id == incomplete.document_id
    assert {
        concept_id
        for beat in grounded.beats
        for concept_id in beat.concept_ids
    }.issuperset(lesson.concept_graph.teaching_sequence)
    assert len(grounded.beats) == len(lesson.concept_graph.nodes) + 1
