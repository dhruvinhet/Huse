"""Contract and state-transition tests for pipeline V2."""

import pytest
from pydantic import ValidationError

from app.domain.lesson import (
    ConceptEdge,
    ConceptGraph,
    ConceptNode,
    ConceptRelation,
)
from app.domain.operations import OperationType, VisualOperation
from app.domain.storyboard import Storyboard, VisualBeat, VisualObjectSpec
from app.domain.visual_document import ObjectLifecycle
from app.state import VisualStateTransitionEngine


def concept_graph() -> ConceptGraph:
    """Return a small valid teaching graph."""

    return ConceptGraph(
        objectives=["Explain the transformation"],
        nodes=[
            ConceptNode(
                concept_id="input",
                label="Input",
                definition="Incoming data",
                importance=0.8,
                prerequisites=[],
                teaching_order=0,
                visual_affordances=["pipeline"],
            ),
            ConceptNode(
                concept_id="output",
                label="Output",
                definition="Transformed data",
                importance=0.9,
                prerequisites=["input"],
                teaching_order=1,
                visual_affordances=["pipeline"],
            ),
        ],
        edges=[
            ConceptEdge(
                edge_id="input_to_output",
                source_id="input",
                target_id="output",
                relation=ConceptRelation.TRANSFORMS_TO,
            )
        ],
        teaching_sequence=["input", "output"],
    )


def storyboard() -> Storyboard:
    """Return a storyboard that creates and later emphasizes a hierarchy."""

    root = VisualObjectSpec(
        object_id="pipeline",
        kind="pipeline",
        semantic_role="process",
        content={"label": "Process", "layout": "horizontal"},
        accessibility_label="Processing pipeline",
        children=[
            VisualObjectSpec(
                object_id="input_box",
                kind="component",
                semantic_role="input",
                content={"label": "Input"},
                accessibility_label="Input stage",
            ),
            VisualObjectSpec(
                object_id="output_box",
                kind="component",
                semantic_role="output",
                content={"label": "Output"},
                accessibility_label="Output stage",
            ),
        ],
    )
    return Storyboard(
        document_id="document",
        title="Transformation",
        beats=[
            VisualBeat(
                beat_id="beat_1",
                section_id="section_1",
                concept_ids=["input"],
                teaching_intent="Introduce the pipeline",
                phrase_intent="Show the input and output stages",
                estimated_duration=2,
                operations=[
                    VisualOperation(
                        operation_id="create_pipeline",
                        operation=OperationType.CREATE,
                        target_ids=["pipeline"],
                        arguments={"objects": [root.model_dump(mode="json")]},
                    )
                ],
            ),
            VisualBeat(
                beat_id="beat_2",
                section_id="section_1",
                concept_ids=["output"],
                teaching_intent="Emphasize the result",
                phrase_intent="Highlight the output",
                estimated_duration=2,
                operations=[
                    VisualOperation(
                        operation_id="highlight_output",
                        operation=OperationType.HIGHLIGHT,
                        target_ids=["output_box"],
                    )
                ],
            ),
        ],
    )


def test_concept_graph_rejects_dangling_edges() -> None:
    """Edges may reference only declared concept nodes."""

    with pytest.raises(ValidationError, match="known nodes"):
        ConceptGraph(
            objectives=["Explain"],
            nodes=[concept_graph().nodes[0]],
            edges=[
                ConceptEdge(
                    edge_id="bad",
                    source_id="input",
                    target_id="missing",
                    relation=ConceptRelation.FLOWS_TO,
                )
            ],
            teaching_sequence=["input"],
        )


def test_concept_graph_normalizes_teaching_labels_to_ids() -> None:
    """Human-readable model labels are normalized to contract IDs."""

    graph = ConceptGraph(
        objectives=["Explain"],
        nodes=[
            ConceptNode(
                concept_id="root",
                label="Root Concept",
                definition="The starting concept",
                importance=1,
                teaching_order=0,
            ),
            ConceptNode(
                concept_id="leaf",
                label="Leaf Concept",
                definition="The dependent concept",
                importance=1,
                prerequisites=["root"],
                teaching_order=1,
            ),
        ],
        teaching_sequence=["Root Concept", "Leaf Concept"],
    )

    assert graph.teaching_sequence == ["root", "leaf"]


def test_concept_graph_repairs_unknown_and_missing_teaching_sequence_ids() -> None:
    """Model sequence noise is repaired without losing declared concepts."""

    graph = ConceptGraph.model_validate({
        "objectives": ["Explain"],
        "nodes": [
            {
                "concept_id": "root",
                "label": "Root",
                "definition": "The starting concept",
                "importance": 1,
                "teaching_order": 0,
            },
            {
                "concept_id": "middle",
                "label": "Middle",
                "definition": "The intermediate concept",
                "importance": 1,
                "prerequisites": ["root"],
                "teaching_order": 1,
            },
            {
                "concept_id": "leaf",
                "label": "Leaf",
                "definition": "The final concept",
                "importance": 1,
                "prerequisites": ["middle"],
                "teaching_order": 2,
            },
        ],
        "teaching_sequence": ["root", "unknown", "root", "Leaf"],
    })

    assert graph.teaching_sequence == ["root", "middle", "leaf"]


def test_concept_graph_drops_echoed_json_schema_metadata() -> None:
    """Schema definitions echoed by a model are not lesson data."""

    payload = concept_graph().model_dump(mode="json")
    payload["$defs"] = {"ConceptEdge": {"type": "object"}}

    graph = ConceptGraph.model_validate(payload)

    assert graph.nodes
    assert "$defs" not in graph.model_dump(mode="json")


def test_storyboard_allows_created_descendant_operations() -> None:
    """Stable child IDs remain available in later beats."""

    model = storyboard()
    assert model.beats[1].operations[0].target_ids == ["output_box"]


def test_hiding_a_container_hides_its_descendants() -> None:
    """A hidden hierarchy must not leave visible orphan child roots."""

    board = storyboard()
    hide = board.beats[1].model_copy(update={
        "operations": [VisualOperation(
            operation_id="hide_pipeline",
            operation=OperationType.HIDE,
            target_ids=["pipeline"],
        )],
    })
    document = VisualStateTransitionEngine().materialize(
        board.model_copy(update={"beats": [board.beats[0], hide]})
    )
    states = document.states[-1].object_states

    assert states["pipeline"].lifecycle is ObjectLifecycle.HIDDEN
    assert states["input_box"].lifecycle is ObjectLifecycle.HIDDEN
    assert states["output_box"].lifecycle is ObjectLifecycle.HIDDEN


def test_storyboard_drops_empty_model_generated_beats() -> None:
    """A prose-only beat does not block an otherwise valid storyboard."""

    payload = storyboard().model_dump(mode="json")
    payload["beats"].append(
        {
            "beat_id": "summary",
            "section_id": "section_1",
            "concept_ids": ["output"],
            "teaching_intent": "Summarize the lesson",
            "phrase_intent": "Summarize the lesson",
            "purpose": "summarize",
            "estimated_duration": 1,
            "operations": [],
        }
    )

    result = Storyboard.model_validate(payload)

    assert [beat.beat_id for beat in result.beats] == ["beat_1", "beat_2"]


def test_storyboard_normalizes_nested_initial_objects_and_missing_id() -> None:
    """Small models may place initial objects inside the first beat."""

    payload = storyboard().model_dump(mode="json")
    initial_objects = payload["beats"][0]["operations"][0]["arguments"][
        "objects"
    ]
    payload.pop("document_id")
    payload["beats"][0]["operations"][0] = {
        "operation_id": "show_pipeline",
        "operation": "highlight",
        "target_ids": ["pipeline"],
        "arguments": {},
    }
    payload["beats"][0]["initial_objects"] = initial_objects

    result = Storyboard.model_validate(payload)

    assert result.document_id == "storyboard"
    assert [item.object_id for item in result.initial_objects] == ["pipeline"]
    assert result.beats[0].operations


def test_storyboard_drops_duplicate_create_operations() -> None:
    """Repeated model creates do not invalidate the persistent document."""

    payload = storyboard().model_dump(mode="json")
    create_operation = payload["beats"][0]["operations"][0]
    payload["beats"][1]["operations"].append(
        {
            **create_operation,
            "operation_id": "duplicate_pipeline",
        }
    )

    result = Storyboard.model_validate(payload)

    assert [operation.operation_id for operation in result.beats[1].operations] == [
        "highlight_output"
    ]


def test_storyboard_drops_create_operations_without_objects() -> None:
    """Malformed create operations do not block valid visual beats."""

    payload = storyboard().model_dump(mode="json")
    payload["beats"].append(
        {
            "beat_id": "bad_create",
            "section_id": "section_1",
            "concept_ids": ["output"],
            "teaching_intent": "Show the result",
            "phrase_intent": "Show the result",
            "purpose": "demonstrate",
            "estimated_duration": 1,
            "operations": [
                {
                    "operation_id": "missing_objects",
                    "operation": "create",
                    "target_ids": ["bad_object"],
                    "arguments": {},
                }
            ],
        }
    )

    result = Storyboard.model_validate(payload)

    assert [beat.beat_id for beat in result.beats] == ["beat_1", "beat_2"]


def test_state_engine_preserves_objects_across_beats() -> None:
    """Later states contain earlier objects with changed presentation state."""

    document = VisualStateTransitionEngine().materialize(storyboard())
    assert len(document.states) == 2
    assert set(document.states[0].object_states) == {
        "pipeline",
        "input_box",
        "output_box",
    }
    assert document.states[1].object_states["input_box"].lifecycle is ObjectLifecycle.VISIBLE
    assert document.states[1].object_states["output_box"].lifecycle is ObjectLifecycle.EMPHASIZED
    assert document.states[1].parent_state_id == document.states[0].state_id
