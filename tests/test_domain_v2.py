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


def test_storyboard_allows_created_descendant_operations() -> None:
    """Stable child IDs remain available in later beats."""

    model = storyboard()
    assert model.beats[1].operations[0].target_ids == ["output_box"]


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
