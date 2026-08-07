"""Mandatory topology contracts for hierarchy-focused lessons."""

import pytest

from app.domain.generation import AudienceProfile
from app.domain.lesson import ConceptGraph, ConceptNode, LessonPlan
from app.domain.operations import OperationType
from app.planning import PedagogyRouter, TemplateCompiler
from app.quality.educational import EducationalQualityEvaluator
from app.state import VisualStateTransitionEngine
from app.templates import TemplateRegistry
from app.templates.builtins import builtin_templates


@pytest.mark.parametrize(
    "objective",
    [
        "Explain a binary tree with root, children, leaves, height, and traversal",
        "Teach binary-tree structure and visit its nodes in order",
        "Show how a binary tree branches and how traversal proceeds",
        "Introduce binary trees using a concrete connected example",
        "Demonstrate binary tree levels, subtrees, and an ordered traversal",
    ],
)
def test_binary_tree_prompts_compile_topology_and_ordered_traversal(
    objective: str,
) -> None:
    """Accepted binary-tree programs cannot collapse to unrelated cards."""

    labels = [
        "Root 8",
        "Left child 4",
        "Right child 12",
        "Left leaf 2",
        "Right leaf 6",
        "Height",
        "Traversal",
    ]
    nodes = [
        ConceptNode(
            concept_id=f"node_{index}",
            label=label,
            definition=f"{label} is part of the concrete binary-tree example.",
            importance=1,
            teaching_order=index,
            visual_affordances=["binary tree", "tree node"],
        )
        for index, label in enumerate(labels)
    ]
    lesson = LessonPlan(
        title="Binary Tree",
        summary="A binary tree connects each parent to at most two children.",
        concept_graph=ConceptGraph(
            objectives=[objective],
            nodes=nodes,
            teaching_sequence=[node.concept_id for node in nodes],
        ),
    )
    audience = AudienceProfile(learning_goal=objective)
    pedagogy = PedagogyRouter().route(lesson, audience)
    registry = TemplateRegistry(builtin_templates())

    program = TemplateCompiler().compile(
        lesson,
        [],
        registry.match(lesson.concept_graph, audience),
        registry,
        pedagogy,
        audience=audience,
    )

    assert program is not None
    created_roots = [
        root
        for beat in program.storyboard.beats
        for operation in beat.operations
        if operation.operation is OperationType.CREATE
        for root in operation.arguments["objects"]
    ]
    tree = next(root for root in created_roots if root["kind"] == "tree")
    tree_nodes = [item for item in tree["children"] if item["kind"] == "tree_node"]
    connectors = [item for item in tree["children"] if item["kind"] == "connector"]
    outgoing: dict[str, int] = {}
    for connector in connectors:
        source = connector["content"]["source_id"]
        outgoing[source] = outgoing.get(source, 0) + 1

    assert len(tree_nodes) >= 5
    assert len(connectors) == len(tree_nodes) - 1
    assert max(outgoing.values()) <= 2
    route_beat = next(
        beat
        for beat in program.storyboard.beats
        if any(action.action == "route" for action in beat.semantic_actions)
    )
    route = next(
        action for action in route_beat.semantic_actions if action.action == "route"
    )
    assert len([route.source_id, *route.path_ids, route.target_id]) >= 3

    document = VisualStateTransitionEngine().materialize(program.storyboard)
    report = EducationalQualityEvaluator().evaluate(
        "binary_tree_program",
        program.storyboard,
        {
            "storyboard": program.storyboard,
            "concept_graph": lesson.concept_graph,
            "pedagogy": pedagogy,
            "document": document,
        },
    )
    assert not {
        "binary_tree_topology_missing",
        "binary_tree_traversal_missing",
    }.intersection(item.code for item in report.findings)
    assert report.scores["topic_visual_contracts"] == 1.0
