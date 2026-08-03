"""Regression tests for procedural topic-template operators."""

import ast
from pathlib import Path

from app.templates import TemplateRegistry
from app.templates.builtins import builtin_templates


def test_binary_search_operator_has_algorithm_state_not_components() -> None:
    """Binary search compiles cells, pointers, comparison, and code state."""

    registry = TemplateRegistry(builtin_templates())
    root = registry.instantiate(
        "binary_search.v1",
        {
            "object_id": "binary",
            "label": "Binary Search",
            "values": [2, 5, 8, 12, 16],
            "target": 12,
            "low": 1,
            "high": 4,
            "mid": 2,
            "comparison": "8 is less than 12",
            "active_code_line": "low = mid + 1",
        },
    )

    assert root.kind == "array"
    assert root.content["operator"] == "binary_search"
    assert root.content["low"] == 1
    assert root.content["high"] == 4
    assert root.content["mid"] == 2
    assert root.content["target"] == 12
    assert root.content["active_code_line"] == "low = mid + 1"
    assert {child.kind for child in root.children} == {"array_cell"}
    assert [child.content["label"] for child in root.children] == [
        "2", "5", "8", "12", "16"
    ]


def test_topic_operator_families_compile_different_structures() -> None:
    """CNN, protocol, traversal, and memory templates are not renamed chains."""

    registry = TemplateRegistry(builtin_templates())
    roots = {
        template_id: registry.instantiate(
            template_id,
            {"object_id": template_id.replace(".", "_"), "label": template_id},
        )
        for template_id in (
            "cnn.v1",
            "tcp_handshake.v1",
            "bfs.v1",
            "memory_allocation.v1",
        )
    }

    signatures = {
        (
            root.kind,
            root.content["operator"],
            tuple(child.semantic_role for child in root.children),
        )
        for root in roots.values()
    }
    assert len(signatures) == len(roots)
    assert roots["cnn.v1"].kind == "neural_network"
    assert roots["tcp_handshake.v1"].content["participants"] == [
        "Client", "Server"
    ]
    assert "frontier" in roots["bfs.v1"].content
    assert all(
        "address" in child.content
        for child in roots["memory_allocation.v1"].children
    )


def test_topic_operators_expose_validated_parameter_schemas() -> None:
    """Reviewed topic families declare operands instead of arbitrary dictionaries."""

    registry = TemplateRegistry(builtin_templates())

    binary = registry.parameter_schema("binary_search.v1")
    protocol = registry.parameter_schema("rest_api.v1")

    assert {"values", "target", "low", "high", "mid", "active_code_line"}.issubset(
        binary["properties"]
    )
    assert {"participants", "messages", "active_message"}.issubset(
        protocol["properties"]
    )
    assert binary["additionalProperties"] is False


def test_operator_operands_replace_fixed_recipe_labels() -> None:
    """Lesson-derived operands parameterize the procedural topic structure."""

    registry = TemplateRegistry(builtin_templates())
    root = registry.instantiate(
        "rest_api.v1",
        {
            "object_id": "rest",
            "label": "REST request",
            "components": [
                "Browser sends GET /users",
                "Router selects handler",
                "Database returns rows",
                "Server serializes JSON",
            ],
        },
    )

    assert [child.content["label"] for child in root.children] == [
        "Browser sends GET /users",
        "Router selects handler",
        "Database returns rows",
        "Server serializes JSON",
    ]


def test_process_concepts_use_readable_stage_cards_not_affordance_nodes() -> None:
    """Affordance words cannot turn long process concepts into tiny node glyphs."""

    root = TemplateRegistry(builtin_templates()).instantiate(
        "pipeline.v1",
        {
            "object_id": "vision",
            "label": "Vision overview",
            "stages": ["Pixels", "Features", "Recognition"],
            "concepts": [
                {
                    "concept_id": "pixels",
                    "label": "Pixels",
                    "definition": "Discrete measurements of light and color.",
                    "importance": 1,
                    "order": 0,
                    "visual_affordances": ["matrix"],
                },
                {
                    "concept_id": "features",
                    "label": "Features",
                    "definition": "Patterns learned or extracted from pixels.",
                    "importance": 1,
                    "order": 1,
                    "visual_affordances": ["graph"],
                },
                {
                    "concept_id": "recognition",
                    "label": "Recognition",
                    "definition": "Semantic predictions produced from features.",
                    "importance": 1,
                    "order": 2,
                    "visual_affordances": ["tree"],
                },
            ],
            "relations": [
                {
                    "source_id": "pixels",
                    "target_id": "features",
                    "relation": "transforms_to",
                    "label": "supports feature learning",
                },
                {
                    "source_id": "features",
                    "target_id": "recognition",
                    "relation": "flows_to",
                    "label": "supports prediction",
                },
            ],
        },
    )

    concepts = [child for child in root.children if child.kind != "connector"]
    assert {child.kind for child in concepts} == {"component"}
    assert all(child.content.get("detail") for child in concepts)


def test_fallback_and_compatibility_planners_do_not_author_scene_objects() -> None:
    """Only deterministic compiler modules may construct low-level scene trees."""

    root = Path(__file__).parents[1]
    for relative in (
        "app/planning/storyboard_fallback.py",
        "app/planning/template_storyboard.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        constructed = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "VisualObjectSpec" not in constructed
        assert "VisualOperation" not in constructed
