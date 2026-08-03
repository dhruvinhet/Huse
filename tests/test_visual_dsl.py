"""Regression coverage for bounded VisualDSL operators and asset-aware layout."""

from app.domain.assets import AssetSource, ResolvedAssetSet, ResolvedSemanticAsset
from app.domain.layout import Viewport
from app.domain.visual_document import ObjectLifecycle, ObjectState, VisualDocument, VisualState
from app.domain.visual_intent import RendererOperator
from app.layout import HierarchicalLayoutEngine
from app.templates.operator_templates import (
    GenericOperatorParameters,
    SemanticOperatorCompiler,
    operator_template,
)
from app.templates import TemplateRegistry
from app.templates.builtins import builtin_templates


def _node(root, object_id: str):
    """Find one laid-out node in a persistent layout tree."""

    if root.object_id == object_id:
        return root
    for child in root.children:
        found = _node(child, object_id)
        if found is not None:
            return found
    return None


def test_visual_dsl_exposes_bounded_compositional_operators() -> None:
    """Broad diagram families compile to typed structures, not arbitrary trees."""

    requested = {
        "icon", "group", "flow", "timeline", "cycle", "graph", "tree",
        "plot", "equation", "table", "matrix", "code_trace", "circuit",
        "molecule", "map", "anatomy", "callout", "transform",
    }
    exposed = {operator.value for operator in RendererOperator}
    assert requested.issubset(exposed)

    parameters = GenericOperatorParameters(
        object_id="dsl",
        label="Signal flow",
        operands=["Input", "State", "Output"],
    )
    root = SemanticOperatorCompiler().compile(RendererOperator.FLOW, parameters)
    assert root.kind == "flow"
    assert root.content["operator"] == "flow"
    assert root.content["dsl_version"] == "1.0"
    assert {child.kind for child in root.children} >= {"component", "connector"}


def test_layout_uses_resolved_asset_aspect_ratio_and_focal_weight() -> None:
    """Wide and square assets receive different readable component geometry."""

    states = [
        VisualState(
            state_id="wide_state",
            beat_id="wide",
            object_states={
                "wide": ObjectState(
                    object_id="wide",
                    kind="component",
                    lifecycle=ObjectLifecycle.EMPHASIZED,
                    content={"label": "Wide illustration", "detail": "Evidence"},
                    metadata={"importance": 1.0, "focal_weight": 1.0},
                )
            },
        ),
        VisualState(
            state_id="square_state",
            beat_id="square",
            object_states={
                "square": ObjectState(
                    object_id="square",
                    kind="component",
                    content={"label": "Square illustration", "detail": "Evidence"},
                    metadata={"importance": 0.2, "focal_weight": 0.2},
                )
            },
        ),
    ]
    document = VisualDocument(document_id="asset_layout", states=states)
    assets = ResolvedAssetSet(assets=[
        ResolvedSemanticAsset(
            asset_id="asset_wide",
            query_digest="wide_digest",
            source=AssetSource.GENERATED,
            path="wide.svg",
            mime_type="image/svg+xml",
            content_hash="wide_hash",
            editable=True,
            ready=True,
            intrinsic_width=320,
            intrinsic_height=80,
            aspect_ratio=4.0,
        ),
        ResolvedSemanticAsset(
            asset_id="asset_square",
            query_digest="square_digest",
            source=AssetSource.GENERATED,
            path="square.svg",
            mime_type="image/svg+xml",
            content_hash="square_hash",
            editable=True,
            ready=True,
            intrinsic_width=100,
            intrinsic_height=100,
            aspect_ratio=1.0,
        ),
    ])

    plan = HierarchicalLayoutEngine().layout(
        document,
        assets,
        Viewport(width=1280, height=720, margin=40),
    )
    wide = _node(plan.state_roots["wide_state"], "wide")
    square = _node(plan.state_roots["square_state"], "square")
    assert wide is not None and square is not None
    assert wide.box.width > square.box.width
    assert wide.box.height >= square.box.height


def test_directed_educational_templates_route_to_bounded_flow_dsl() -> None:
    """Cause/effect and process matches do not regress to generic card trees."""

    template = operator_template(
        "cause_effect.v1",
        {"cause", "effect"},
        RendererOperator.CAUSE_EFFECT,
        "Cause and Effect",
        ("cause", "mechanism", "effect"),
    )
    compiled = template.instantiate(
        {
            "object_id": "cause_effect",
            "label": "Earth evolution",
            "concepts": [
                {
                    "concept_id": "c1",
                    "label": "Cause",
                    "definition": "A starting condition",
                    "importance": 0.8,
                    "order": 0,
                },
                {
                    "concept_id": "c2",
                    "label": "Effect",
                    "definition": "A resulting condition",
                    "importance": 0.7,
                    "order": 1,
                },
            ],
            "relations": [
                {
                    "source_id": "c1",
                    "target_id": "c2",
                    "relation": "causes",
                    "label": "drives",
                }
            ],
        }
    )
    assert compiled.kind == "flow"
    assert compiled.content["operator"] == "flow"


def test_every_builtin_family_has_a_bounded_operator_root() -> None:
    """All reviewed families compile to a declared operator, not a card recipe."""

    templates = builtin_templates()
    registry = TemplateRegistry(templates)
    for template in templates:
        root = registry.instantiate(
            template.template_id,
            {"object_id": "family", "label": template.template_id},
        )
        assert root.content["dsl_version"] == "1.0"
        assert root.content["operator"]
        assert root.content["operator"] != "semantic_structure"


def test_concept_rich_educational_families_keep_distinct_root_operators() -> None:
    """Layered, funnel, Venn, and chart families retain their visual grammar."""

    concepts = [
        {
            "concept_id": f"c{index}",
            "label": label,
            "definition": f"Definition for {label}",
            "importance": 0.8,
            "order": index,
        }
        for index, label in enumerate(("Input", "Middle", "Output"))
    ]
    registry = TemplateRegistry(builtin_templates())
    roots = {
        template_id: registry.instantiate(
            template_id,
            {
                "object_id": template_id.replace(".", "_"),
                "label": template_id,
                "concepts": concepts,
            },
        )
        for template_id in (
            "layered_architecture.v1",
            "funnel.v1",
            "venn.v1",
            "timeline.v1",
        )
    }
    assert {root.content["operator"] for root in roots.values()} == {
        "layered",
        "funnel",
        "venn",
        "timeline",
    }
    assert all(root.kind != "nested_group" for root in roots.values())


def test_typed_relations_survive_cause_effect_compilation() -> None:
    """Non-sequential causal edges become grounded, typed connectors."""

    template = operator_template(
        "cause_effect.v1",
        {"cause", "effect"},
        RendererOperator.CAUSE_EFFECT,
        "Cause and Effect",
        ("cause", "mechanism", "effect"),
    )
    root = template.instantiate(
        {
            "object_id": "photosynthesis",
            "label": "Photosynthesis",
            "concepts": [
                {
                    "concept_id": "c1",
                    "label": "Light",
                    "definition": "Energy input",
                    "importance": 0.8,
                    "order": 0,
                },
                {
                    "concept_id": "c2",
                    "label": "Reactions",
                    "definition": "Chemical steps",
                    "importance": 0.8,
                    "order": 1,
                },
                {
                    "concept_id": "c3",
                    "label": "Sugar",
                    "definition": "Stored output",
                    "importance": 0.8,
                    "order": 2,
                },
            ],
            "relations": [
                {
                    "source_id": "c1",
                    "target_id": "c3",
                    "relation": "causes",
                    "label": "drives",
                },
                {
                    "source_id": "c2",
                    "target_id": "c3",
                    "relation": "flows_to",
                    "label": "produces",
                },
            ],
        }
    )
    connectors = [item for item in root.flatten() if item.kind == "connector"]
    assert {
        (item.concept_ids[0], item.concept_ids[1], item.content["relation"])
        for item in connectors
    } == {("c1", "c3", "causes"), ("c2", "c3", "flows_to")}
