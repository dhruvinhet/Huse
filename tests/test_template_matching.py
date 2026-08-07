"""Tests for relation-aware local BM25 template matching."""

from app.domain.generation import AudienceLevel, AudienceProfile
from app.domain.lesson import (
    ConceptEdge,
    ConceptGraph,
    ConceptNode,
    ConceptRelation,
)
from app.domain.storyboard import VisualObjectSpec
from app.domain.strategy import TemplateCapabilities
from app.domain.pedagogy import PedagogyMode, PedagogyPlan, PedagogyShot
from app.templates import TemplateRegistry
from app.templates.builtins import builtin_templates


class CapabilityTemplate:
    """Small reviewed template fixture with explicit machine capabilities."""

    def __init__(
        self,
        template_id: str,
        keywords: set[str],
        capabilities: TemplateCapabilities,
    ) -> None:
        self.template_id = template_id
        self.keywords = frozenset(keywords)
        self.capabilities = capabilities

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        return VisualObjectSpec(
            object_id=str(parameters["object_id"]),
            kind="component",
            semantic_role="fixture",
            accessibility_label="Capability fixture",
        )


def _graph(
    objective: str,
    labels: list[str],
    relation: ConceptRelation | None = None,
) -> ConceptGraph:
    nodes = [
        ConceptNode(
            concept_id=f"concept_{index}",
            label=label,
            definition=f"Explain {label} in this lesson.",
            importance=1,
            teaching_order=index,
        )
        for index, label in enumerate(labels)
    ]
    edges = []
    if relation is not None:
        edges.append(
            ConceptEdge(
                edge_id="structural_relation",
                source_id="concept_0",
                target_id="concept_1",
                relation=relation,
            )
        )
    return ConceptGraph(
        objectives=[objective],
        nodes=nodes,
        edges=edges,
        teaching_sequence=[node.concept_id for node in nodes],
    )


def test_bm25_understands_synonyms_for_binary_search() -> None:
    """Conceptual wording can select a topic template without exact tokens."""

    registry = TemplateRegistry(builtin_templates())
    graph = _graph(
        "Locate a target by repeatedly halving an ordered sequence",
        ["Lookup interval", "Middle candidate", "Remaining half"],
    )

    matches = registry.match(
        graph,
        AudienceProfile(
            level=AudienceLevel.BEGINNER,
            learning_goal="Find values efficiently",
        ),
    )

    assert matches[0].template_id == "binary_search.v1"
    assert "bm25=" in matches[0].reason
    assert matches[0].prototype is None


def test_relation_stage_selects_comparison_without_keyword_overlap() -> None:
    """A contrasts-with edge supplies representation semantics before BM25."""

    registry = TemplateRegistry(builtin_templates())
    graph = _graph(
        "Choose the appropriate alternative",
        ["Approach Alpha", "Approach Beta"],
        ConceptRelation.CONTRASTS_WITH,
    )

    matches = registry.match(
        graph,
        AudienceProfile(learning_goal="Make a justified choice"),
    )

    assert matches
    assert matches[0].template_id == "comparison.v1"
    assert matches[0].concept_ids == ["concept_0", "concept_1"]
    assert "contrasts_with" in matches[0].reason


def test_relation_stage_selects_cause_effect_for_causal_operands() -> None:
    """Required source and outcome operands favor a causal visual grammar."""

    registry = TemplateRegistry(builtin_templates())
    graph = _graph(
        "Explain why one condition produces another",
        ["Initial condition", "Observed outcome"],
        ConceptRelation.CAUSES,
    )

    matches = registry.match(
        graph,
        AudienceProfile(learning_goal="Understand the causal mechanism"),
    )

    assert matches[0].template_id == "cause_effect.v1"
    assert "operands=2/2" in matches[0].reason


def test_high_bm25_cannot_override_incompatible_capabilities() -> None:
    """Lexical similarity never authorizes a template that cannot route flow."""

    incompatible = CapabilityTemplate(
        "request_pipeline_words.v1",
        {"request", "response", "pipeline", "flow", "server"},
        TemplateCapabilities(
            relation_types=[ConceptRelation.CONTRASTS_WITH],
            semantic_actions=["compare"],
            minimum_operands=2,
            maximum_operands=4,
        ),
    )
    compatible = CapabilityTemplate(
        "route_capable.v1",
        {"route"},
        TemplateCapabilities(
            relation_types=[ConceptRelation.FLOWS_TO],
            semantic_actions=["route"],
            minimum_operands=2,
            maximum_operands=4,
        ),
    )
    registry = TemplateRegistry([incompatible, compatible])
    graph = _graph(
        "A request flows through a server pipeline to a response",
        ["Request", "Response"],
        ConceptRelation.FLOWS_TO,
    )

    matches = registry.match(graph)

    assert [item.template_id for item in matches] == ["route_capable.v1"]
    assert matches[0].capability_evidence
    assert matches[0].match_confidence == matches[0].score
    assert matches[0].parameter_provenance["components"].source == "extracted"


def test_capabilities_are_evaluated_against_each_shot_local_graph() -> None:
    """Unrelated relations cannot disqualify a valid local template family."""

    nodes = [
        ConceptNode(
            concept_id=concept_id,
            label=label,
            definition=f"Explain {label}.",
            importance=1,
            teaching_order=index,
            visual_affordances=[affordance],
        )
        for index, (concept_id, label, affordance) in enumerate([
            ("root", "Root", "tree"),
            ("left", "Left child", "tree"),
            ("right", "Right child", "tree"),
            ("alpha", "Option Alpha", "comparison"),
            ("beta", "Option Beta", "comparison"),
            ("input", "Input", "flow"),
            ("output", "Output", "flow"),
        ])
    ]
    graph = ConceptGraph(
        objectives=["Explain hierarchy, comparison, and process views"],
        nodes=nodes,
        edges=[
            ConceptEdge(
                edge_id="left_part",
                source_id="left",
                target_id="root",
                relation=ConceptRelation.PART_OF,
            ),
            ConceptEdge(
                edge_id="right_part",
                source_id="right",
                target_id="root",
                relation=ConceptRelation.PART_OF,
            ),
            ConceptEdge(
                edge_id="contrast",
                source_id="alpha",
                target_id="beta",
                relation=ConceptRelation.CONTRASTS_WITH,
            ),
            ConceptEdge(
                edge_id="flow",
                source_id="input",
                target_id="output",
                relation=ConceptRelation.FLOWS_TO,
            ),
        ],
        teaching_sequence=[node.concept_id for node in nodes],
    )
    pedagogy = PedagogyPlan(
        mode=PedagogyMode.CONCEPT_OVERVIEW,
        rationale="Exercise three local representations.",
        shots=[
            PedagogyShot(
                shot_id="hierarchy",
                purpose="demonstrate",
                visual_obligation="Draw the tree hierarchy.",
                narration_obligation="Explain parent and child roles.",
            ),
            PedagogyShot(
                shot_id="comparison",
                purpose="compare",
                visual_obligation="Compare both alternatives.",
                narration_obligation="Contrast the same criteria.",
            ),
            PedagogyShot(
                shot_id="process",
                purpose="connect",
                visual_obligation="Route input to output.",
                narration_obligation="Explain the directional flow.",
            ),
        ],
    )

    matches = TemplateRegistry(builtin_templates()).match_shots(
        graph,
        pedagogy,
        [["root", "left", "right"], ["alpha", "beta"], ["input", "output"]],
    )
    by_shot = {
        shot_id: {match.template_id for match in matches if shot_id in match.shot_ids}
        for shot_id in ("hierarchy", "comparison", "process")
    }

    assert "tree.v1" in by_shot["hierarchy"]
    assert "comparison.v1" in by_shot["comparison"]
    assert by_shot["process"].intersection({"pipeline.v1", "input_output.v1"})
