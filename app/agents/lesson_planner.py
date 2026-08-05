"""Specialized AI lesson planner."""

import re

from app.agents.base import StructuredGeminiAgent
from app.domain.generation import GenerationRequest
from app.domain.lesson import ConceptRelation, LessonPlan
from app.planning.graph_semantics import (
    CAUSAL_RELATIONS,
    analyze_graph,
    normalize_lesson_structure,
    relation_label_is_compatible,
)
from app.services.gemini_client import GeminiClient


class GeminiLessonPlanner:
    """Create a concept graph without narration or visual geometry."""

    def __init__(self, client: GeminiClient, max_attempts: int = 5) -> None:
        """Initialize the structured lesson-planning agent."""

        effective_attempts = (
            min(max_attempts, 2)
            if client.PROVIDER == "nvidia"
            else max_attempts
        )
        self._agent = StructuredGeminiAgent(
            client,
            LessonPlan,
            "Lesson Planner",
            effective_attempts,
        )

    def plan(self, request: GenerationRequest) -> LessonPlan:
        """Plan learning objectives, concepts, dependencies, and order."""

        instructions = (
            "Create an accurate educational lesson plan for the requested "
            "audience and duration. Write a concise, grammatically polished "
            "lesson title rather than copying an informal request verbatim. "
            "Identify only the concepts needed to meet "
            "the learning goal. Concept IDs and edge IDs must be stable, concise, "
            "and unique. The concept_graph.teaching_sequence array must contain "
            "each node's concept_id exactly once, in prerequisite-respecting "
            "order; use IDs from nodes[].concept_id, never human-readable labels "
            "from nodes[].label. Before returning, compare the set of sequence "
            "values with the set of node concept_id values. Model relationships "
            "truthfully: part_of means containment and must never be used merely "
            "to express teaching order; flows_to/transforms_to/causes describe "
            "mechanism; depends_on describes a prerequisite dependency; and "
            "contrasts_with is reserved for alternatives. For a question about "
            "how something works, include intermediate state changes and at least "
            "one dynamic relationship instead of returning only a parts list. "
            "Include relevant inputs, intermediate carriers or state changes, "
            "operating context, and outputs when they are needed to make the "
            "mechanism accurate. Do not collapse distinct causal stages into one "
            "vague concept. Scale explanatory depth to target_duration. "
            "Every mechanism concept should participate in a relationship. "
            "For dynamic edges, use a concise factual edge label that names what "
            "changes, moves, or causes the next state; do not merely repeat generic "
            "phrases such as 'flows to' or 'causes'. "
            "Use causes for labels such as enables, triggers, or produces; use "
            "flows_to only when a signal, input, request, response, or other "
            "payload actually moves between concepts. "
            "Never use natural-language verbs such as powers, controls, or monitors "
            "as the relation value; put the verb in label and choose the closest "
            "allowed relation token. "
            "Visual affordances describe useful diagram families such as pipeline, matrix, graph, timeline, "
            "array, tree, probability distribution, or component hierarchy. Do "
            "not write narration, storyboard operations, animations, or coordinates. "
            "Keep the lesson compact: normally use 4-8 concepts and concise strings; "
            "do not add unrelated examples or long assessment text. "
            "Do not copy JSON Schema metadata into the result: never output keys "
            "such as $defs, $schema, definitions, $ref, properties, or required "
            "inside concept_graph."
        )
        if request.sources:
            instructions += (
                " The request includes authoritative sources. Copy their source "
                "records into lesson.sources without changing IDs or assertions. "
                "Create claim-level lesson.claims for factual definitions and "
                "relations; every claim must cite source_ids, every concept node "
                "and edge must reference its supporting claim_ids, and confidence "
                "must reflect the evidence. Preserve relation direction exactly "
                "as stated by source relation_assertions. If a claim or relation "
                "is not supported, mark its claim status review instead of "
                "inventing support."
            )
        lesson = self._agent.generate(
            instructions,
            request.model_dump(mode="json"),
            validator=lambda candidate: self._validate_semantics(
                request,
                candidate,
            ),
        )
        return normalize_lesson_structure(lesson)

    @staticmethod
    def _validate_semantics(
        request: GenerationRequest,
        lesson: LessonPlan,
    ) -> None:
        """Reject relation graphs that cannot explain the requested mechanism."""

        normalized = normalize_lesson_structure(lesson)
        lesson.concept_graph = normalized.concept_graph

        request_text = " ".join(
            [request.topic, request.audience.learning_goal]
        ).casefold()
        terms = set(re.findall(r"[a-z0-9]+", request_text))
        asks_for_mechanism = (
            "mechanism" in terms
            or "process" in terms
            or (
                "how" in terms
                and bool(terms & {"work", "works", "working"})
            )
        )
        relations = {edge.relation for edge in lesson.concept_graph.edges}
        dynamic = set(CAUSAL_RELATIONS)
        if asks_for_mechanism and not relations.intersection(dynamic):
            raise ValueError(
                "a mechanism lesson must include a causal, flow, transformation, "
                "or transformation relation; a containment-only or "
                "dependency-only graph does not describe behavior"
            )
        minimum_concepts = min(
            8,
            max(4, round(request.target_duration / 12)),
        )
        if asks_for_mechanism and len(lesson.concept_graph.nodes) < minimum_concepts:
            raise ValueError(
                "the mechanism graph is too compressed for the requested "
                f"duration: expected at least {minimum_concepts} explanatory "
                "concepts"
            )
        generic_labels = {
            "flows to", "transforms to", "causes", "depends on", "leads to",
        }
        dynamic_edges = [
            edge for edge in lesson.concept_graph.edges if edge.relation in dynamic
        ]
        if asks_for_mechanism and dynamic_edges and not any(
            edge.label
            and edge.label.strip().casefold() not in generic_labels
            for edge in dynamic_edges
        ):
            raise ValueError(
                "mechanism edges need factual labels naming the transferred "
                "signal, material, state change, or causal action"
            )
        incompatible = [
            edge.edge_id
            for edge in dynamic_edges
            if not relation_label_is_compatible(edge.relation, edge.label)
        ]
        if incompatible:
            raise ValueError(
                "dynamic relationship labels must describe the declared flow, "
                "transformation, or causal action; incompatible edges: "
                f"{incompatible}"
            )
        prerequisites = {
            (node.concept_id, prerequisite)
            for node in lesson.concept_graph.nodes
            for prerequisite in node.prerequisites
        }
        reversed_dependencies = [
            edge.edge_id
            for edge in lesson.concept_graph.edges
            if edge.relation is ConceptRelation.DEPENDS_ON
            and (edge.source_id, edge.target_id) not in prerequisites
            and (edge.target_id, edge.source_id) in prerequisites
        ]
        if reversed_dependencies:
            raise ValueError(
                "depends_on points from the dependent concept to its "
                "prerequisite; reversed edges: "
                f"{reversed_dependencies}"
            )
        structure = analyze_graph(lesson.concept_graph)
        if asks_for_mechanism and not structure.is_process:
            raise ValueError(
                "a mechanism lesson needs a connected causal sequence in "
                "teaching order, not isolated or contradictory dynamic edges"
            )
        if len(lesson.concept_graph.nodes) > 2 and asks_for_mechanism:
            related = {
                concept_id
                for edge in lesson.concept_graph.edges
                for concept_id in (edge.source_id, edge.target_id)
            }
            isolated = [
                node.concept_id
                for node in lesson.concept_graph.nodes
                if node.concept_id not in related
            ]
            if isolated:
                raise ValueError(
                    f"mechanism concepts cannot be isolated: {isolated}"
                )
