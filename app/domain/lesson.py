"""Lesson-planning and concept-graph contracts."""

from enum import Enum
import re
from typing import Literal, Self

from pydantic import Field, model_validator

from app.models.base import BaseModel, NonEmptyString


class ConceptRelation(str, Enum):
    """Supported educational relationships between concepts."""

    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    CAUSES = "causes"
    TRANSFORMS_TO = "transforms_to"
    CONTRASTS_WITH = "contrasts_with"
    EXAMPLE_OF = "example_of"
    FLOWS_TO = "flows_to"


_FLOW_RELATION_LABELS = frozenset({
    "communicate", "connect", "interface", "output", "pass", "provide",
    "read", "receive", "request", "respond", "send", "signal", "supply",
    "transfer", "transmit", "write",
})
_RELATION_ALIASES = {
    "cause": ConceptRelation.CAUSES.value,
    "causes": ConceptRelation.CAUSES.value,
    "create": ConceptRelation.CAUSES.value,
    "creates": ConceptRelation.CAUSES.value,
    "drive": ConceptRelation.CAUSES.value,
    "drives": ConceptRelation.CAUSES.value,
    "enable": ConceptRelation.CAUSES.value,
    "enables": ConceptRelation.CAUSES.value,
    "generate": ConceptRelation.CAUSES.value,
    "generates": ConceptRelation.CAUSES.value,
    "lead_to": ConceptRelation.CAUSES.value,
    "leads_to": ConceptRelation.CAUSES.value,
    "power": ConceptRelation.CAUSES.value,
    "powers": ConceptRelation.CAUSES.value,
    "produce": ConceptRelation.CAUSES.value,
    "produces": ConceptRelation.CAUSES.value,
    "result_in": ConceptRelation.CAUSES.value,
    "results_in": ConceptRelation.CAUSES.value,
    "trigger": ConceptRelation.CAUSES.value,
    "triggers": ConceptRelation.CAUSES.value,
    "control": ConceptRelation.CAUSES.value,
    "controls": ConceptRelation.CAUSES.value,
    "manage": ConceptRelation.CAUSES.value,
    "manages": ConceptRelation.CAUSES.value,
    "depend": ConceptRelation.DEPENDS_ON.value,
    "depends": ConceptRelation.DEPENDS_ON.value,
    "needs": ConceptRelation.DEPENDS_ON.value,
    "requires": ConceptRelation.DEPENDS_ON.value,
    "relies_on": ConceptRelation.DEPENDS_ON.value,
    "monitors": ConceptRelation.DEPENDS_ON.value,
    "monitor": ConceptRelation.DEPENDS_ON.value,
    "communicates": ConceptRelation.FLOWS_TO.value,
    "connects": ConceptRelation.FLOWS_TO.value,
    "interfaces": ConceptRelation.FLOWS_TO.value,
    "outputs": ConceptRelation.FLOWS_TO.value,
    "passes": ConceptRelation.FLOWS_TO.value,
    "provides": ConceptRelation.FLOWS_TO.value,
    "reads": ConceptRelation.FLOWS_TO.value,
    "receives": ConceptRelation.FLOWS_TO.value,
    "requests": ConceptRelation.FLOWS_TO.value,
    "responds_to": ConceptRelation.FLOWS_TO.value,
    "sends": ConceptRelation.FLOWS_TO.value,
    "supplies": ConceptRelation.FLOWS_TO.value,
    "transfers": ConceptRelation.FLOWS_TO.value,
    "transmits": ConceptRelation.FLOWS_TO.value,
    "writes": ConceptRelation.FLOWS_TO.value,
    "becomes": ConceptRelation.TRANSFORMS_TO.value,
    "changes_to": ConceptRelation.TRANSFORMS_TO.value,
    "converts": ConceptRelation.TRANSFORMS_TO.value,
    "decodes": ConceptRelation.TRANSFORMS_TO.value,
    "encodes": ConceptRelation.TRANSFORMS_TO.value,
    "maps_to": ConceptRelation.TRANSFORMS_TO.value,
    "transforms": ConceptRelation.TRANSFORMS_TO.value,
    "compares_with": ConceptRelation.CONTRASTS_WITH.value,
    "contrasts": ConceptRelation.CONTRASTS_WITH.value,
    "illustrates": ConceptRelation.EXAMPLE_OF.value,
    "instance_of": ConceptRelation.EXAMPLE_OF.value,
}


def normalize_relation_alias(value: object, label: object = None) -> object:
    """Convert common model verbs to the closed relation vocabulary.

    The provider receives the schema as prompt text, so it can still emit a
    natural-language verb such as ``powers`` instead of an enum value. Keep
    unknown values untouched so genuinely unsupported relations still fail
    validation instead of being silently assigned an arbitrary meaning.
    """

    if not isinstance(value, str):
        return value
    key = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if key in {relation.value for relation in ConceptRelation}:
        return key
    if key in {"powers", "power", "controls", "control"} and isinstance(label, str):
        label_tokens = set(re.findall(r"[a-z0-9]+", label.casefold()))
        if label_tokens.intersection(_FLOW_RELATION_LABELS):
            return ConceptRelation.FLOWS_TO.value
    return _RELATION_ALIASES.get(key, value)


def _normalize_teaching_sequence(
    nodes: list[object],
    sequence: object,
) -> list[str]:
    """Return every known concept exactly once in dependency order."""

    node_ids: list[str] = []
    labels: dict[str, str] = {}
    prerequisites: dict[str, set[str]] = {}
    teaching_orders: dict[str, int] = {}
    for index, raw_node in enumerate(nodes):
        if isinstance(raw_node, dict):
            concept_id = raw_node.get("concept_id")
            label = raw_node.get("label")
            raw_prerequisites = raw_node.get("prerequisites", [])
            teaching_order = raw_node.get("teaching_order", index)
        else:
            concept_id = getattr(raw_node, "concept_id", None)
            label = getattr(raw_node, "label", None)
            raw_prerequisites = getattr(raw_node, "prerequisites", [])
            teaching_order = getattr(raw_node, "teaching_order", index)
        if not isinstance(concept_id, str):
            continue
        node_ids.append(concept_id)
        if isinstance(label, str):
            labels[label.strip().casefold()] = concept_id
        prerequisites[concept_id] = (
            {item for item in raw_prerequisites if isinstance(item, str)}
            if isinstance(raw_prerequisites, list)
            else set()
        )
        teaching_orders[concept_id] = (
            teaching_order if isinstance(teaching_order, int) else index
        )

    known_ids = set(node_ids)
    if not known_ids:
        return []
    preferred: list[str] = []
    seen: set[str] = set()
    raw_sequence = sequence if isinstance(sequence, list) else []
    for item in raw_sequence:
        if not isinstance(item, str):
            continue
        candidate = item if item in known_ids else labels.get(item.strip().casefold())
        if candidate in known_ids and candidate not in seen:
            preferred.append(candidate)
            seen.add(candidate)

    preferred_rank = {concept_id: rank for rank, concept_id in enumerate(preferred)}
    ordered: list[str] = []
    remaining = set(known_ids)
    while remaining:
        placed = set(ordered)
        ready = [
            concept_id
            for concept_id in remaining
            if prerequisites.get(concept_id, set()).issubset(placed)
        ]
        candidates = ready or list(remaining)
        concept_id = min(
            candidates,
            key=lambda item: (
                preferred_rank.get(item, len(preferred_rank)),
                teaching_orders.get(item, len(teaching_orders)),
                item,
            ),
        )
        ordered.append(concept_id)
        remaining.remove(concept_id)
    return ordered


class ConceptNode(BaseModel):
    """Represent one teachable concept and its visual affordances."""

    concept_id: NonEmptyString
    label: NonEmptyString
    definition: NonEmptyString
    importance: float = Field(ge=0, le=1)
    prerequisites: list[NonEmptyString] = Field(default_factory=list)
    teaching_order: int = Field(ge=0)
    visual_affordances: list[NonEmptyString] = Field(default_factory=list)


class ConceptEdge(BaseModel):
    """Represent one directed semantic relationship."""

    edge_id: NonEmptyString
    source_id: NonEmptyString
    target_id: NonEmptyString
    relation: ConceptRelation
    label: str | None = None


class ConceptGraph(BaseModel):
    """Represent validated concepts, relationships, and teaching order."""

    schema_version: Literal["2.0"] = "2.0"
    objectives: list[NonEmptyString] = Field(min_length=1)
    nodes: list[ConceptNode] = Field(min_length=1)
    edges: list[ConceptEdge] = Field(default_factory=list)
    teaching_sequence: list[NonEmptyString] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_relation_aliases(cls, value: object) -> object:
        """Normalize natural-language edge types before enum validation."""

        if not isinstance(value, dict) or not isinstance(value.get("edges"), list):
            return value
        changed = False
        edges: list[object] = []
        for raw_edge in value["edges"]:
            if not isinstance(raw_edge, dict):
                edges.append(raw_edge)
                continue
            relation = raw_edge.get("relation")
            normalized_relation = normalize_relation_alias(
                relation,
                raw_edge.get("label"),
            )
            if normalized_relation != relation:
                changed = True
                edges.append({**raw_edge, "relation": normalized_relation})
            else:
                edges.append(raw_edge)
        return {**value, "edges": edges} if changed else value

    @model_validator(mode="before")
    @classmethod
    def normalize_teaching_sequence_labels(cls, value: object) -> object:
        """Repair model sequence omissions, duplicates, and label references.

        Language models sometimes return human-readable concept labels in the
        sequence even though the contract requires concept IDs. Unknown or
        duplicate entries are discarded, and omitted node IDs are appended in
        stable prerequisite-respecting order.
        """

        if not isinstance(value, dict):
            return value
        nodes = value.get("nodes")
        sequence = value.get("teaching_sequence")
        if not isinstance(nodes, list) or not isinstance(sequence, list):
            return value

        normalized_sequence = _normalize_teaching_sequence(nodes, sequence)
        return {**value, "teaching_sequence": normalized_sequence}

    @model_validator(mode="before")
    @classmethod
    def drop_schema_metadata(cls, value: object) -> object:
        """Ignore JSON Schema definitions accidentally echoed by a model.

        The structured prompt includes a JSON Schema. Some models copy schema
        metadata such as ``$defs`` into the nested concept graph even though it
        is not part of the lesson data. Removing only these schema-only keys at
        the boundary keeps the data contract strict for real graph fields.
        """

        if not isinstance(value, dict):
            return value
        schema_metadata = {"$defs", "$schema", "definitions"}
        return {
            key: item
            for key, item in value.items()
            if key not in schema_metadata
        }

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        """Require unique nodes, valid references, and a complete sequence."""

        node_ids = [node.concept_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("concept IDs must be unique")
        known_ids = set(node_ids)
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("concept edge IDs must be unique")
        for node in self.nodes:
            if not set(node.prerequisites).issubset(known_ids):
                raise ValueError("concept prerequisites must reference known nodes")
            if node.concept_id in node.prerequisites:
                raise ValueError("a concept cannot require itself")
        for edge in self.edges:
            if edge.source_id not in known_ids or edge.target_id not in known_ids:
                raise ValueError("concept edges must reference known nodes")
            if edge.source_id == edge.target_id:
                raise ValueError("concept edges cannot be self-referential")
        if len(self.teaching_sequence) != len(set(self.teaching_sequence)):
            raise ValueError("teaching sequence IDs must be unique")
        if set(self.teaching_sequence) != known_ids:
            raise ValueError("teaching sequence must contain every concept exactly once")
        return self


class LessonPlan(BaseModel):
    """Describe the educational plan consumed by visual planning."""

    schema_version: Literal["2.0"] = "2.0"
    title: NonEmptyString
    summary: NonEmptyString
    concept_graph: ConceptGraph
    misconceptions: list[NonEmptyString] = Field(default_factory=list)
    assessment_prompts: list[NonEmptyString] = Field(default_factory=list)
