"""Lesson-planning and concept-graph contracts."""

from enum import Enum
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
    def normalize_teaching_sequence_labels(cls, value: object) -> object:
        """Convert matching node labels to IDs at the model boundary.

        Language models sometimes return human-readable concept labels in the
        sequence even though the contract requires concept IDs. Matching labels
        are normalized deterministically; unknown values remain untouched and
        are rejected by the existing strict graph validator.
        """

        if not isinstance(value, dict):
            return value
        nodes = value.get("nodes")
        sequence = value.get("teaching_sequence")
        if not isinstance(nodes, list) or not isinstance(sequence, list):
            return value

        label_to_id: dict[str, str] = {}
        known_ids: set[str] = set()
        for node in nodes:
            if isinstance(node, dict):
                label = node.get("label")
                concept_id = node.get("concept_id")
            else:
                label = getattr(node, "label", None)
                concept_id = getattr(node, "concept_id", None)
            if isinstance(label, str) and isinstance(concept_id, str):
                label_to_id[label.strip().casefold()] = concept_id
                known_ids.add(concept_id)
        normalized_sequence = [
            item
            if isinstance(item, str) and item in known_ids
            else label_to_id.get(item.strip().casefold(), item)
            if isinstance(item, str)
            else item
            for item in sequence
        ]
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
