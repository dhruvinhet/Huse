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
