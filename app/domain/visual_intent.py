"""Compact model-facing visual intent contracts."""

from enum import Enum
from typing import Self

from pydantic import Field, model_validator

from app.domain.lesson import ConceptRelation
from app.models.base import BaseModel, NonEmptyString


class RendererOperator(str, Enum):
    """High-level renderer programs available to visual planning."""

    PROCESS = "process"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    ARRAY = "array"
    TREE = "tree"
    GRAPH = "graph"
    EQUATION = "equation"
    CODE_TRACE = "code_trace"
    SIMULATION = "simulation"
    SPATIAL = "spatial"
    BINARY_SEARCH = "binary_search"
    SORTING = "sorting"
    GRAPH_TRAVERSAL = "graph_traversal"
    NEURAL_NETWORK = "neural_network"
    PROTOCOL = "protocol"
    SYSTEM = "system"
    TREE_INDEX = "tree_index"
    SCHEDULING = "scheduling"
    MEMORY_MAP = "memory_map"
    HASH_MAP = "hash_map"
    BLOCKCHAIN = "blockchain"
    CYCLE = "cycle"
    CAUSE_EFFECT = "cause_effect"
    LAYERED = "layered"
    FLOWCHART = "flowchart"
    FUNNEL = "funnel"
    VENN = "venn"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SEMANTIC_STRUCTURE = "semantic_structure"


class ShotSpec(BaseModel):
    """Describe teaching intent without scene-graph implementation details."""

    shot_id: NonEmptyString
    concept_ids: list[NonEmptyString] = Field(min_length=1)
    relation: ConceptRelation | None = None
    focal_object: NonEmptyString = Field(max_length=120)
    evidence: NonEmptyString = Field(max_length=280)
    transformation: NonEmptyString = Field(max_length=180)
    renderer_operator: RendererOperator


class VisualIntent(BaseModel):
    """Represent a complete high-level visual lesson program."""

    lesson_focus: NonEmptyString
    shots: list[ShotSpec] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_shot_ids(self) -> Self:
        """Require stable unique shot identifiers."""

        shot_ids = [shot.shot_id for shot in self.shots]
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("visual intent shot IDs must be unique")
        return self
