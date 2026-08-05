"""Compact model-facing visual intent contracts."""

from enum import Enum
from typing import Literal, Self

from pydantic import Field, model_validator

from app.domain.lesson import ConceptRelation
from app.models.base import BaseModel, NonEmptyString


class RendererOperator(str, Enum):
    """High-level, bounded VisualDSL programs available to visual planning.

    The names are intentionally semantic rather than pixel-oriented. Each
    operator has a validated parameter contract and is expanded by the
    deterministic operator compiler before layout or rendering.
    """

    ICON = "icon"
    GROUP = "group"
    FLOW = "flow"
    CALLOUT = "callout"
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
    PLOT = "plot"
    TABLE = "table"
    MATRIX = "matrix"
    MOLECULE = "molecule"
    CIRCUIT = "circuit"
    MAP = "map"
    ANATOMY = "anatomy"
    TRANSFORM = "transform"
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
    supporting_operators: list[RendererOperator] = Field(
        default_factory=list,
        max_length=2,
    )
    state_ref: str | None = Field(default=None, max_length=80)
    shared_object_keys: list[NonEmptyString] = Field(
        default_factory=list,
        max_length=4,
    )
    action_obligations: list[NonEmptyString] = Field(
        default_factory=list,
        max_length=6,
    )


class OperatorInstance(BaseModel):
    """One bounded semantic operator owned by a shot or shared program state."""

    instance_id: NonEmptyString
    operator: RendererOperator
    concept_ids: list[NonEmptyString] = Field(min_length=1, max_length=12)
    ownership: Literal["shared", "shot"] = "shot"
    layout_region: Literal[
        "full", "left", "right", "top", "bottom", "overlay"
    ] = "full"
    state_ref: str | None = None
    action_obligations: list[NonEmptyString] = Field(
        default_factory=list,
        max_length=6,
    )


class VisualProgramShot(BaseModel):
    """Composition and lifecycle boundary for one reviewed teaching shot."""

    shot_id: NonEmptyString
    operator_instance_ids: list[NonEmptyString] = Field(min_length=1, max_length=3)
    state_ref: str | None = None
    cleanup_instance_ids: list[NonEmptyString] = Field(default_factory=list)
    action_obligations: list[NonEmptyString] = Field(
        default_factory=list,
        max_length=6,
    )


class VisualProgram(BaseModel):
    """Compiler-owned, multi-operator program without renderer primitives."""

    program_id: NonEmptyString
    shared_instance_ids: list[NonEmptyString] = Field(default_factory=list)
    operator_instances: list[OperatorInstance] = Field(min_length=1, max_length=18)
    shots: list[VisualProgramShot] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        instance_ids = [item.instance_id for item in self.operator_instances]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("visual program operator instance IDs must be unique")
        known = set(instance_ids)
        shared = set(self.shared_instance_ids)
        if not shared.issubset(known):
            raise ValueError("visual program shared instances must exist")
        ownership = {
            item.instance_id: item.ownership for item in self.operator_instances
        }
        if any(ownership[item] != "shared" for item in shared):
            raise ValueError("shared instance IDs must have shared ownership")
        for shot in self.shots:
            referenced = set(shot.operator_instance_ids)
            cleanup = set(shot.cleanup_instance_ids)
            if not referenced.issubset(known) or not cleanup.issubset(known):
                raise ValueError("visual program shot references unknown instances")
            if cleanup.intersection(shared):
                raise ValueError("shared instances cannot be shot cleanup targets")
        return self


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


class VisualIntentPatch(BaseModel):
    """Return only the high-level shots named by a beat-local repair request."""

    shots: list[ShotSpec] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_shot_ids(self) -> Self:
        shot_ids = [shot.shot_id for shot in self.shots]
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("visual intent patch shot IDs must be unique")
        return self
