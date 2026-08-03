"""Validated procedural template operators with topic-specific structures."""

from dataclasses import dataclass
import re
from pydantic import Field

from app.domain.layout import ConstraintStrength, ConstraintType, LayoutConstraint
from app.domain.lesson import ConceptRelation
from app.domain.storyboard import VisualObjectSpec
from app.domain.visual_intent import RendererOperator
from app.models.base import BaseModel, NonEmptyString


class SemanticConceptParameter(BaseModel):
    """One factual concept operand supplied by the validated lesson graph."""

    concept_id: NonEmptyString
    label: NonEmptyString
    definition: NonEmptyString
    importance: float = Field(ge=0, le=1)
    order: int = Field(ge=0)
    visual_affordances: list[NonEmptyString] = Field(default_factory=list)


class SemanticRelationParameter(BaseModel):
    """One typed relationship supplied by the validated lesson graph."""

    source_id: NonEmptyString
    target_id: NonEmptyString
    relation: ConceptRelation
    label: str | None = None


class OperatorParameters(BaseModel):
    """Common validated operands for procedural semantic operators."""

    object_id: NonEmptyString
    label: NonEmptyString
    operands: list[NonEmptyString] = Field(default_factory=list, max_length=10)
    concepts: list[SemanticConceptParameter] = Field(default_factory=list, max_length=12)
    relations: list[SemanticRelationParameter] = Field(default_factory=list, max_length=24)


class BinarySearchParameters(OperatorParameters):
    """State required to render an actual binary-search step."""

    values: list[int] = Field(default_factory=lambda: [2, 5, 8, 12, 16], min_length=3, max_length=10)
    target: int = 12
    low: int = Field(default=0, ge=0)
    high: int | None = Field(default=None, ge=0)
    mid: int | None = Field(default=None, ge=0)
    comparison: NonEmptyString = "target compared with middle value"
    active_code_line: NonEmptyString = "mid = (low + high) // 2"


class SortingParameters(OperatorParameters):
    """State required for partition- or merge-based sorting."""

    values: list[int] = Field(default_factory=lambda: [8, 3, 7, 2, 6], min_length=3, max_length=10)
    pivot_index: int = Field(default=2, ge=0)
    active_range: tuple[int, int] = (0, 4)
    phase: NonEmptyString = "partition"


class GraphTraversalParameters(OperatorParameters):
    """State required for BFS/DFS execution."""

    nodes: list[NonEmptyString] = Field(default_factory=lambda: ["A", "B", "C", "D"], min_length=2, max_length=10)
    edges: list[tuple[int, int]] = Field(default_factory=lambda: [(0, 1), (0, 2), (1, 3)], max_length=16)
    current_index: int = Field(default=0, ge=0)
    frontier: list[int] = Field(default_factory=lambda: [1, 2], max_length=10)
    visited: list[int] = Field(default_factory=lambda: [0], max_length=10)
    strategy: NonEmptyString = "breadth first"


class NetworkParameters(OperatorParameters):
    """Layers and active tensors for neural-network operators."""

    layers: list[NonEmptyString] = Field(default_factory=lambda: ["Input", "Feature extraction", "Hidden representation", "Output"], min_length=2, max_length=8)
    tensor_shapes: list[NonEmptyString] = Field(default_factory=lambda: ["H×W×C", "H/2×W/2", "features", "classes"], max_length=8)
    active_layer: int = Field(default=1, ge=0)


class ProtocolParameters(OperatorParameters):
    """Participants and exchanged messages for protocols/API flows."""

    participants: list[NonEmptyString] = Field(default_factory=lambda: ["Client", "Server"], min_length=2, max_length=5)
    messages: list[NonEmptyString] = Field(default_factory=lambda: ["Request", "Validation", "Response"], min_length=2, max_length=8)
    active_message: int = Field(default=0, ge=0)


class SystemParameters(OperatorParameters):
    """Typed nodes and connections for system architecture."""

    nodes: list[NonEmptyString] = Field(default_factory=lambda: ["Client", "Gateway", "Service", "Data store"], min_length=2, max_length=10)
    roles: list[NonEmptyString] = Field(default_factory=lambda: ["caller", "routing", "compute", "storage"], max_length=10)
    edges: list[tuple[int, int]] = Field(default_factory=lambda: [(0, 1), (1, 2), (2, 3)], max_length=16)


class TreeIndexParameters(OperatorParameters):
    """Hierarchy and lookup path for tree/database-index operators."""

    nodes: list[NonEmptyString] = Field(default_factory=lambda: ["Root", "Index A", "Index B", "Target row"], min_length=2, max_length=10)
    parents: list[int | None] = Field(default_factory=lambda: [None, 0, 0, 1], max_length=10)
    lookup_path: list[int] = Field(default_factory=lambda: [0, 1, 3], max_length=10)


class SchedulingParameters(OperatorParameters):
    """Queues and CPU state for scheduling timelines."""

    states: list[NonEmptyString] = Field(default_factory=lambda: ["Ready", "Running", "Waiting", "Complete"], min_length=2, max_length=8)
    ready_queue: list[NonEmptyString] = Field(default_factory=lambda: ["P2", "P3"], max_length=8)
    running: NonEmptyString = "P1"


class MemoryParameters(OperatorParameters):
    """Addressed memory regions rather than renamed boxes."""

    regions: list[NonEmptyString] = Field(default_factory=lambda: ["Code", "Stack", "Free", "Heap"], min_length=2, max_length=8)
    addresses: list[NonEmptyString] = Field(
        default_factory=lambda: ["low", "middle-low", "middle-high", "high"],
        max_length=8,
    )
    active_region: int = Field(default=3, ge=0)


class HashMapParameters(OperatorParameters):
    """Hash calculation and explicit buckets."""

    key: NonEmptyString = "key"
    hash_value: int = Field(default=2, ge=0)
    buckets: list[NonEmptyString] = Field(default_factory=lambda: ["—", "A", "target", "B"], min_length=2, max_length=10)


class BlockchainParameters(OperatorParameters):
    """Linked blocks with hash identity."""

    blocks: list[NonEmptyString] = Field(default_factory=lambda: ["Genesis", "Block 1", "Block 2"], min_length=2, max_length=8)
    hashes: list[NonEmptyString] = Field(default_factory=lambda: ["0000", "8af2", "31bd"], max_length=8)
    active_block: int = Field(default=2, ge=0)


class GenericOperatorParameters(OperatorParameters):
    """Validated operands for reusable educational operators."""

    values: list[float] = Field(default_factory=list, max_length=10)
    active_index: int = Field(default=0, ge=0)


PARAMETER_MODELS: dict[RendererOperator, type[OperatorParameters]] = {
    RendererOperator.BINARY_SEARCH: BinarySearchParameters,
    RendererOperator.SORTING: SortingParameters,
    RendererOperator.GRAPH_TRAVERSAL: GraphTraversalParameters,
    RendererOperator.NEURAL_NETWORK: NetworkParameters,
    RendererOperator.PROTOCOL: ProtocolParameters,
    RendererOperator.SYSTEM: SystemParameters,
    RendererOperator.TREE_INDEX: TreeIndexParameters,
    RendererOperator.SCHEDULING: SchedulingParameters,
    RendererOperator.MEMORY_MAP: MemoryParameters,
    RendererOperator.HASH_MAP: HashMapParameters,
    RendererOperator.BLOCKCHAIN: BlockchainParameters,
}


@dataclass(frozen=True, slots=True)
class OperatorTemplate:
    """Bind reviewed retrieval metadata to one procedural renderer operator."""

    template_id: str
    keywords: frozenset[str]
    operator: RendererOperator
    default_label: str
    metadata: tuple[str, ...]
    parameter_model: type[OperatorParameters] = GenericOperatorParameters

    @property
    def diagram_kind(self) -> str:
        """Expose operator identity to local retrieval metadata."""

        return self.operator.value

    @property
    def components(self) -> tuple[str, ...]:
        """Expose descriptive operands to BM25 without defining scene boxes."""

        return self.metadata

    def parameter_schema(self) -> dict[str, object]:
        """Return the explicit validated operand schema for this operator."""

        return self.parameter_model.model_json_schema()

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Validate operands and compile a topic-specific semantic hierarchy."""

        normalized = dict(parameters)
        normalized.setdefault("label", self.default_label)
        candidate_operands = next(
            (
                normalized[key]
                for key in ("operands", "components", "stages", "nodes")
                if isinstance(normalized.get(key), list)
            ),
            [],
        )
        normalized["operands"] = candidate_operands
        operand_field = {
            RendererOperator.GRAPH_TRAVERSAL: "nodes",
            RendererOperator.NEURAL_NETWORK: "layers",
            RendererOperator.PROTOCOL: "messages",
            RendererOperator.SYSTEM: "nodes",
            RendererOperator.TREE_INDEX: "nodes",
            RendererOperator.SCHEDULING: "states",
            RendererOperator.MEMORY_MAP: "regions",
            RendererOperator.BLOCKCHAIN: "blocks",
        }.get(self.operator)
        if operand_field is not None and candidate_operands:
            normalized.setdefault(operand_field, candidate_operands)
        allowed = self.parameter_model.model_fields
        validated = self.parameter_model.model_validate(
            {key: value for key, value in normalized.items() if key in allowed}
        )
        return SemanticOperatorCompiler().compile(self.operator, validated)


class SemanticOperatorCompiler:
    """Expand validated operator operands into visibly distinct structures."""

    def compile(
        self,
        operator: RendererOperator,
        parameters: OperatorParameters,
    ) -> VisualObjectSpec:
        """Dispatch to a procedural operator implementation."""

        if parameters.concepts and self._uses_semantic_structure(operator):
            return self._compile_semantic_structure(operator, parameters)
        method = getattr(self, f"_compile_{operator.value}", None)
        if callable(method):
            return method(parameters)
        return self._compile_educational(operator, parameters)

    @staticmethod
    def _uses_semantic_structure(operator: RendererOperator) -> bool:
        """Use the lesson graph for compositional operators, not a label chain."""

        stateful = {
            RendererOperator.BINARY_SEARCH,
            RendererOperator.SORTING,
            RendererOperator.GRAPH_TRAVERSAL,
            RendererOperator.PROTOCOL,
            RendererOperator.TREE_INDEX,
            RendererOperator.SCHEDULING,
            RendererOperator.MEMORY_MAP,
            RendererOperator.HASH_MAP,
            RendererOperator.BLOCKCHAIN,
            RendererOperator.EQUATION,
            RendererOperator.CODE_TRACE,
            RendererOperator.SIMULATION,
            RendererOperator.BAR_CHART,
            RendererOperator.LINE_CHART,
        }
        return operator not in stateful

    def _compile_semantic_structure(
        self,
        operator: RendererOperator,
        parameters: OperatorParameters,
    ) -> VisualObjectSpec:
        """Compile typed graph relations into hierarchy and truthful connectors."""

        concepts = {item.concept_id: item for item in parameters.concepts}
        ordered = sorted(parameters.concepts, key=lambda item: item.order)
        parent_of = {
            item.source_id: item.target_id
            for item in parameters.relations
            if item.relation is ConceptRelation.PART_OF
            and item.source_id in concepts
            and item.target_id in concepts
        }
        children_of: dict[str, list[str]] = {}
        for child_id, parent_id in parent_of.items():
            children_of.setdefault(parent_id, []).append(child_id)
        order = {item.concept_id: item.order for item in ordered}
        for children in children_of.values():
            children.sort(key=lambda item: order[item])

        object_ids = {
            item.concept_id: (
                f"{parameters.object_id}_concept_{self._safe_id(item.concept_id)}"
            )
            for item in ordered
        }

        def build(concept_id: str, lineage: frozenset[str]) -> VisualObjectSpec:
            if concept_id in lineage:
                raise ValueError("part_of relationships must form an acyclic hierarchy")
            concept = concepts[concept_id]
            members = [
                build(member_id, lineage | {concept_id})
                for member_id in children_of.get(concept_id, [])
            ]
            if members:
                return self._semantic_group(
                    object_ids[concept_id], concept, members
                )
            leaf_kind = self._semantic_leaf_kind(
                operator,
                concept.visual_affordances,
                concept_id in parent_of,
            )
            content: dict[str, object] = {
                "label": concept.label,
                "definition": concept.definition,
                "importance": concept.importance,
            }
            if leaf_kind == "component":
                content["detail"] = concept.definition
            return VisualObjectSpec(
                object_id=object_ids[concept_id],
                kind=leaf_kind,
                semantic_role="concept",
                concept_ids=[concept.concept_id],
                content=content,
                accessibility_label=f"{concept.label}: {concept.definition}",
            )

        roots = [
            build(item.concept_id, frozenset())
            for item in ordered
            if item.concept_id not in parent_of
        ]
        connectors: list[VisualObjectSpec] = []
        for index, relation in enumerate(parameters.relations):
            if relation.relation is ConceptRelation.PART_OF:
                continue
            if (
                relation.source_id not in object_ids
                or relation.target_id not in object_ids
            ):
                continue
            relation_label = (
                relation.label
                or relation.relation.value.replace("_", " ")
            )
            connectors.append(
                VisualObjectSpec(
                    object_id=f"{parameters.object_id}_relation_{index:03d}",
                    kind="connector",
                    semantic_role=f"relation_{relation.relation.value}",
                    concept_ids=[relation.source_id, relation.target_id],
                    content={
                        "source_id": object_ids[relation.source_id],
                        "target_id": object_ids[relation.target_id],
                        "label": relation_label,
                        "relation": relation.relation.value,
                    },
                    accessibility_label=(
                        f"{concepts[relation.source_id].label} "
                        f"{relation_label} "
                        f"{concepts[relation.target_id].label}"
                    ),
                )
            )

        children = [*roots, *connectors]
        root_ids = [item.object_id for item in roots]
        linear_operator = operator in {
            RendererOperator.PROCESS,
            RendererOperator.TIMELINE,
            RendererOperator.CAUSE_EFFECT,
        }
        layout = (
            "horizontal"
            if len(roots) <= 3 or (linear_operator and len(roots) <= 5)
            else "grid"
        )
        relation_kinds = sorted(
            {item.relation.value for item in parameters.relations}
        )
        return VisualObjectSpec(
            object_id=parameters.object_id,
            kind="nested_group",
            semantic_role="procedural_operator",
            concept_ids=[item.concept_id for item in ordered],
            content={
                "label": parameters.label,
                "layout": layout,
                "operator": "semantic_structure",
                "source_operator": operator.value,
                "relation_kinds": relation_kinds,
                "connector_mode": "explicit" if connectors else "none",
            },
            children=children,
            constraints=self._constraints(
                parameters.object_id, root_ids, layout
            ),
            accessibility_label=(
                f"{parameters.label} semantic structure with "
                f"{', '.join(relation_kinds) or 'concept'} relationships"
            ),
        )

    def _semantic_group(
        self,
        object_id: str,
        concept: SemanticConceptParameter,
        members: list[VisualObjectSpec],
    ) -> VisualObjectSpec:
        """Render a whole as a real container around its declared parts."""

        layout = "horizontal" if len(members) <= 4 else "grid"
        member_ids = [item.object_id for item in members]
        return VisualObjectSpec(
            object_id=object_id,
            kind="nested_group",
            semantic_role="concept_whole",
            concept_ids=[concept.concept_id],
            content={
                "label": concept.label,
                "definition": concept.definition,
                "layout": layout,
                "relation": "contains",
            },
            children=members,
            constraints=self._constraints(object_id, member_ids, layout),
            accessibility_label=(
                f"{concept.label}, containing its declared parts"
            ),
        )

    @staticmethod
    def _semantic_leaf_kind(
        operator: RendererOperator,
        affordances: list[str],
        is_declared_part: bool,
    ) -> str:
        """Choose a leaf kind from the operator's structural contract."""

        del affordances
        if is_declared_part:
            return "component"
        if operator is RendererOperator.ARRAY:
            return "array_cell"
        if operator is RendererOperator.TREE:
            return "tree_node"
        if operator in {
            RendererOperator.GRAPH,
            RendererOperator.SYSTEM,
        }:
            return "graph_node"
        if operator is RendererOperator.EQUATION:
            return "equation"
        return "component"

    @staticmethod
    def _constraints(
        object_id: str,
        child_ids: list[str],
        layout: str,
    ) -> list[LayoutConstraint]:
        if not child_ids:
            return []
        return [
            LayoutConstraint(
                constraint_id=f"{object_id}_contain",
                type=ConstraintType.CONTAIN,
                subject_ids=child_ids,
                reference_id=object_id,
                strength=ConstraintStrength.REQUIRED,
            ),
            LayoutConstraint(
                constraint_id=f"{object_id}_layout",
                type=ConstraintType.DISTRIBUTE,
                subject_ids=child_ids,
                reference_id=object_id,
                parameters={"axis": layout, "gap": 48},
            ),
        ]

    @staticmethod
    def _safe_id(value: str) -> str:
        return (
            re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
            or "concept"
        )

    def _compile_binary_search(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = BinarySearchParameters.model_validate(raw.model_dump())
        high = parameters.high if parameters.high is not None else len(parameters.values) - 1
        mid = parameters.mid if parameters.mid is not None else (parameters.low + high) // 2
        children = [
            self._leaf(parameters.object_id, "array_cell", index, str(value), "array_value")
            for index, value in enumerate(parameters.values)
        ]
        return self._container(
            parameters,
            "array",
            children,
            "horizontal",
            {
                "operator": "binary_search",
                "low": min(parameters.low, len(children) - 1),
                "high": min(high, len(children) - 1),
                "mid": min(mid, len(children) - 1),
                "target": parameters.target,
                "comparison": parameters.comparison,
                "active_code_line": parameters.active_code_line,
            },
        )

    def _compile_sorting(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = SortingParameters.model_validate(raw.model_dump())
        children = [
            self._leaf(parameters.object_id, "array_cell", index, str(value), "sortable_value")
            for index, value in enumerate(parameters.values)
        ]
        return self._container(
            parameters, "array", children, "horizontal",
            {"operator": "sorting", "pivot_index": parameters.pivot_index, "active_range": list(parameters.active_range), "phase": parameters.phase},
        )

    def _compile_graph_traversal(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = GraphTraversalParameters.model_validate(raw.model_dump())
        return self._graph_container(parameters, "graph", parameters.nodes, parameters.edges, {
            "operator": "graph_traversal", "current_index": parameters.current_index,
            "frontier": parameters.frontier, "visited": parameters.visited,
            "strategy": parameters.strategy,
        })

    def _compile_neural_network(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = NetworkParameters.model_validate(raw.model_dump())
        children = [
            self._leaf(parameters.object_id, "component", index, label, "neural_layer", {"tensor_shape": parameters.tensor_shapes[index] if index < len(parameters.tensor_shapes) else ""})
            for index, label in enumerate(parameters.layers)
        ]
        children.extend(self._connectors(parameters.object_id, len(parameters.layers)))
        return self._container(parameters, "neural_network", children, "horizontal", {"operator": "neural_network", "active_layer": parameters.active_layer})

    def _compile_protocol(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = ProtocolParameters.model_validate(raw.model_dump())
        children = [
            self._leaf(parameters.object_id, "component", index, message, "protocol_message", {"participant_from": parameters.participants[index % len(parameters.participants)], "participant_to": parameters.participants[(index + 1) % len(parameters.participants)]})
            for index, message in enumerate(parameters.messages)
        ]
        return self._container(parameters, "timeline", children, "horizontal", {"operator": "protocol", "participants": parameters.participants, "active_message": parameters.active_message})

    def _compile_system(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = SystemParameters.model_validate(raw.model_dump())
        return self._graph_container(parameters, "graph", parameters.nodes, parameters.edges, {"operator": "system", "roles": parameters.roles})

    def _compile_tree_index(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = TreeIndexParameters.model_validate(raw.model_dump())
        children = [
            self._leaf(parameters.object_id, "tree_node", index, label, "index_node", {"parent_index": parameters.parents[index] if index < len(parameters.parents) else None, "on_lookup_path": index in parameters.lookup_path})
            for index, label in enumerate(parameters.nodes)
        ]
        return self._container(parameters, "tree", children, "tree", {"operator": "tree_index", "lookup_path": parameters.lookup_path})

    def _compile_scheduling(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = SchedulingParameters.model_validate(raw.model_dump())
        children = [self._leaf(parameters.object_id, "component", index, state, "scheduler_state") for index, state in enumerate(parameters.states)]
        return self._container(parameters, "timeline", children, "horizontal", {"operator": "scheduling", "running": parameters.running, "ready_queue": parameters.ready_queue})

    def _compile_memory_map(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = MemoryParameters.model_validate(raw.model_dump())
        children = [self._leaf(parameters.object_id, "array_cell", index, region, "memory_region", {"address": parameters.addresses[index] if index < len(parameters.addresses) else "", "active": index == parameters.active_region}) for index, region in enumerate(parameters.regions)]
        return self._container(parameters, "array", children, "vertical", {"operator": "memory_map", "active_region": parameters.active_region})

    def _compile_hash_map(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = HashMapParameters.model_validate(raw.model_dump())
        children = [self._leaf(parameters.object_id, "array_cell", index, value, "hash_bucket", {"bucket": index, "selected": index == parameters.hash_value % len(parameters.buckets)}) for index, value in enumerate(parameters.buckets)]
        return self._container(parameters, "hash_table", children, "vertical", {"operator": "hash_map", "key": parameters.key, "hash_value": parameters.hash_value})

    def _compile_blockchain(self, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = BlockchainParameters.model_validate(raw.model_dump())
        children = [self._leaf(parameters.object_id, "component", index, block, "block", {"hash": parameters.hashes[index] if index < len(parameters.hashes) else "", "active": index == parameters.active_block}) for index, block in enumerate(parameters.blocks)]
        children.extend(self._connectors(parameters.object_id, len(parameters.blocks), role="hash_link"))
        return self._container(parameters, "blockchain", children, "horizontal", {"operator": "blockchain", "active_block": parameters.active_block})

    def _compile_educational(self, operator: RendererOperator, raw: OperatorParameters) -> VisualObjectSpec:
        parameters = GenericOperatorParameters.model_validate(raw.model_dump())
        labels = parameters.operands or ["Input", "Relation", "Result"]
        kind_map = {
            RendererOperator.COMPARISON: "table", RendererOperator.TIMELINE: "timeline",
            RendererOperator.CYCLE: "graph", RendererOperator.CAUSE_EFFECT: "pipeline",
            RendererOperator.PROCESS: "pipeline", RendererOperator.LAYERED: "nested_group",
            RendererOperator.FLOWCHART: "flowchart", RendererOperator.FUNNEL: "nested_group",
            RendererOperator.VENN: "nested_group", RendererOperator.BAR_CHART: "histogram",
            RendererOperator.LINE_CHART: "coordinate_axes", RendererOperator.EQUATION: "nested_group",
            RendererOperator.CODE_TRACE: "document",
        }
        root_kind = kind_map.get(operator, "nested_group")
        leaf_kind = "histogram_bar" if operator is RendererOperator.BAR_CHART else "equation" if operator is RendererOperator.EQUATION else "text" if operator is RendererOperator.CODE_TRACE else "component"
        children = [self._leaf(parameters.object_id, leaf_kind, index, label, f"{operator.value}_operand", {"value": parameters.values[index] if index < len(parameters.values) else (index + 1) / len(labels)}) for index, label in enumerate(labels)]
        if operator in {RendererOperator.PROCESS, RendererOperator.CAUSE_EFFECT, RendererOperator.TIMELINE, RendererOperator.FLOWCHART, RendererOperator.CYCLE}:
            children.extend(self._connectors(parameters.object_id, len(labels), close_cycle=operator is RendererOperator.CYCLE))
        layout = "vertical" if operator in {RendererOperator.LAYERED, RendererOperator.FUNNEL, RendererOperator.EQUATION, RendererOperator.CODE_TRACE} else "graph" if operator in {RendererOperator.CYCLE, RendererOperator.FLOWCHART, RendererOperator.VENN, RendererOperator.LINE_CHART} else "horizontal"
        return self._container(parameters, root_kind, children, layout, {"operator": operator.value, "active_index": parameters.active_index})

    def _graph_container(self, parameters: OperatorParameters, kind: str, labels: list[str], edges: list[tuple[int, int]], content: dict[str, object]) -> VisualObjectSpec:
        children = [self._leaf(parameters.object_id, "graph_node", index, label, "graph_vertex") for index, label in enumerate(labels)]
        for index, (source, target) in enumerate(edges):
            if source >= len(labels) or target >= len(labels):
                continue
            children.append(self._connector(parameters.object_id, index, source, target, "graph_edge"))
        return self._container(parameters, kind, children, "graph", content)

    @staticmethod
    def _leaf(object_id: str, kind: str, index: int, label: str, role: str, extra: dict[str, object] | None = None) -> VisualObjectSpec:
        return VisualObjectSpec(object_id=f"{object_id}_item_{index:03d}", kind=kind, semantic_role=role, content={"label": label, "order": index, **(extra or {})}, accessibility_label=label)

    def _connectors(self, object_id: str, count: int, role: str = "flow", close_cycle: bool = False) -> list[VisualObjectSpec]:
        limit = count if close_cycle and count > 2 else max(0, count - 1)
        return [self._connector(object_id, index, index, (index + 1) % count, role) for index in range(limit)]

    @staticmethod
    def _connector(object_id: str, index: int, source: int, target: int, role: str) -> VisualObjectSpec:
        return VisualObjectSpec(object_id=f"{object_id}_connector_{index:03d}", kind="connector", semantic_role=role, content={"source_id": f"{object_id}_item_{source:03d}", "target_id": f"{object_id}_item_{target:03d}", "label": ""}, accessibility_label=f"{role} from item {source} to item {target}")

    @staticmethod
    def _container(parameters: OperatorParameters, kind: str, children: list[VisualObjectSpec], layout: str, content: dict[str, object]) -> VisualObjectSpec:
        child_ids = [child.object_id for child in children]
        connector_mode = (
            "explicit" if any(child.kind == "connector" for child in children)
            else "none"
        )
        return VisualObjectSpec(
            object_id=parameters.object_id, kind=kind, semantic_role="procedural_operator",
            content={"label": parameters.label, "layout": layout, "connector_mode": connector_mode, **content}, children=children,
            constraints=[LayoutConstraint(constraint_id=f"{parameters.object_id}_contain", type=ConstraintType.CONTAIN, subject_ids=child_ids, reference_id=parameters.object_id, strength=ConstraintStrength.REQUIRED), LayoutConstraint(constraint_id=f"{parameters.object_id}_layout", type=ConstraintType.DISTRIBUTE, subject_ids=[child.object_id for child in children if child.kind != "connector"], reference_id=parameters.object_id, parameters={"axis": layout, "gap": 32})],
            accessibility_label=f"{parameters.label} procedural {content.get('operator', kind)} operator",
        )


def operator_template(
    template_id: str,
    keywords: set[str],
    operator: RendererOperator,
    label: str,
    metadata: tuple[str, ...],
) -> OperatorTemplate:
    """Create one reviewed operator binding with its validated schema."""

    return OperatorTemplate(
        template_id=template_id,
        keywords=frozenset(keywords),
        operator=operator,
        default_label=label,
        metadata=metadata,
        parameter_model=PARAMETER_MODELS.get(operator, GenericOperatorParameters),
    )
