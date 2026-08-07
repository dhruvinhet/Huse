"""Semantic implementations of the initial educational template set."""

from dataclasses import dataclass, field

from app.domain.layout import (
    ConstraintStrength,
    ConstraintType,
    LayoutConstraint,
)
from app.domain.storyboard import VisualObjectSpec
from app.domain.visual_intent import RendererOperator
from app.domain.semantic_bounds import bound_semantic_operands
from app.templates.operator_templates import (
    GenericOperatorParameters,
    OperatorTemplate,
    SemanticOperatorCompiler,
    operator_template,
    reviewed_operator_capabilities,
)
from app.domain.strategy import TemplateCapabilities


def _identifier(parameters: dict[str, object], fallback: str) -> str:
    """Return a normalized template object ID."""

    value = str(parameters.get("object_id", fallback)).strip()
    if not value:
        raise ValueError("template object_id cannot be empty")
    return value


def _strings(value: object, fallback: list[str]) -> list[str]:
    """Normalize a parameter to a non-empty string list."""

    if not isinstance(value, list):
        return fallback
    normalized = [str(item).strip() for item in value if str(item).strip()]
    return normalized or fallback


def _label_object(object_id: str, text: str, role: str = "label") -> VisualObjectSpec:
    """Create one semantic label child."""

    return VisualObjectSpec(
        object_id=object_id,
        kind="label",
        semantic_role=role,
        content={"text": text},
        accessibility_label=text,
    )


def _container(
    object_id: str,
    kind: str,
    label: str,
    children: list[VisualObjectSpec],
    layout: str,
) -> VisualObjectSpec:
    """Create a labeled hierarchical semantic container."""

    child_ids = [child.object_id for child in children]
    constraints = []
    if child_ids:
        constraints.append(
            LayoutConstraint(
                constraint_id=f"{object_id}_contain",
                type=ConstraintType.CONTAIN,
                subject_ids=child_ids,
                reference_id=object_id,
                strength=ConstraintStrength.REQUIRED,
            )
        )
        constraints.append(
            LayoutConstraint(
                constraint_id=f"{object_id}_{layout}",
                type=ConstraintType.DISTRIBUTE,
                subject_ids=child_ids,
                reference_id=object_id,
                parameters={"axis": layout, "gap": 32},
            )
        )
    operator = {
        "array": RendererOperator.ARRAY.value,
        "graph": RendererOperator.GRAPH.value,
        "tree": RendererOperator.TREE.value,
        "matrix": RendererOperator.MATRIX.value,
        "pipeline": RendererOperator.FLOW.value,
        "probability_distribution": RendererOperator.BAR_CHART.value,
    }.get(kind, kind)
    return VisualObjectSpec(
        object_id=object_id,
        kind=kind,
        semantic_role="educational_diagram",
        content={
            "label": label,
            "layout": layout,
            "operator": operator,
            "dsl_version": "1.0",
        },
        style_token="concept.primary",
        children=children,
        constraints=constraints,
        accessibility_label=label,
    )


@dataclass(frozen=True, slots=True)
class ArrayTemplate:
    """Create a semantic array with labeled cells."""

    template_id: str = "array.v1"
    keywords: frozenset[str] = frozenset({"array", "binary", "search", "sort"})
    capabilities: TemplateCapabilities = field(
        default_factory=lambda: reviewed_operator_capabilities(RendererOperator.ARRAY)
    )

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build an array hierarchy."""

        object_id = _identifier(parameters, "Array_001")
        values = _strings(parameters.get("values"), ["2", "5", "8", "12", "16"])
        children = [
            VisualObjectSpec(
                object_id=f"{object_id}_cell_{index:03d}",
                kind="array_cell",
                semantic_role="array_value",
                content={"value": value, "index": index},
                style_token="data.input",
                accessibility_label=f"Array index {index}, value {value}",
            )
            for index, value in enumerate(values)
        ]
        return _container(
            object_id,
            "array",
            str(parameters.get("label", "Array")),
            children,
            "horizontal",
        )


@dataclass(frozen=True, slots=True)
class PipelineTemplate:
    """Create connected stages representing a process flow."""

    template_id: str = "pipeline.v1"
    keywords: frozenset[str] = frozenset(
        {"pipeline", "flow", "process", "api", "request", "tokenization"}
    )
    capabilities: TemplateCapabilities = field(
        default_factory=lambda: reviewed_operator_capabilities(RendererOperator.FLOW)
    )

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build stages and semantic connectors."""

        object_id = _identifier(parameters, "Pipeline_001")
        stages = _strings(parameters.get("stages"), ["Input", "Process", "Output"])
        if isinstance(parameters.get("concepts"), list) and parameters["concepts"]:
            bounded = bound_semantic_operands(parameters)
            validated = GenericOperatorParameters.model_validate({
                "object_id": object_id,
                "label": str(bounded.get("label", "Pipeline")),
                "operands": stages[:10],
                "concepts": bounded.get("concepts", []),
                "relations": bounded.get("relations", []),
            })
            operator = (
                RendererOperator.FLOW
                if parameters.get("dsl_version") == "1.0"
                else RendererOperator.PROCESS
            )
            return SemanticOperatorCompiler().compile(operator, validated)
        raw_details = parameters.get("stage_details")
        details = raw_details if isinstance(raw_details, list) else []
        children: list[VisualObjectSpec] = []
        stage_ids: list[str] = []
        for index, stage in enumerate(stages):
            stage_id = f"{object_id}_stage_{index:03d}"
            stage_ids.append(stage_id)
            children.append(
                VisualObjectSpec(
                    object_id=stage_id,
                    kind="component",
                    semantic_role="pipeline_stage",
                    content={
                        "label": stage,
                        "order": index,
                        "detail": (
                            str(details[index].get("detail", "")).strip()
                            if index < len(details)
                            and isinstance(details[index], dict)
                            else ""
                        ),
                    },
                    accessibility_label=f"Pipeline stage {stage}",
                )
            )
        for index in range(len(stage_ids) - 1):
            relation = (
                details[index].get("relation_to_next")
                if index < len(details) and isinstance(details[index], dict)
                else "flows_to"
            )
            if not isinstance(relation, str) or not relation.strip():
                continue
            relation_label = (
                details[index].get("relation_label")
                if index < len(details) and isinstance(details[index], dict)
                else ""
            )
            children.append(
                VisualObjectSpec(
                    object_id=f"{object_id}_connector_{index:03d}",
                    kind="connector",
                    semantic_role="flow",
                    content={
                        "source_id": stage_ids[index],
                        "target_id": stage_ids[index + 1],
                        "relation": relation,
                        "label": (
                            str(relation_label).strip()
                            if relation_label is not None
                            else ""
                        ),
                    },
                    style_token="process.active",
                    accessibility_label=(
                        f"Flow from {stages[index]} to {stages[index + 1]}"
                    ),
                )
            )
        root = _container(
            object_id,
            "pipeline",
            str(parameters.get("label", "Pipeline")),
            children,
            "horizontal",
        )
        if any(child.kind == "connector" for child in children):
            root.content["connector_mode"] = "explicit"
        return root


@dataclass(frozen=True, slots=True)
class TreeTemplate:
    """Create a small labeled hierarchy suitable for tree algorithms."""

    template_id: str = "tree.v1"
    keywords: frozenset[str] = frozenset({"tree", "dfs", "bfs", "hierarchy"})
    capabilities: TemplateCapabilities = field(
        default_factory=lambda: reviewed_operator_capabilities(RendererOperator.TREE)
    )

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build a tree from level-order labels."""

        object_id = _identifier(parameters, "Tree_001")
        labels = _strings(parameters.get("nodes"), ["A", "B", "C", "D", "E"])
        children = [
            VisualObjectSpec(
                object_id=f"{object_id}_node_{index:03d}",
                kind="tree_node",
                semantic_role="tree_node",
                content={
                    "label": label,
                    "parent_index": None if index == 0 else (index - 1) // 2,
                },
                accessibility_label=f"Tree node {label}",
            )
            for index, label in enumerate(labels)
        ]
        node_ids = [item.object_id for item in children]
        children.extend(
            VisualObjectSpec(
                object_id=f"{object_id}_edge_{index:03d}",
                kind="connector",
                semantic_role="parent_child_edge",
                content={
                    "source_id": node_ids[(index - 1) // 2],
                    "target_id": node_ids[index],
                    "relation": "parent_child",
                    "label": "",
                },
                accessibility_label=(
                    f"Parent-child edge from "
                    f"{labels[(index - 1) // 2]} to {labels[index]}"
                ),
            )
            for index in range(1, len(node_ids))
        )
        return _container(
            object_id,
            "tree",
            str(parameters.get("label", "Tree")),
            children,
            "tree",
        )


@dataclass(frozen=True, slots=True)
class GraphTemplate:
    """Create a semantic graph with node and edge records."""

    template_id: str = "graph.v1"
    keywords: frozenset[str] = frozenset({"graph", "network", "nodes", "edges"})
    capabilities: TemplateCapabilities = field(
        default_factory=lambda: reviewed_operator_capabilities(RendererOperator.GRAPH)
    )

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build a graph hierarchy with deterministic default edges."""

        object_id = _identifier(parameters, "Graph_001")
        labels = _strings(parameters.get("nodes"), ["A", "B", "C", "D"])
        node_ids = [f"{object_id}_node_{index:03d}" for index in range(len(labels))]
        children = [
            VisualObjectSpec(
                object_id=node_id,
                kind="graph_node",
                semantic_role="graph_vertex",
                content={"label": labels[index]},
                accessibility_label=f"Graph node {labels[index]}",
            )
            for index, node_id in enumerate(node_ids)
        ]
        for index in range(max(0, len(node_ids) - 1)):
            children.append(
                VisualObjectSpec(
                    object_id=f"{object_id}_edge_{index:03d}",
                    kind="connector",
                    semantic_role="graph_edge",
                    content={
                        "source_id": node_ids[index],
                        "target_id": node_ids[index + 1],
                    },
                    accessibility_label=(
                        f"Edge from {labels[index]} to {labels[index + 1]}"
                    ),
                )
            )
        return _container(
            object_id,
            "graph",
            str(parameters.get("label", "Graph")),
            children,
            "graph",
        )


@dataclass(frozen=True, slots=True)
class MatrixTemplate:
    """Create a labeled matrix with semantic cells."""

    template_id: str = "matrix.v1"
    keywords: frozenset[str] = frozenset({"matrix", "attention", "grid"})
    capabilities: TemplateCapabilities = field(
        default_factory=lambda: reviewed_operator_capabilities(RendererOperator.MATRIX)
    )

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build a matrix hierarchy."""

        object_id = _identifier(parameters, "Matrix_001")
        rows = int(parameters.get("rows", 3))
        columns = int(parameters.get("columns", 3))
        if rows < 1 or columns < 1 or rows * columns > 100:
            raise ValueError("matrix dimensions must contain 1 to 100 cells")
        children = [
            VisualObjectSpec(
                object_id=f"{object_id}_cell_{row:03d}_{column:03d}",
                kind="matrix_cell",
                semantic_role="matrix_value",
                content={"row": row, "column": column, "value": ""},
                accessibility_label=f"Matrix cell row {row}, column {column}",
            )
            for row in range(rows)
            for column in range(columns)
        ]
        return _container(
            object_id,
            "matrix",
            str(parameters.get("label", "Matrix")),
            children,
            "grid",
        )


@dataclass(frozen=True, slots=True)
class ProbabilityDistributionTemplate:
    """Create a labeled probability distribution."""

    template_id: str = "probability_distribution.v1"
    keywords: frozenset[str] = frozenset(
        {"probability", "distribution", "softmax", "histogram"}
    )
    capabilities: TemplateCapabilities = field(
        default_factory=lambda: reviewed_operator_capabilities(
            RendererOperator.BAR_CHART
        )
    )

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build axes and semantic probability bars."""

        object_id = _identifier(parameters, "Distribution_001")
        labels = _strings(parameters.get("labels"), ["A", "B", "C", "D"])
        values_source = parameters.get("values", [0.1, 0.2, 0.6, 0.1])
        if not isinstance(values_source, list) or len(values_source) != len(labels):
            raise ValueError("probability labels and values must have equal length")
        values = [float(value) for value in values_source]
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("probabilities must be between zero and one")
        children = [
            VisualObjectSpec(
                object_id=f"{object_id}_bar_{index:03d}",
                kind="histogram_bar",
                semantic_role="probability",
                content={"label": label, "value": values[index]},
                style_token="data.output",
                accessibility_label=f"{label} probability {values[index]:.2f}",
            )
            for index, label in enumerate(labels)
        ]
        return _container(
            object_id,
            "probability_distribution",
            str(parameters.get("label", "Probability Distribution")),
            children,
            "horizontal",
        )


@dataclass(frozen=True, slots=True)
class TransformerBlockTemplate:
    """Create a meaningful nested Transformer block diagram."""

    template_id: str = "transformer_block.v1"
    keywords: frozenset[str] = frozenset(
        {"transformer", "attention", "embedding", "llm", "language"}
    )
    capabilities: TemplateCapabilities = field(
        default_factory=lambda: reviewed_operator_capabilities(
            RendererOperator.NEURAL_NETWORK
        )
    )

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Compile explicit neural layers and tensor-flow state."""

        return operator_template(
            self.template_id,
            set(self.keywords),
            RendererOperator.NEURAL_NETWORK,
            "Transformer Block",
            (
                "Embedding",
                "Multi-head attention",
                "Residual normalization",
                "Feed-forward network",
            ),
        ).instantiate(parameters)


def topic_templates() -> list[OperatorTemplate]:
    """Return reviewed topic families backed by procedural operators."""

    definitions = [
        ("binary_search.v1", {"binary", "search"}, RendererOperator.BINARY_SEARCH, "Binary Search", ("sorted values", "low high mid pointers", "comparison", "code trace")),
        ("quick_sort.v1", {"quick", "quicksort", "sort"}, RendererOperator.SORTING, "Quick Sort", ("values", "pivot", "partition range")),
        ("merge_sort.v1", {"merge", "mergesort", "sort"}, RendererOperator.SORTING, "Merge Sort", ("values", "active ranges", "merge phase")),
        ("dfs.v1", {"dfs", "depth", "first"}, RendererOperator.GRAPH_TRAVERSAL, "Depth-First Search", ("nodes", "edges", "stack frontier", "visited")),
        ("bfs.v1", {"bfs", "breadth", "first"}, RendererOperator.GRAPH_TRAVERSAL, "Breadth-First Search", ("nodes", "edges", "queue frontier", "visited")),
        ("cnn.v1", {"cnn", "convolutional"}, RendererOperator.NEURAL_NETWORK, "Convolutional Neural Network", ("image tensor", "convolution", "pooling", "classifier")),
        ("rnn.v1", {"rnn", "recurrent"}, RendererOperator.NEURAL_NETWORK, "Recurrent Neural Network", ("input timestep", "hidden state", "recurrent edge", "output")),
        ("tcp_handshake.v1", {"tcp", "handshake"}, RendererOperator.PROTOCOL, "TCP Handshake", ("client server", "SYN", "SYN-ACK", "ACK")),
        ("rest_api.v1", {"rest", "api"}, RendererOperator.PROTOCOL, "REST API", ("client server", "HTTP request", "handler", "JSON response")),
        ("microservices.v1", {"microservices", "service"}, RendererOperator.SYSTEM, "Microservices", ("gateway", "services", "data stores", "typed edges")),
        ("database_index.v1", {"database", "index", "btree"}, RendererOperator.TREE_INDEX, "Database Index", ("root page", "index pages", "leaf page", "lookup path")),
        ("os_scheduling.v1", {"scheduling", "scheduler", "process"}, RendererOperator.SCHEDULING, "OS Scheduling", ("ready queue", "CPU", "waiting", "completion timeline")),
        ("memory_allocation.v1", {"memory", "allocation", "heap"}, RendererOperator.MEMORY_MAP, "Memory Allocation", ("addresses", "code", "stack", "free space", "heap")),
        ("hash_map.v1", {"hash", "map", "table"}, RendererOperator.HASH_MAP, "Hash Map", ("key", "hash value", "bucket index", "stored value")),
        ("blockchain.v1", {"blockchain", "block", "ledger"}, RendererOperator.BLOCKCHAIN, "Blockchain", ("blocks", "previous hashes", "active block", "chain validity")),
        ("authentication.v1", {"authentication", "auth", "login"}, RendererOperator.PROTOCOL, "Authentication", ("client resource", "credentials", "identity check", "token")),
        ("system_design.v1", {"system", "design", "architecture"}, RendererOperator.SYSTEM, "System Design", ("client", "load balancer", "services", "data stores")),
    ]
    return [operator_template(*definition) for definition in definitions]


def educational_templates() -> list[OperatorTemplate]:
    """Return reusable educational grammars as procedural operators."""

    definitions = [
        ("before_after.v1", {"before", "after", "change", "improve"}, RendererOperator.COMPARISON, "Before and After", ("before state", "change evidence", "after state")),
        ("comparison.v1", {"compare", "versus", "difference", "tradeoff"}, RendererOperator.COMPARISON, "Comparison", ("option A", "shared criteria", "option B")),
        ("concept_set.v1", {"collection", "framework", "principle", "rule", "law", "type", "category"}, RendererOperator.COMPARISON, "Concept Collection", ("co-equal members", "defining evidence", "scope", "shared framework")),
        ("timeline.v1", {"timeline", "history", "sequence", "stages"}, RendererOperator.TIMELINE, "Timeline", ("events", "dates", "turning points")),
        ("cycle.v1", {"cycle", "loop", "repeat", "lifecycle"}, RendererOperator.CYCLE, "Cycle", ("states", "feedback edge", "repeat condition")),
        ("cause_effect.v1", {"cause", "effect", "impact", "because"}, RendererOperator.CAUSE_EFFECT, "Cause and Effect", ("cause", "mechanism", "effect")),
        ("input_output.v1", {"input", "output", "function", "mapping"}, RendererOperator.PROCESS, "Input to Output", ("input", "transformation", "output")),
        ("layered_architecture.v1", {"layer", "architecture", "stack", "tier"}, RendererOperator.LAYERED, "Layered Architecture", ("interface layer", "logic layer", "data layer")),
        ("flowchart.v1", {"flowchart", "decision", "branch", "condition"}, RendererOperator.FLOWCHART, "Decision Flow", ("input", "decision", "true branch", "false branch")),
        ("funnel.v1", {"funnel", "filter", "conversion", "narrow"}, RendererOperator.FUNNEL, "Funnel", ("all inputs", "filter stages", "selected result")),
        ("venn.v1", {"venn", "overlap", "intersection", "union"}, RendererOperator.VENN, "Set Relationship", ("set A", "intersection", "set B")),
        ("bar_chart.v1", {"bar", "chart", "compare", "quantity"}, RendererOperator.BAR_CHART, "Bar Chart", ("categories", "numeric values", "baseline")),
        ("line_chart.v1", {"line", "trend", "growth", "over time"}, RendererOperator.LINE_CHART, "Trend", ("time points", "values", "trend")),
        ("equation_derivation.v1", {"equation", "derive", "formula", "proof"}, RendererOperator.EQUATION, "Equation Derivation", ("known expression", "substitution", "simplification", "result")),
        ("code_trace.v1", {"code", "trace", "execute", "debug"}, RendererOperator.CODE_TRACE, "Code Trace", ("active statement", "variable state", "control flow", "result")),
    ]
    return [operator_template(*definition) for definition in definitions]
