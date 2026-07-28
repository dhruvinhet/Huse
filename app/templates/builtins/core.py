"""Semantic implementations of the initial educational template set."""

from dataclasses import dataclass

from app.domain.layout import (
    ConstraintStrength,
    ConstraintType,
    LayoutConstraint,
)
from app.domain.storyboard import VisualObjectSpec


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
    return VisualObjectSpec(
        object_id=object_id,
        kind=kind,
        semantic_role="educational_diagram",
        content={"label": label, "layout": layout},
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

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build stages and semantic connectors."""

        object_id = _identifier(parameters, "Pipeline_001")
        stages = _strings(parameters.get("stages"), ["Input", "Process", "Output"])
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
                    content={"label": stage, "order": index},
                    accessibility_label=f"Pipeline stage {stage}",
                )
            )
        for index in range(len(stage_ids) - 1):
            children.append(
                VisualObjectSpec(
                    object_id=f"{object_id}_connector_{index:03d}",
                    kind="connector",
                    semantic_role="flow",
                    content={
                        "source_id": stage_ids[index],
                        "target_id": stage_ids[index + 1],
                        "label": "",
                    },
                    style_token="process.active",
                    accessibility_label=(
                        f"Flow from {stages[index]} to {stages[index + 1]}"
                    ),
                )
            )
        return _container(
            object_id,
            "pipeline",
            str(parameters.get("label", "Pipeline")),
            children,
            "horizontal",
        )


@dataclass(frozen=True, slots=True)
class TreeTemplate:
    """Create a small labeled hierarchy suitable for tree algorithms."""

    template_id: str = "tree.v1"
    keywords: frozenset[str] = frozenset({"tree", "dfs", "bfs", "hierarchy"})

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

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build nested attention, normalization, and feed-forward components."""

        object_id = _identifier(parameters, "TransformerBlock_001")
        components = [
            ("attention", "Multi-Head Attention"),
            ("residual_1", "Residual + LayerNorm"),
            ("feed_forward", "Feed-Forward Network"),
            ("residual_2", "Residual + LayerNorm"),
        ]
        children = [
            VisualObjectSpec(
                object_id=f"{object_id}_{suffix}",
                kind="component",
                semantic_role=suffix,
                content={"label": label, "order": index},
                style_token=(
                    "process.active" if suffix in {"attention", "feed_forward"}
                    else "annotation"
                ),
                accessibility_label=label,
            )
            for index, (suffix, label) in enumerate(components)
        ]
        return _container(
            object_id,
            "transformer_block",
            str(parameters.get("label", "Transformer Block")),
            children,
            "vertical",
        )


@dataclass(frozen=True, slots=True)
class DeclarativeTopicTemplate:
    """Create a professional labeled diagram from a reviewed topic recipe."""

    template_id: str
    keywords: frozenset[str]
    diagram_kind: str
    default_label: str
    components: tuple[str, ...]
    layout: str = "horizontal"

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build reviewed components and explicit flow connectors."""

        object_id = _identifier(
            parameters,
            self.template_id.replace(".", "_").title(),
        )
        labels = _strings(parameters.get("components"), list(self.components))
        children: list[VisualObjectSpec] = []
        component_ids: list[str] = []
        for index, label in enumerate(labels):
            child_id = f"{object_id}_component_{index:03d}"
            component_ids.append(child_id)
            children.append(
                VisualObjectSpec(
                    object_id=child_id,
                    kind=(
                        "array_cell" if self.diagram_kind in {"array", "hash_table"}
                        else "tree_node" if self.diagram_kind == "tree"
                        else "graph_node" if self.diagram_kind == "graph"
                        else "component"
                    ),
                    semantic_role=f"{self.template_id}_step",
                    content={"label": label, "order": index},
                    accessibility_label=label,
                )
            )
        if self.diagram_kind in {
            "pipeline", "graph", "blockchain", "linked_list", "timeline",
            "cycle", "cause_effect", "flowchart", "architecture",
            "equation_derivation", "code_trace",
        }:
            for index in range(len(component_ids) - 1):
                children.append(
                    VisualObjectSpec(
                        object_id=f"{object_id}_connector_{index:03d}",
                        kind="connector",
                        semantic_role="flow",
                        content={
                            "source_id": component_ids[index],
                            "target_id": component_ids[index + 1],
                            "label": "",
                        },
                        style_token="process.active",
                        accessibility_label=(
                            f"Flow from {labels[index]} to {labels[index + 1]}"
                        ),
                    )
                )
        return _container(
            object_id,
            self.diagram_kind,
            str(parameters.get("label", self.default_label)),
            children,
            self.layout,
        )


def topic_templates() -> list[DeclarativeTopicTemplate]:
    """Return reviewed templates for the initial educational topic library."""

    definitions = [
        ("binary_search.v1", {"binary", "search"}, "array", "Binary Search", ("Low", "Middle", "High"), "horizontal"),
        ("quick_sort.v1", {"quick", "quicksort", "sort"}, "array", "Quick Sort", ("Partition", "Pivot", "Left", "Right"), "horizontal"),
        ("merge_sort.v1", {"merge", "mergesort", "sort"}, "tree", "Merge Sort", ("Input", "Split", "Subarrays", "Merge"), "tree"),
        ("dfs.v1", {"dfs", "depth", "first"}, "graph", "Depth-First Search", ("Start", "Explore", "Backtrack", "Complete"), "graph"),
        ("bfs.v1", {"bfs", "breadth", "first"}, "graph", "Breadth-First Search", ("Start", "Queue", "Frontier", "Visited"), "graph"),
        ("cnn.v1", {"cnn", "convolutional"}, "pipeline", "Convolutional Neural Network", ("Image", "Convolution", "Pooling", "Classifier"), "horizontal"),
        ("rnn.v1", {"rnn", "recurrent"}, "pipeline", "Recurrent Neural Network", ("Input t", "Hidden State", "Input t+1", "Output"), "horizontal"),
        ("tcp_handshake.v1", {"tcp", "handshake"}, "timeline", "TCP Handshake", ("SYN", "SYN-ACK", "ACK", "Connected"), "horizontal"),
        ("rest_api.v1", {"rest", "api"}, "pipeline", "REST API", ("Client", "HTTP Request", "Server", "JSON Response"), "horizontal"),
        ("microservices.v1", {"microservices", "service"}, "graph", "Microservices", ("Gateway", "Service A", "Service B", "Database"), "graph"),
        ("database_index.v1", {"database", "index", "btree"}, "tree", "Database Index", ("Root Page", "Index Pages", "Leaf Pages", "Rows"), "tree"),
        ("os_scheduling.v1", {"scheduling", "scheduler", "process"}, "timeline", "OS Scheduling", ("Ready", "Running", "Waiting", "Complete"), "horizontal"),
        ("memory_allocation.v1", {"memory", "allocation", "heap"}, "array", "Memory Allocation", ("Code", "Stack", "Free", "Heap"), "vertical"),
        ("hash_map.v1", {"hash", "map", "table"}, "hash_table", "Hash Map", ("Key", "Hash", "Bucket", "Value"), "horizontal"),
        ("blockchain.v1", {"blockchain", "block", "ledger"}, "blockchain", "Blockchain", ("Block 1", "Block 2", "Block 3", "Consensus"), "horizontal"),
        ("authentication.v1", {"authentication", "auth", "login"}, "pipeline", "Authentication", ("Credentials", "Identity Check", "Token", "Protected Resource"), "horizontal"),
        ("system_design.v1", {"system", "design", "architecture"}, "graph", "System Design", ("Client", "Load Balancer", "Services", "Data Stores"), "graph"),
    ]
    return [
        DeclarativeTopicTemplate(
            template_id=template_id,
            keywords=frozenset(keywords),
            diagram_kind=kind,
            default_label=label,
            components=components,
            layout=layout,
        )
        for template_id, keywords, kind, label, components, layout in definitions
    ]


def educational_templates() -> list[DeclarativeTopicTemplate]:
    """Return reusable visual grammars that work across subject areas."""

    definitions = [
        ("before_after.v1", {"before", "after", "change", "improve"}, "comparison", "Before and After", ("Before", "Change", "After"), "horizontal"),
        ("comparison.v1", {"compare", "versus", "difference", "tradeoff"}, "comparison", "Comparison", ("Option A", "Criteria", "Option B"), "horizontal"),
        ("timeline.v1", {"timeline", "history", "sequence", "stages"}, "timeline", "Timeline", ("Beginning", "Development", "Result"), "horizontal"),
        ("cycle.v1", {"cycle", "loop", "repeat", "lifecycle"}, "cycle", "Cycle", ("Start", "Process", "Feedback", "Repeat"), "graph"),
        ("cause_effect.v1", {"cause", "effect", "impact", "because"}, "cause_effect", "Cause and Effect", ("Cause", "Mechanism", "Effect"), "horizontal"),
        ("input_output.v1", {"input", "output", "function", "mapping"}, "pipeline", "Input to Output", ("Input", "Transformation", "Output"), "horizontal"),
        ("layered_architecture.v1", {"layer", "architecture", "stack", "tier"}, "architecture", "Layered Architecture", ("Interface", "Logic", "Data"), "vertical"),
        ("flowchart.v1", {"flowchart", "decision", "branch", "condition"}, "flowchart", "Decision Flow", ("Input", "Decision", "Path A", "Path B"), "graph"),
        ("funnel.v1", {"funnel", "filter", "conversion", "narrow"}, "funnel", "Funnel", ("All Inputs", "Filtered", "Selected"), "vertical"),
        ("venn.v1", {"venn", "overlap", "intersection", "union"}, "venn", "Set Relationship", ("Set A", "Shared", "Set B"), "horizontal"),
        ("bar_chart.v1", {"bar", "chart", "compare", "quantity"}, "bar_chart", "Bar Chart", ("Category A", "Category B", "Category C"), "horizontal"),
        ("line_chart.v1", {"line", "trend", "growth", "over time"}, "line_chart", "Trend", ("Start", "Middle", "End"), "horizontal"),
        ("equation_derivation.v1", {"equation", "derive", "formula", "proof"}, "equation_derivation", "Equation Derivation", ("Known", "Substitute", "Simplify", "Result"), "vertical"),
        ("code_trace.v1", {"code", "trace", "execute", "debug"}, "code_trace", "Code Trace", ("Statement", "State Change", "Next Step", "Result"), "vertical"),
    ]
    return [
        DeclarativeTopicTemplate(
            template_id=template_id,
            keywords=frozenset(keywords),
            diagram_kind=kind,
            default_label=label,
            components=components,
            layout=layout,
        )
        for template_id, keywords, kind, label, components, layout in definitions
    ]
