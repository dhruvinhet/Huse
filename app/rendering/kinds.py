"""Extensible semantic visual-language registry."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SemanticKind:
    """Describe one first-class visual kind and renderer capability."""

    name: str
    category: str
    default_animation: str
    supports_children: bool = False


class SemanticKindRegistry:
    """Register the supported rich visual language explicitly."""

    def __init__(self) -> None:
        """Register the built-in semantic vocabulary."""

        self._kinds: dict[str, SemanticKind] = {}
        for kind in self._builtins():
            self.register(kind)

    def register(self, kind: SemanticKind) -> None:
        """Register a unique semantic kind."""

        if not kind.name.strip() or not kind.category.strip():
            raise ValueError("semantic kinds require name and category")
        if kind.name in self._kinds:
            raise ValueError(f"semantic kind already registered: {kind.name}")
        self._kinds[kind.name] = kind

    def get(self, name: str) -> SemanticKind:
        """Return a registered kind or fail before rendering."""

        try:
            return self._kinds[name]
        except KeyError as exc:
            raise KeyError(f"unsupported semantic kind: {name}") from exc

    def names(self) -> list[str]:
        """Return supported names in deterministic order."""

        return sorted(self._kinds)

    @staticmethod
    def _builtins() -> list[SemanticKind]:
        """Return the initial first-class visual language."""

        containers = {
            "array", "linked_list", "tree", "graph", "timeline", "pipeline",
            "stack", "queue", "hash_table", "flowchart", "decision_tree",
            "table", "matrix", "probability_distribution", "histogram",
            "pie_chart", "coordinate_axes", "nested_group", "component",
            "transformer_block", "attention_matrix", "embedding_vector",
            "neural_network", "blockchain", "document", "flow", "molecule",
            "circuit", "map", "anatomy", "transform", "icon", "layered",
            "funnel", "venn",
        }
        text = {"text", "label", "annotation", "equation", "callout", "speech_bubble"}
        connectors = {"connector", "brace", "bracket", "underline", "highlight"}
        assets = {"database", "server", "browser", "phone", "cloud", "cpu", "gpu", "memory", "semantic_asset"}
        leaves = {"token_chip", "array_cell", "matrix_cell", "tree_node", "graph_node", "histogram_bar", "legacy_svg"}
        kinds = [
            SemanticKind(name, "container", "progressive_reveal", True)
            for name in containers
        ]
        kinds.extend(SemanticKind(name, "text", "handwriting") for name in text)
        kinds.extend(SemanticKind(name, "connector", "grow_edge") for name in connectors)
        kinds.extend(SemanticKind(name, "asset", "stroke_reveal") for name in assets)
        kinds.extend(SemanticKind(name, "primitive", "reveal") for name in leaves)
        return kinds
