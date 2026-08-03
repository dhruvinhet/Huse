"""Fail-closed pixel plugins for semantic renderer operators and kinds."""

from collections.abc import Callable
from dataclasses import dataclass
from PIL import Image, ImageDraw

from app.domain.layout import LayoutBox
from app.domain.visual_document import ObjectLifecycle, ObjectState


class UnsupportedSemanticKindError(RuntimeError):
    """Raised when no explicit pixel implementation exists."""


@dataclass(slots=True)
class OperatorDrawingContext:
    """Provide shared vector primitives to one explicit renderer plugin."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    state: ObjectState
    box: LayoutBox
    boxes: dict[str, LayoutBox]
    coordinates: tuple[int, int, int, int]
    ink: tuple[int, int, int, int]
    fill: tuple[int, int, int, int]
    accent: tuple[int, int, int, int]
    highlight: tuple[int, int, int, int]
    stroke_width: int
    font_size: int
    label: str
    draw_text: Callable[
        [ImageDraw.ImageDraw, tuple[int, int, int, int], str, int, tuple[int, int, int, int]],
        None,
    ]


class _Plugin:
    """Internal renderer-plugin protocol."""

    def draw(self, context: OperatorDrawingContext) -> None:
        raise NotImplementedError


class TextPlugin(_Plugin):
    """Draw text, equations, annotations, and callouts distinctly."""

    def draw(self, context: OperatorDrawingContext) -> None:
        box = context.coordinates
        kind = context.state.kind
        if kind in {"callout", "speech_bubble"}:
            context.draw.rounded_rectangle(
                box,
                radius=18,
                fill=context.fill,
                outline=context.ink,
                width=context.stroke_width,
            )
            tail_x = box[0] + max(18, (box[2] - box[0]) // 5)
            context.draw.polygon(
                [(tail_x, box[3]), (tail_x + 22, box[3]), (tail_x + 8, box[3] + 16)],
                fill=context.fill,
                outline=context.ink,
            )
        context.draw_text(
            context.draw,
            box,
            context.label,
            context.font_size,
            context.ink,
        )
        if kind == "annotation":
            context.draw.line((box[0], box[3] - 4, box[2], box[3] - 4), fill=context.ink, width=3)
        elif kind == "equation":
            center = (box[1] + box[3]) // 2
            context.draw.line((box[0] + 8, center + 18, box[2] - 8, center + 18), fill=context.accent, width=2)


class CardPlugin(_Plugin):
    """Draw explicitly supported leaf kinds with distinct silhouettes."""

    def __init__(self, motif: str) -> None:
        self._motif = motif

    def draw(self, context: OperatorDrawingContext) -> None:
        box = context.coordinates
        draw = context.draw
        asset_slot = context.state.content.get("asset_slot") == "left"
        text_left = (
            box[0] + min(96, max(58, round((box[2] - box[0]) * 0.26)))
            if asset_slot
            else box[0]
        )
        if context.state.lifecycle is ObjectLifecycle.EMPHASIZED:
            expanded = (box[0] - 9, box[1] - 9, box[2] + 9, box[3] + 9)
            draw.rounded_rectangle(expanded, radius=20, fill=context.highlight)
        if self._motif in {"tree_node", "graph_node", "cloud"}:
            draw.ellipse(box, fill=context.fill, outline=context.ink, width=context.stroke_width)
            if self._motif == "graph_node":
                inner = (box[0] + 7, box[1] + 7, box[2] - 7, box[3] - 7)
                draw.ellipse(inner, outline=context.accent, width=2)
        elif self._motif in {"array_cell", "matrix_cell", "memory"}:
            draw.rectangle(box, fill=context.fill, outline=context.ink, width=context.stroke_width)
            index = context.state.content.get("bucket", context.state.content.get("order"))
            if index is not None:
                draw.text((box[0] + 5, box[1] + 3), str(index), fill=context.accent)
        elif self._motif == "token":
            draw.rounded_rectangle(box, radius=max(8, (box[3] - box[1]) // 2), fill=context.fill, outline=context.ink, width=context.stroke_width)
        elif self._motif == "database":
            draw.rectangle((box[0], box[1] + 12, box[2], box[3] - 12), fill=context.fill, outline=context.ink, width=context.stroke_width)
            draw.ellipse((box[0], box[1], box[2], box[1] + 24), fill=context.fill, outline=context.ink, width=context.stroke_width)
            draw.arc((box[0], box[3] - 24, box[2], box[3]), 0, 180, fill=context.ink, width=context.stroke_width)
        elif self._motif == "server":
            draw.rounded_rectangle(box, radius=8, fill=context.fill, outline=context.ink, width=context.stroke_width)
            for offset in (0.3, 0.55, 0.8):
                y = round(box[1] + (box[3] - box[1]) * offset)
                draw.line((box[0] + 12, y, box[2] - 12, y), fill=context.accent, width=3)
        elif self._motif == "browser":
            draw.rectangle(box, fill=context.fill, outline=context.ink, width=context.stroke_width)
            draw.line((box[0], box[1] + 22, box[2], box[1] + 22), fill=context.ink, width=2)
            for x in (box[0] + 10, box[0] + 24, box[0] + 38):
                draw.ellipse((x, box[1] + 7, x + 6, box[1] + 13), fill=context.accent)
        elif self._motif == "phone":
            draw.rounded_rectangle(box, radius=18, fill=context.fill, outline=context.ink, width=context.stroke_width)
            draw.line((box[0] + 20, box[3] - 14, box[2] - 20, box[3] - 14), fill=context.ink, width=3)
        elif self._motif == "chip":
            draw.rectangle(box, fill=context.fill, outline=context.ink, width=context.stroke_width)
            for index in range(1, 5):
                y = box[1] + index * (box[3] - box[1]) // 5
                draw.line((box[0] - 7, y, box[0], y), fill=context.ink, width=2)
                draw.line((box[2], y, box[2] + 7, y), fill=context.ink, width=2)
        else:
            draw.rounded_rectangle(box, radius=14, fill=context.fill, outline=context.ink, width=context.stroke_width)
        detail = context.state.content.get("detail")
        if isinstance(detail, str) and detail.strip():
            divider_y = box[1] + max(42, round((box[3] - box[1]) * 0.34))
            draw.line(
                (text_left + 10, divider_y, box[2] - 14, divider_y),
                fill=context.accent,
                width=2,
            )
            context.draw_text(
                draw,
                (text_left + 6, box[1] + 5, box[2] - 10, divider_y - 4),
                context.label,
                context.font_size,
                context.ink,
            )
            context.draw_text(
                draw,
                (text_left + 8, divider_y + 7, box[2] - 12, box[3] - 8),
                detail.strip(),
                max(16, context.font_size - 9),
                context.ink,
            )
        else:
            context.draw_text(
                draw,
                (text_left, box[1], box[2], box[3]),
                context.label,
                context.font_size,
                context.ink,
            )


class DecorationPlugin(_Plugin):
    """Draw explicit braces, brackets, underlines, and highlights."""

    def draw(self, context: OperatorDrawingContext) -> None:
        left, top, right, bottom = context.coordinates
        kind = context.state.kind
        if kind == "underline":
            context.draw.line((left, bottom - 3, right, bottom - 3), fill=context.ink, width=4)
        elif kind == "highlight":
            context.draw.rounded_rectangle((left, top, right, bottom), radius=12, fill=context.highlight)
        elif kind == "brace":
            middle = (top + bottom) // 2
            context.draw.arc((left, top, right, middle), 90, 270, fill=context.ink, width=3)
            context.draw.arc((left, middle, right, bottom), 90, 270, fill=context.ink, width=3)
        else:
            context.draw.line((left + 10, top, left, top, left, bottom, left + 10, bottom), fill=context.ink, width=3)


class HistogramBarPlugin(_Plugin):
    """Draw a quantitative bar from its declared normalized value."""

    def draw(self, context: OperatorDrawingContext) -> None:
        left, top, right, bottom = context.coordinates
        value = float(context.state.content.get("value", 0.5))
        value = max(0.0, min(1.0, value))
        fill_top = bottom - round((bottom - top) * value)
        context.draw.rectangle((left, top, right, bottom), outline=context.ink, width=context.stroke_width)
        context.draw.rectangle((left + 4, fill_top, right - 4, bottom - 4), fill=context.accent)
        context.draw_text(context.draw, (left, top, right, bottom), context.label, max(16, context.font_size - 4), context.ink)


class ContainerPlugin(_Plugin):
    """Draw a declared semantic container motif, never a generic fallback."""

    def __init__(self, motif: str) -> None:
        self._motif = motif

    def draw(self, context: OperatorDrawingContext) -> None:
        left, top, right, bottom = context.coordinates
        draw = context.draw
        width = max(1, right - left)
        height = max(1, bottom - top)
        if self._motif in {"array", "linked_list", "stack", "queue", "hash_table", "embedding"}:
            draw.line((left, top + 18, left, bottom - 8, left + 12, bottom - 8), fill=context.ink, width=4)
            draw.line((right, top + 18, right, bottom - 8, right - 12, bottom - 8), fill=context.ink, width=4)
        elif self._motif in {"tree", "graph", "network"}:
            child_boxes = [context.boxes[item] for item in context.state.child_ids if item in context.boxes]
            centers = [(box.x + box.width / 2, box.y + box.height / 2) for box in child_boxes]
            for index, center in enumerate(centers[1:], start=1):
                parent = centers[(index - 1) // 2] if self._motif == "tree" else centers[index - 1]
                draw.line((*parent, *center), fill=context.accent, width=3)
        elif self._motif in {"timeline", "protocol"}:
            y = top + height // 2
            draw.line((left + 24, y, right - 24, y), fill=context.ink, width=4)
            for child_id in context.state.child_ids:
                child = context.boxes.get(child_id)
                if child is not None:
                    x = round(child.x + child.width / 2)
                    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=context.accent)
        elif (
            self._motif in {"pipeline", "cause_effect"}
            and context.state.content.get("connector_mode") != "explicit"
        ):
            y = top + height // 2
            draw.line((left + 20, y, right - 28, y), fill=context.accent, width=8)
            draw.polygon([(right - 28, y - 14), (right - 8, y), (right - 28, y + 14)], fill=context.accent)
        elif self._motif in {"table", "matrix"}:
            draw.rectangle((left, top + 34, right, bottom), outline=context.ink, width=3)
            draw.line((left + width // 2, top + 34, left + width // 2, bottom), fill=context.accent, width=3)
        elif self._motif in {"histogram", "axes"}:
            draw.line((left + 30, top + 30, left + 30, bottom - 24, right - 18, bottom - 24), fill=context.ink, width=4)
        elif self._motif == "flowchart":
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            draw.polygon([(center_x, top + 26), (right - 24, center_y), (center_x, bottom - 20), (left + 24, center_y)], outline=context.accent)
        elif self._motif == "document":
            draw.rectangle((left, top, right, bottom), fill=context.fill, outline=context.ink, width=3)
            draw.line((left + 34, top + 28, left + 34, bottom - 18), fill=context.accent, width=3)
        elif self._motif == "blockchain":
            y = top + height // 2
            for index in range(4):
                x = left + 30 + index * max(30, (width - 60) // 4)
                draw.rectangle((x, y - 18, x + 28, y + 18), outline=context.ink, width=3)
                if index:
                    draw.line((x - 14, y, x, y), fill=context.accent, width=4)
        elif self._motif == "layered":
            if context.state.content.get("relation") == "contains":
                draw.rounded_rectangle(
                    (left, top + 48, right, bottom),
                    radius=18,
                    outline=context.accent,
                    width=3,
                )
                draw.text((left + 14, top + 36), "contains", fill=context.accent)
            else:
                for index in range(3):
                    y = top + 40 + index * max(26, (height - 60) // 3)
                    inset = index * 14
                    draw.rounded_rectangle((left + 20 + inset, y, right - 20 - inset, y + 22), radius=6, outline=context.accent, width=3)
        else:
            draw.rectangle((left, top, right, bottom), outline=context.ink, width=3)
        if context.label:
            context.draw_text(draw, (left + 10, top + 4, right - 10, min(bottom, top + 48)), context.label, context.font_size, context.ink)


class SemanticOperatorPlugin(_Plugin):
    """Draw one named operator's characteristic structure."""

    def __init__(self, operator: str) -> None:
        self.operator = operator

    def draw(self, context: OperatorDrawingContext) -> None:
        left, top, right, bottom = context.coordinates
        draw = context.draw
        width = right - left
        height = bottom - top
        operator = self.operator
        if operator == "array":
            ContainerPlugin("array").draw(context)
        elif operator == "tree":
            ContainerPlugin("tree").draw(context)
        elif operator == "graph":
            ContainerPlugin("graph").draw(context)
        elif operator == "binary_search":
            draw.line((left, bottom - 34, right, bottom - 34), fill=context.ink, width=3)
            count = max(1, len([item for item in context.state.child_ids if item in context.boxes]))
            for name, color in (("low", context.accent), ("mid", context.ink), ("high", context.accent)):
                index = int(context.state.content.get(name, 0))
                x = left + round((index + 0.5) * width / count)
                draw.polygon([(x, bottom - 12), (x - 8, bottom - 24), (x + 8, bottom - 24)], fill=color)
                draw.text((x - 12, bottom - 11), name, fill=color)
            code = str(context.state.content.get("active_code_line", ""))
            if code:
                context.draw_text(draw, (left + 12, top + 36, right - 12, top + 72), code, 18, context.ink)
        elif operator == "sorting":
            pivot = int(context.state.content.get("pivot_index", 0))
            x = left + round((pivot + 0.5) * width / max(1, len(context.state.child_ids)))
            draw.polygon([(x, top + 30), (x - 10, top + 12), (x + 10, top + 12)], fill=context.accent)
            draw.text((x + 12, top + 10), "pivot", fill=context.accent)
        elif operator in {"graph_traversal", "system"}:
            ContainerPlugin("graph").draw(context)
            status = "frontier: " + ", ".join(map(str, context.state.content.get("frontier", [])))
            draw.text((left + 14, bottom - 24), status, fill=context.accent)
        elif operator == "neural_network":
            for index in range(5):
                x = left + 24 + index * max(24, (width - 48) // 5)
                for row in range(3):
                    y = top + 50 + row * max(20, (height - 90) // 3)
                    draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=context.accent, width=2)
        elif operator in {"protocol", "scheduling"}:
            ContainerPlugin("protocol").draw(context)
            participants = context.state.content.get("participants", [])
            for index, participant in enumerate(participants if isinstance(participants, list) else []):
                x = left + (index + 1) * width // (len(participants) + 1)
                draw.line((x, top + 44, x, bottom - 16), fill=context.accent, width=2)
                draw.text((x - 22, top + 24), str(participant), fill=context.ink)
        elif operator in {"tree_index"}:
            ContainerPlugin("tree").draw(context)
        elif operator in {"memory_map", "hash_map"}:
            ContainerPlugin("array").draw(context)
            draw.text((left + 12, top + 36), f"{operator.replace('_', ' ')} state", fill=context.accent)
        elif operator == "blockchain":
            ContainerPlugin("blockchain").draw(context)
        elif operator in {"comparison"}:
            ContainerPlugin("table").draw(context)
        elif operator == "timeline":
            ContainerPlugin("timeline").draw(context)
        elif operator == "cycle":
            draw.ellipse((left + 30, top + 32, right - 30, bottom - 24), outline=context.accent, width=5)
            draw.polygon([(right - 32, top + height // 2), (right - 50, top + height // 2 - 10), (right - 50, top + height // 2 + 10)], fill=context.accent)
        elif operator in {"cause_effect", "process"}:
            ContainerPlugin("cause_effect").draw(context)
        elif operator in {"layered", "spatial"}:
            ContainerPlugin("layered").draw(context)
        elif operator == "flowchart":
            ContainerPlugin("flowchart").draw(context)
        elif operator == "funnel":
            for index in range(3):
                inset = 14 + index * 22
                y = top + 42 + index * max(24, (height - 70) // 3)
                draw.polygon([(left + inset, y), (right - inset, y), (right - inset - 14, y + 22), (left + inset + 14, y + 22)], outline=context.accent)
        elif operator == "venn":
            draw.ellipse((left + width // 6, top + 42, left + 2 * width // 3, bottom - 16), outline=context.accent, width=4)
            draw.ellipse((left + width // 3, top + 42, right - width // 6, bottom - 16), outline=context.ink, width=4)
        elif operator in {"bar_chart", "line_chart", "simulation"}:
            ContainerPlugin("axes").draw(context)
            if operator == "line_chart":
                points = [(left + 38 + index * max(20, (width - 70) // 3), bottom - 40 - round((0.25 + index * 0.2) * height)) for index in range(4)]
                draw.line(points, fill=context.accent, width=4)
        elif operator in {"equation", "proof_derivation"}:
            ContainerPlugin("document").draw(context)
            draw.text((left + 48, top + 52), "given  →  substitute  →  result", fill=context.accent)
        elif operator in {"code_trace"}:
            ContainerPlugin("document").draw(context)
            for index in range(4):
                y = top + 48 + index * 22
                draw.text((left + 46, y), f"{index + 1:>2}  state[{index}]", fill=context.ink if index else context.accent)
        elif operator == "semantic_structure":
            draw.rounded_rectangle(
                (left, top + 48, right, bottom),
                radius=20,
                outline=context.ink,
                width=3,
            )
            relations = context.state.content.get("relation_kinds", [])
            if (
                context.state.content.get("connector_mode") != "explicit"
                and isinstance(relations, list)
            ):
                legend = "  |  ".join(
                    str(item).replace("_", " ") for item in relations[:4]
                )
                if legend:
                    draw.text((left + 16, bottom - 24), legend, fill=context.accent)
        else:
            ContainerPlugin("layered").draw(context)
        if context.label:
            context.draw_text(draw, (left + 8, top + 2, right - 8, min(bottom, top + 42)), context.label, context.font_size, context.ink)


class OperatorRendererRegistry:
    """Resolve every declared operator/kind explicitly and fail closed."""

    def __init__(self) -> None:
        operator_names = {
            "process", "comparison", "timeline", "array", "tree", "graph",
            "equation", "code_trace", "simulation", "spatial", "binary_search",
            "sorting", "graph_traversal", "neural_network", "protocol", "system",
            "tree_index", "scheduling", "memory_map", "hash_map", "blockchain",
            "cycle", "cause_effect", "layered", "flowchart", "funnel", "venn",
            "bar_chart", "line_chart",
            "semantic_structure",
        }
        self._operators = {
            name: SemanticOperatorPlugin(name)
            for name in operator_names
        }
        self._kinds: dict[str, _Plugin] = {}
        self._register_kind(TextPlugin(), {"text", "label", "annotation", "equation", "callout", "speech_bubble"})
        self._register_kind(HistogramBarPlugin(), {"histogram_bar"})
        self._register_kind(DecorationPlugin(), {"brace", "bracket", "underline", "highlight"})
        motif_kinds = {
            "component": {"component"},
            "array_cell": {"array_cell"},
            "matrix_cell": {"matrix_cell"},
            "tree_node": {"tree_node"},
            "graph_node": {"graph_node"},
            "token": {"token_chip"},
            "database": {"database"},
            "server": {"server"},
            "browser": {"browser"},
            "phone": {"phone"},
            "cloud": {"cloud"},
            "chip": {"cpu", "gpu"},
            "memory": {"memory"},
            "component": {"component", "legacy_svg"},
        }
        for motif, kinds in motif_kinds.items():
            self._register_kind(CardPlugin(motif), kinds)
        container_motifs = {
            "array": {"array"},
            "linked_list": {"linked_list"},
            "tree": {"tree", "decision_tree"},
            "graph": {"graph"},
            "timeline": {"timeline"},
            "pipeline": {"pipeline", "transformer_block"},
            "stack": {"stack"},
            "queue": {"queue"},
            "hash_table": {"hash_table"},
            "flowchart": {"flowchart"},
            "table": {"table"},
            "matrix": {"matrix", "attention_matrix"},
            "histogram": {"probability_distribution", "histogram"},
            "axes": {"pie_chart", "coordinate_axes"},
            "layered": {"nested_group"},
            "pipeline": {"pipeline", "transformer_block"},
            "network": {"neural_network"},
            "blockchain": {"blockchain"},
            "document": {"document"},
            "array": {"array", "embedding_vector"},
        }
        for motif, kinds in container_motifs.items():
            self._register_kind(ContainerPlugin(motif), kinds)

    def draw(self, context: OperatorDrawingContext) -> None:
        """Draw through an operator plugin first, then a declared kind plugin."""

        operator = str(context.state.content.get("operator", "")).strip()
        plugin = self._operators.get(operator) if operator else None
        if plugin is None:
            plugin = self._kinds.get(context.state.kind)
        if plugin is None:
            raise UnsupportedSemanticKindError(
                f"no renderer plugin for semantic kind {context.state.kind!r}"
            )
        plugin.draw(context)

    def supports(self, kind: str) -> bool:
        """Return whether a declared kind has a concrete plugin or core path."""

        return kind in self._kinds or kind in {"connector", "semantic_asset"}

    def supports_operator(self, operator: str) -> bool:
        """Return whether a high-level operator has an explicit pixel plugin."""

        return operator in self._operators

    def _register_kind(self, plugin: _Plugin, kinds: set[str]) -> None:
        for kind in kinds:
            self._kinds[kind] = plugin
