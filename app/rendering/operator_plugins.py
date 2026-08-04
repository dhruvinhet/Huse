"""Fail-closed pixel plugins for semantic renderer operators and kinds."""

from collections.abc import Callable
from dataclasses import dataclass
from PIL import Image, ImageDraw

from app.domain.layout import LayoutBox
from app.domain.visual_document import ObjectLifecycle, ObjectState


class UnsupportedSemanticKindError(RuntimeError):
    """Raised when no explicit pixel implementation exists."""


class UnsupportedSemanticOperatorError(RuntimeError):
    """Raised when a declared operator has no pixel implementation."""


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
        # Card typography owns the full card width.  Resolved SVGs may be
        # full mini-diagrams or renderer-dependent glyphs; painting them into
        # a small left slot caused clipped black fragments and reduced text
        # readability.  Dedicated semantic_asset objects render artwork at
        # their own measured aspect ratio instead.
        asset_slot = False
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
            self._draw_binary_search(context)
        elif operator == "sorting":
            self._draw_sorting(context)
        elif operator in {"graph_traversal", "system"}:
            self._draw_graph_traversal(context)
        elif operator == "neural_network":
            self._draw_neural_network(context)
        elif operator in {"protocol", "scheduling"}:
            self._draw_protocol(context)
        elif operator in {"tree_index"}:
            self._draw_tree_index(context)
        elif operator in {"memory_map", "hash_map"}:
            self._draw_memory_map(context)
        elif operator == "blockchain":
            ContainerPlugin("blockchain").draw(context)
        elif operator in {"comparison"}:
            self._draw_comparison(context)
        elif operator == "timeline":
            self._draw_timeline(context)
        elif operator == "cycle":
            draw.ellipse((left + 30, top + 32, right - 30, bottom - 24), outline=context.accent, width=5)
            draw.polygon([(right - 32, top + height // 2), (right - 50, top + height // 2 - 10), (right - 50, top + height // 2 + 10)], fill=context.accent)
        elif operator in {"cause_effect", "process"}:
            self._draw_cause_effect(context)
        elif operator in {"layered", "spatial"}:
            self._draw_layered(context)
        elif operator == "flowchart":
            self._draw_flowchart(context)
        elif operator == "funnel":
            for index in range(3):
                inset = 14 + index * 22
                y = top + 42 + index * max(24, (height - 70) // 3)
                draw.polygon([(left + inset, y), (right - inset, y), (right - inset - 14, y + 22), (left + inset + 14, y + 22)], outline=context.accent)
        elif operator == "venn":
            draw.ellipse((left + width // 6, top + 42, left + 2 * width // 3, bottom - 16), outline=context.accent, width=4)
            draw.ellipse((left + width // 3, top + 42, right - width // 6, bottom - 16), outline=context.ink, width=4)
        elif operator in {"bar_chart", "line_chart", "simulation"}:
            self._draw_chart(context)
        elif operator == "callout":
            TextPlugin().draw(context)
        elif operator == "group":
            ContainerPlugin("layered").draw(context)
        elif operator in {"plot", "table", "matrix"}:
            if operator == "plot":
                self._draw_plot(context)
            elif operator == "matrix":
                self._draw_matrix(context)
            else:
                self._draw_comparison(context)
        elif operator == "flow":
            self._draw_flow(context)
        elif operator == "molecule":
            self._draw_molecule(context)
        elif operator == "circuit":
            self._draw_circuit(context)
        elif operator == "map":
            self._draw_map(context)
        elif operator == "anatomy":
            self._draw_anatomy(context)
        elif operator == "transform":
            self._draw_transform(context)
        elif operator == "icon":
            draw.ellipse(
                (left + width // 4, top + height // 5, right - width // 4, bottom - height // 5),
                fill=context.fill,
                outline=context.accent,
                width=5,
            )
        elif operator in {"equation", "proof_derivation"}:
            self._draw_equation(context)
        elif operator in {"code_trace"}:
            self._draw_code_trace(context)
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
            raise UnsupportedSemanticOperatorError(
                f"no renderer plugin for semantic operator {operator!r}"
            )
        if context.label:
            context.draw_text(draw, (left + 8, top + 2, right - 8, min(bottom, top + 42)), context.label, context.font_size, context.ink)

    @staticmethod
    def _child_boxes(context: OperatorDrawingContext) -> list[LayoutBox]:
        """Return visible child geometry in declared order."""

        return [
            context.boxes[item_id]
            for item_id in context.state.child_ids
            if item_id in context.boxes
            and context.boxes[item_id].width > 8
            and context.boxes[item_id].height > 8
        ]

    @classmethod
    def _draw_binary_search(cls, context: OperatorDrawingContext) -> None:
        """Draw search bounds, midpoint, eliminated ranges, and code."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        children = cls._child_boxes(context)
        count = len(children)
        baseline = bottom - 34
        draw.line((left + 12, baseline, right - 12, baseline), fill=context.ink, width=3)
        if not children:
            draw.text((left + 14, top + 36), "low  ->  mid  ->  high", fill=context.accent)
            return
        low = int(context.state.content.get("low", 0))
        high = int(context.state.content.get("high", count - 1) or count - 1)
        mid = int(context.state.content.get("mid", (low + high) // 2) or 0)
        high = min(count - 1, max(low, high))
        for index, child in enumerate(children):
            center = round(child.x + child.width / 2)
            if index < low or index > high:
                draw.line((center - 16, baseline - 20, center + 16, baseline - 2), fill=context.muted if hasattr(context, "muted") else context.ink, width=3)
            if index == mid:
                draw.polygon([(center, baseline - 30), (center - 9, baseline - 16), (center + 9, baseline - 16)], fill=context.accent)
        draw.text((left + 14, top + 36), f"low={low}  mid={mid}  high={high}", fill=context.accent)
        code = str(context.state.content.get("active_code_line", "")).strip()
        if code:
            context.draw_text(draw, (left + 14, bottom - 28, right - 14, bottom - 2), code, 16, context.ink)

    @classmethod
    def _draw_sorting(cls, context: OperatorDrawingContext) -> None:
        """Draw a sorting partition with active-range brackets and pivot."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        children = cls._child_boxes(context)
        if not children:
            return
        pivot = max(0, min(len(children) - 1, int(context.state.content.get("pivot_index", 0))))
        active = context.state.content.get("active_range", [0, len(children) - 1])
        start = max(0, min(len(children) - 1, int(active[0]))) if isinstance(active, (list, tuple)) and len(active) > 1 else 0
        end = max(start, min(len(children) - 1, int(active[1]))) if isinstance(active, (list, tuple)) and len(active) > 1 else len(children) - 1
        first = children[start]
        last = children[end]
        bracket_y = max(top + 28, min(bottom - 8, min(first.y, last.y) - 10))
        draw.line((first.x, bracket_y, last.x + last.width, bracket_y), fill=context.accent, width=4)
        draw.line((first.x, bracket_y, first.x, bracket_y + 12), fill=context.accent, width=4)
        draw.line((last.x + last.width, bracket_y, last.x + last.width, bracket_y + 12), fill=context.accent, width=4)
        pivot_box = children[pivot]
        px = round(pivot_box.x + pivot_box.width / 2)
        draw.polygon([(px, top + 14), (px - 10, top + 30), (px + 10, top + 30)], fill=context.accent)
        draw.text((px + 12, top + 10), "pivot", fill=context.accent)
        draw.text((left + 14, bottom - 24), str(context.state.content.get("phase", "partition")), fill=context.ink)

    @classmethod
    def _draw_graph_traversal(cls, context: OperatorDrawingContext) -> None:
        """Draw traversal edges plus visited/frontier state."""

        ContainerPlugin("graph").draw(context)
        left, _, _, bottom = context.coordinates
        visited = context.state.content.get("visited", [])
        frontier = context.state.content.get("frontier", [])
        draw = context.draw
        draw.text((left + 14, bottom - 38), f"visited: {visited}", fill=context.ink)
        draw.text((left + 14, bottom - 20), f"frontier: {frontier}", fill=context.accent)

    @classmethod
    def _draw_neural_network(cls, context: OperatorDrawingContext) -> None:
        """Draw connected layer columns with an active tensor highlight."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        layer_count = max(2, len(cls._child_boxes(context)))
        columns: list[list[tuple[int, int]]] = []
        for layer in range(layer_count):
            x = left + 32 + round(layer * max(28, (right - left - 64) / max(1, layer_count - 1)))
            nodes = [(x, top + 56 + row * max(20, (bottom - top - 100) // 3)) for row in range(3)]
            columns.append(nodes)
        for first, second in zip(columns, columns[1:], strict=False):
            for source in first:
                for target in second:
                    draw.line((*source, *target), fill=context.accent, width=1)
        active = int(context.state.content.get("active_layer", 0))
        for index, column in enumerate(columns):
            for x, y in column:
                radius = 9 if index == active else 6
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=context.highlight if index == active else context.fill, outline=context.ink, width=2)
        draw.text((left + 14, bottom - 24), "layered tensor flow", fill=context.accent)

    @classmethod
    def _draw_protocol(cls, context: OperatorDrawingContext) -> None:
        """Draw participant lifelines and a highlighted message arrow."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        participants = context.state.content.get("participants", [])
        names = participants if isinstance(participants, list) and participants else ["Sender", "Receiver"]
        xs = [left + (index + 1) * (right - left) // (len(names) + 1) for index in range(len(names))]
        for x, name in zip(xs, names, strict=False):
            draw.line((x, top + 50, x, bottom - 16), fill=context.accent, width=2)
            draw.text((x - 28, top + 24), str(name), fill=context.ink)
        for index, (source, target) in enumerate(zip(xs, xs[1:], strict=False)):
            y = top + 86 + index * 34
            draw.line((source, y, target, y), fill=context.ink, width=3)
            draw.polygon([(target, y), (target - 12, y - 7), (target - 12, y + 7)], fill=context.ink)
        draw.text((left + 14, bottom - 24), f"message {context.state.content.get('active_message', 0)}", fill=context.accent)

    @classmethod
    def _draw_tree_index(cls, context: OperatorDrawingContext) -> None:
        """Draw a hierarchy with a highlighted lookup path."""

        ContainerPlugin("tree").draw(context)
        draw = context.draw
        path = context.state.content.get("lookup_path", [])
        if isinstance(path, list):
            draw.text((context.coordinates[0] + 14, context.coordinates[3] - 24), f"lookup path: {path}", fill=context.accent)

    @classmethod
    def _draw_memory_map(cls, context: OperatorDrawingContext) -> None:
        """Draw addressed memory/hash buckets with an active region marker."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        children = cls._child_boxes(context)
        if children:
            active = int(context.state.content.get("active_region", context.state.content.get("hash_value", 0)))
            for index, child in enumerate(children):
                if index == active % len(children):
                    draw.rectangle((child.x, child.y, child.x + child.width, child.y + child.height), outline=context.accent, width=5)
        draw.text((left + 14, top + 36), context.state.content.get("operator", "memory map").replace("_", " "), fill=context.accent)
        draw.line((left + 16, bottom - 18, right - 16, bottom - 18), fill=context.ink, width=3)

    @classmethod
    def _draw_comparison(cls, context: OperatorDrawingContext) -> None:
        """Draw a two-column comparison with a central decision axis."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        middle = (left + right) // 2
        draw.rounded_rectangle((left + 10, top + 42, middle - 8, bottom - 10), radius=12, outline=context.accent, width=3)
        draw.rounded_rectangle((middle + 8, top + 42, right - 10, bottom - 10), radius=12, outline=context.ink, width=3)
        draw.line((middle, top + 36, middle, bottom - 8), fill=context.ink, width=3)
        draw.text((left + 20, top + 12), "A", fill=context.accent)
        draw.text((middle + 20, top + 12), "B", fill=context.ink)
        draw.polygon([(middle - 12, top + 28), (middle, top + 18), (middle + 12, top + 28)], fill=context.accent)

    @classmethod
    def _draw_timeline(cls, context: OperatorDrawingContext) -> None:
        """Draw a time axis with ordered event ticks and labels."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        y = top + (bottom - top) * 0.58
        draw.line((left + 24, y, right - 24, y), fill=context.ink, width=4)
        children = cls._child_boxes(context)
        for index, child in enumerate(children):
            x = round(child.x + child.width / 2)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=context.accent)
            draw.line((x, y - 8, x, y - 32), fill=context.accent, width=2)
            draw.text((x - 12, y - 52), str(index + 1), fill=context.ink)

    @classmethod
    def _draw_cause_effect(cls, context: OperatorDrawingContext) -> None:
        """Draw a causal chain with emphasized directional arrows."""

        # Explicit connectors are rendered from their declared source/target
        # objects by SemanticFrameRenderer.  Drawing a second, synthetic rail
        # here makes arrows appear before their cards and divorces them from
        # the actual relationships.
        if context.state.content.get("connector_mode") == "explicit":
            return
        cls._draw_flow(context)
        left, top, _, bottom = context.coordinates
        context.draw.text((left + 14, top + 34), "cause  ->  mechanism  ->  effect", fill=context.accent)
        context.draw.text((left + 14, bottom - 28), "causal relation", fill=context.ink)

    @classmethod
    def _draw_layered(cls, context: OperatorDrawingContext) -> None:
        """Draw nested architecture bands with layer labels."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        height = max(1, bottom - top)
        for index, label in enumerate(("interface", "logic", "data")):
            inset = 18 + index * 18
            y = top + 48 + index * max(28, (height - 86) // 3)
            draw.rounded_rectangle((left + inset, y, right - inset, min(bottom - 10, y + 34)), radius=10, outline=context.accent if index == 1 else context.ink, width=3)
            draw.text((left + inset + 10, y + 7), label, fill=context.ink)

    @classmethod
    def _draw_flowchart(cls, context: OperatorDrawingContext) -> None:
        """Draw decision diamonds connected by a directional spine."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        draw.polygon([(center_x, top + 34), (right - 28, center_y), (center_x, bottom - 26), (left + 28, center_y)], outline=context.accent, width=3)
        draw.line((center_x, top + 8, center_x, top + 34), fill=context.ink, width=3)
        draw.line((center_x, bottom - 26, center_x, bottom - 8), fill=context.ink, width=3)
        draw.text((center_x - 28, center_y - 8), "decision", fill=context.ink)

    @classmethod
    def _draw_chart(cls, context: OperatorDrawingContext) -> None:
        """Draw quantitative axes and operator-specific marks."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        draw.line((left + 34, top + 28, left + 34, bottom - 34), fill=context.ink, width=4)
        draw.line((left + 34, bottom - 34, right - 18, bottom - 34), fill=context.ink, width=4)
        operator = str(context.state.content.get("operator", "line_chart"))
        if operator == "bar_chart":
            children = cls._child_boxes(context)
            for index, child in enumerate(children):
                x = child.x
                draw.rectangle((x, bottom - 34, x + child.width, max(top + 54, bottom - 34 - (index + 1) * 18)), fill=context.accent, outline=context.ink, width=2)
        else:
            points = [(left + 52 + index * max(24, (right - left - 90) // 4), bottom - 54 - round((0.2 + index * 0.16) * max(20, bottom - top - 100))) for index in range(5)]
            draw.line(points, fill=context.accent, width=4)
            for point in points:
                draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=context.highlight, outline=context.accent)

    @classmethod
    def _draw_plot(cls, context: OperatorDrawingContext) -> None:
        """Draw a plotted curve with an annotated observation point."""

        cls._draw_chart(context)
        left, top, _, _ = context.coordinates
        context.draw.text((left + 48, top + 42), "observed trend", fill=context.accent)

    @classmethod
    def _draw_matrix(cls, context: OperatorDrawingContext) -> None:
        """Draw a matrix grid with a highlighted diagonal."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        rows = cols = 4
        width = max(1, right - left - 36)
        height = max(1, bottom - top - 54)
        cell_w = width / cols
        cell_h = height / rows
        for row in range(rows):
            for col in range(cols):
                x1 = left + 18 + round(col * cell_w)
                y1 = top + 42 + round(row * cell_h)
                x2 = left + 18 + round((col + 1) * cell_w)
                y2 = top + 42 + round((row + 1) * cell_h)
                draw.rectangle((x1, y1, x2, y2), fill=context.highlight if row == col else context.fill, outline=context.ink, width=2)
                draw.text((x1 + 8, y1 + 8), str(row * cols + col), fill=context.ink)

    @classmethod
    def _draw_equation(cls, context: OperatorDrawingContext) -> None:
        """Draw a derivation stack with equality rails."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        ContainerPlugin("document").draw(context)
        rows = ("given", "substitute", "simplify", "result")
        for index, label in enumerate(rows):
            y = top + 48 + index * max(22, (bottom - top - 76) // len(rows))
            draw.text((left + 48, y), f"{index + 1}. {label}", fill=context.accent if index == len(rows) - 1 else context.ink)
            if index < len(rows) - 1:
                draw.line((left + 40, y + 19, right - 30, y + 19), fill=context.ink, width=1)

    @classmethod
    def _draw_code_trace(cls, context: OperatorDrawingContext) -> None:
        """Draw code lines beside changing state snapshots."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        ContainerPlugin("document").draw(context)
        split = left + (right - left) * 0.58
        draw.line((split, top + 42, split, bottom - 12), fill=context.accent, width=3)
        for index in range(4):
            y = top + 52 + index * max(22, (bottom - top - 82) // 4)
            draw.text((left + 18, y), f"{index + 1:>2}  step", fill=context.accent if index == 0 else context.ink)
            draw.text((split + 16, y), f"state[{index}]", fill=context.ink)

    @staticmethod
    def _draw_flow(context: OperatorDrawingContext) -> None:
        """Draw a process lane with phase markers behind its child cards."""

        if context.state.content.get("connector_mode") == "explicit":
            return
        left, top, right, bottom = context.coordinates
        draw = context.draw
        cards = [
            context.boxes[item]
            for item in context.state.child_ids
            if item in context.boxes and context.boxes[item].width > 40
        ]
        # Keep the fallback rail aligned with the actual cards.  The previous
        # fixed 78% position placed arrows in empty space for grid layouts.
        y = round(sum(box.y + box.height / 2 for box in cards) / len(cards)) if cards else top
        if len(cards) >= 2:
            centers = [round(box.x + box.width / 2) for box in cards]
            draw.line((centers[0], y, centers[-1], y), fill=context.accent, width=6)
            for index, center in enumerate(centers[:-1]):
                next_center = centers[index + 1]
                draw.polygon(
                    [(next_center, y), (next_center - 18, y - 10), (next_center - 18, y + 10)],
                    fill=context.accent,
                )
            for index, center in enumerate(centers, start=1):
                draw.ellipse((center - 12, y - 12, center + 12, y + 12), fill=context.highlight, outline=context.accent, width=3)
                draw.text((center - 5, y - 9), str(index), fill=context.ink)
        draw.text((left + 16, bottom - 28), "ordered process", fill=context.accent)

    @staticmethod
    def _draw_molecule(context: OperatorDrawingContext) -> None:
        """Draw a molecule backdrop and bond guide."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        draw.ellipse((left + 18, top + 48, right - 18, bottom - 18), outline=context.accent, width=3)
        draw.text((left + 18, top + 50), "bonded structure", fill=context.accent)

    @staticmethod
    def _draw_circuit(context: OperatorDrawingContext) -> None:
        """Draw circuit rails and terminals behind component nodes."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        y = top + round((bottom - top) * 0.72)
        draw.line((left + 20, y, right - 20, y), fill=context.accent, width=5)
        for x in (left + 28, right - 28):
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=context.highlight, outline=context.ink, width=2)
        draw.text((left + 18, bottom - 28), "signal path", fill=context.accent)

    @staticmethod
    def _draw_map(context: OperatorDrawingContext) -> None:
        """Draw a map grid and orientation marker."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        for fraction in (0.25, 0.5, 0.75):
            x = round(left + (right - left) * fraction)
            y = round(top + (bottom - top) * fraction)
            draw.line((x, top + 44, x, bottom - 12), fill=context.accent, width=2)
            draw.line((left + 12, y, right - 12, y), fill=context.accent, width=2)
        draw.polygon([(right - 28, top + 58), (right - 42, top + 88), (right - 14, top + 88)], fill=context.accent)

    @staticmethod
    def _draw_anatomy(context: OperatorDrawingContext) -> None:
        """Draw nested cutaway layers rather than a plain card outline."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        height = bottom - top
        for index in range(3):
            inset = 18 + index * 18
            y = top + 52 + index * max(24, (height - 76) // 3)
            draw.rounded_rectangle((left + inset, y, right - inset, y + 26), radius=10, outline=context.accent, width=3)
        draw.text((left + 18, bottom - 28), "layers and parts", fill=context.accent)

    @staticmethod
    def _draw_transform(context: OperatorDrawingContext) -> None:
        """Draw an input-to-output transformation arrow."""

        left, top, right, bottom = context.coordinates
        draw = context.draw
        y = top + round((bottom - top) * 0.70)
        draw.line((left + 30, y, right - 44, y), fill=context.accent, width=6)
        draw.polygon([(right - 30, y), (right - 52, y - 14), (right - 52, y + 14)], fill=context.accent)
        draw.text((left + 18, bottom - 28), "state change", fill=context.accent)


class OperatorRendererRegistry:
    """Resolve every declared operator/kind explicitly and fail closed."""

    def __init__(self) -> None:
        operator_names = {
            "icon", "group", "flow", "callout", "process", "comparison", "timeline", "array", "tree", "graph",
            "equation", "code_trace", "simulation", "spatial", "binary_search",
            "sorting", "graph_traversal", "neural_network", "protocol", "system",
            "tree_index", "scheduling", "memory_map", "hash_map", "blockchain",
            "cycle", "cause_effect", "layered", "flowchart", "funnel", "venn",
            "bar_chart", "line_chart", "plot", "table", "matrix", "molecule", "circuit",
            "map", "anatomy", "transform",
            "semantic_structure",
        }
        self._operators = {
            name: SemanticOperatorPlugin(name)
            for name in operator_names
        }
        self._kinds: dict[str, _Plugin] = {}
        for operator in {"flow", "molecule", "circuit", "map", "anatomy", "transform"}:
            self._register_kind(SemanticOperatorPlugin(operator), {operator})
        self._register_kind(TextPlugin(), {"text", "label", "annotation", "equation", "callout", "speech_bubble"})
        self._register_kind(HistogramBarPlugin(), {"histogram_bar"})
        self._register_kind(DecorationPlugin(), {"brace", "bracket", "underline", "highlight"})
        motif_kinds = {
            "icon": {"icon"},
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
            "layered_family": {"layered"},
            "funnel": {"funnel"},
            "venn": {"venn"},
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
        if plugin is None and operator == "semantic_structure":
            source_operator = str(
                context.state.content.get("source_operator", "")
            ).strip()
            plugin = self._operators.get(source_operator)
        if operator and plugin is None:
            raise UnsupportedSemanticOperatorError(
                f"no renderer plugin for semantic operator {operator!r}"
            )
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
