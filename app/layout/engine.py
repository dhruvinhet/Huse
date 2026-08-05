"""Specialized, deterministic layout for semantic educational objects."""

from dataclasses import dataclass, field
from math import ceil, cos, pi, sin, sqrt

from app.domain.assets import ResolvedAssetSet
from app.domain.layout import (
    LaidOutNode,
    LayoutBox,
    LayoutPlan,
    Viewport,
)
from app.domain.repair import RepairPlan
from app.domain.visual_document import (
    ObjectLifecycle,
    ObjectState,
    VisualDocument,
    VisualState,
)


@dataclass(slots=True)
class _MeasuredNode:
    """Internal relative layout tree before viewport fitting."""

    object_id: str
    kind: str
    width: float
    height: float
    operator: str = ""
    x: float = 0.0
    y: float = 0.0
    children: list["_MeasuredNode"] = field(default_factory=list)


class HierarchicalLayoutEngine:
    """Lay out nested objects without accepting AI pixel coordinates."""

    DEFAULT_GAP = 32.0
    CONTAINER_PADDING = 48.0
    TITLE_HEIGHT = 48.0

    def layout(
        self,
        document: VisualDocument,
        assets: ResolvedAssetSet,
        viewport: Viewport,
    ) -> LayoutPlan:
        """Compute deterministic geometry for every immutable state."""

        roots: dict[str, LaidOutNode] = {}
        diagnostics: list[str] = []
        for state in document.states:
            state_root = self._layout_state(state, viewport, assets)
            roots[state.state_id] = state_root
            diagnostics.extend(self._diagnostics(state.state_id, state_root, viewport))
        return LayoutPlan(
            viewport=viewport,
            state_roots=roots,
            diagnostics=diagnostics,
        )

    def _layout_state(
        self,
        state: VisualState,
        viewport: Viewport,
        assets: ResolvedAssetSet,
    ) -> LaidOutNode:
        """Build and fit one synthetic state root."""

        visible_states = {
            object_id: item
            for object_id, item in state.object_states.items()
            if item.lifecycle
            not in {ObjectLifecycle.REMOVED, ObjectLifecycle.HIDDEN}
        }
        root_states = [
            item
            for item in visible_states.values()
            if item.parent_id is None or item.parent_id not in visible_states
        ]
        root_states.sort(key=lambda item: item.object_id)
        measured = [self._measure(item, visible_states, assets) for item in root_states]
        synthetic = self._stack_roots(state.state_id, measured)
        available_width = viewport.width - 2 * viewport.margin
        available_height = viewport.height - 2 * viewport.margin
        # Use the available viewport when the semantic diagram is smaller
        # than the canvas.  The previous upper bound of 1.0 preserved tiny
        # intrinsic boxes, which made otherwise valid template states occupy
        # 10-15% of the teaching frame and forced the camera to compensate for
        # a layout problem.  This remains aspect-correct and never exceeds the
        # fit scale, so large diagrams are still reduced rather than clipped.
        fit_scale = min(
            available_width / max(1.0, synthetic.width),
            available_height / max(1.0, synthetic.height),
        )
        # Cap enlargement so intrinsic aspect-ratio differences remain
        # observable and a tiny one-dimensional rail cannot become enormous.
        scale = min(1.75, fit_scale)
        offset_x = viewport.margin + (available_width - synthetic.width * scale) / 2
        offset_y = viewport.margin + (available_height - synthetic.height * scale) / 2
        return self._materialize(
            synthetic,
            parent_x=offset_x,
            parent_y=offset_y,
            scale=scale,
            z_index=0,
        )

    def _measure(
        self,
        item: ObjectState,
        states: dict[str, ObjectState],
        assets: ResolvedAssetSet,
    ) -> _MeasuredNode:
        """Measure a semantic subtree with a specialized layout algorithm."""

        child_ids = self._ordered_child_ids(item, states)
        children = [
            self._measure(states[child_id], states, assets)
            for child_id in child_ids
            if child_id in states
            and states[child_id].lifecycle is not ObjectLifecycle.REMOVED
        ]
        if not children:
            width, height = self._intrinsic_size(item, assets)
            return _MeasuredNode(
                item.object_id,
                item.kind,
                width,
                height,
                operator=str(item.content.get("operator", "")),
            )

        layout, gap = self._layout_preferences(item)
        content_children = [
            child for child in children if child.kind != "connector"
        ] or children
        overlay_connectors = [
            child for child in children if child.kind == "connector"
        ]
        operator = str(item.content.get("operator", "")).strip()
        if operator in {"cycle"} or layout == "radial":
            width, height = self._layout_radial(content_children, overlay_connectors)
        elif operator in {"comparison", "venn"} or layout == "split":
            width, height = self._layout_split(content_children, gap)
        elif operator in {"timeline", "protocol", "scheduling"} or layout == "timeline":
            width, height = self._layout_timeline(content_children, gap)
        elif operator == "funnel" or layout == "funnel":
            width, height = self._layout_funnel(content_children, gap)
        elif operator in {"flow", "process", "cause_effect", "transform"} or layout == "sankey":
            width, height = self._layout_sankey(content_children, gap)
        # A generic group is still allowed to declare its composition axis.
        # Treating every group as a layered stack made horizontal summary
        # diagrams collapse into a tall, narrow rail and left the camera with
        # no useful geometry to frame.  Keep layered/spatial operators on the
        # specialized solver, while honoring an explicit group layout hint.
        elif operator == "group" and layout in {"horizontal", "vertical"}:
            # Long horizontal groups become unreadable rails. Keep short
            # groups on their requested axis, but distribute larger concept
            # sets into a deterministic grid so the teaching frame has a
            # useful occupied area and each card remains readable.
            if layout == "horizontal" and len(content_children) > 5:
                width, height = self._layout_grid(content_children, gap)
            else:
                width, height = self._layout_linear(
                    content_children, horizontal=layout == "horizontal", gap=gap
                )
        elif operator in {"layered", "spatial"} or layout == "layered":
            width, height = self._layout_layered(content_children, gap)
        elif operator == "callout" or layout == "callout":
            width, height = self._layout_callout(content_children, gap)
        elif operator in {"plot", "bar_chart", "line_chart", "simulation"} or layout == "chart":
            width, height = self._layout_chart(content_children, gap)
        elif operator == "code_trace" or layout == "code_trace":
            width, height = self._layout_code_trace(content_children, gap)
        elif operator in {"equation", "proof_derivation"} or layout == "equation":
            width, height = self._layout_equation(content_children, gap)
        elif (
            operator in {"tree", "tree_index"}
            or item.kind == "tree"
            or layout == "tree"
        ):
            width, height = self._layout_tree(children)
        elif (
            operator in {"graph", "graph_traversal"}
            or item.kind == "graph"
            or layout == "graph"
        ):
            width, height = self._layout_graph(children)
        elif item.kind == "matrix" or layout == "grid":
            width, height = self._layout_grid(content_children, gap)
        elif layout == "horizontal" or item.kind in {
            "array",
            "pipeline",
            "probability_distribution",
        }:
            width, height = self._layout_linear(
                content_children, horizontal=True, gap=gap
            )
        else:
            width, height = self._layout_linear(
                content_children, horizontal=False, gap=gap
            )
        if layout not in {"tree", "graph"} and item.kind not in {"tree", "graph"}:
            for connector in overlay_connectors:
                connector.x = max(0.0, (width - connector.width) / 2)
                connector.y = max(0.0, (height - connector.height) / 2)

        width += 2 * self.CONTAINER_PADDING
        height += 2 * self.CONTAINER_PADDING + self.TITLE_HEIGHT
        for child in children:
            child.x += self.CONTAINER_PADDING
            child.y += self.CONTAINER_PADDING + self.TITLE_HEIGHT
        return _MeasuredNode(
            item.object_id,
            item.kind,
            max(width, 240.0),
            max(height, 160.0),
            operator=str(item.content.get("operator", "")),
            children=children,
        )

    def _intrinsic_size(
        self,
        item: ObjectState,
        assets: ResolvedAssetSet,
    ) -> tuple[float, float]:
        """Estimate size from content, importance, focal weight, and asset ratio."""

        label = str(
            item.content.get("text")
            or item.content.get("label")
            or item.content.get("value")
            or item.metadata.get("accessibility_label", "")
        )
        detail = str(item.content.get("detail", "")).strip()
        text_width = max(96.0, min(560.0, 28.0 + len(label) * 15.0))
        if detail:
            text_width = max(text_width, min(360.0, 120.0 + len(detail) * 2.2))
        sizes = {
            "label": (text_width, 56.0),
            "text": (text_width, 64.0),
            "annotation": (text_width, 56.0),
            "array_cell": (112.0, 80.0),
            "matrix_cell": (92.0, 68.0),
            "component": (
                max(220.0, text_width),
                178.0 if detail else 104.0,
            ),
            "tree_node": (max(112.0, text_width), 76.0),
            "graph_node": (max(112.0, text_width), 76.0),
            "connector": (96.0, 24.0),
            "token_chip": (max(96.0, text_width), 64.0),
            "equation": (max(220.0, text_width), 84.0),
            "semantic_asset": (256.0, 256.0),
            "icon": (176.0, 176.0),
        }
        if item.kind == "histogram_bar":
            value = float(item.content.get("value", 0.5))
            return 84.0, 80.0 + 260.0 * max(0.0, min(1.0, value))
        width, height = sizes.get(item.kind, (max(160.0, text_width), 96.0))
        asset = assets.for_object(item.object_id)
        if asset is not None and asset.aspect_ratio is not None:
            ratio = max(0.2, min(5.0, float(asset.aspect_ratio)))
            asset_width = max(48.0, min(132.0, 92.0 * sqrt(ratio)))
            asset_height = max(48.0, min(156.0, asset_width / ratio))
            if item.kind in {"component", "icon", "semantic_asset"}:
                width = max(width, text_width + asset_width + 28.0)
                height = max(height, asset_height + 28.0)
        importance = item.metadata.get("importance", 0.5)
        focal_weight = item.metadata.get("focal_weight", 0.5)
        importance_value = float(importance) if isinstance(importance, (int, float)) else 0.5
        focal_value = float(focal_weight) if isinstance(focal_weight, (int, float)) else 0.5
        emphasis = 1.0 + 0.12 * max(0.0, min(1.0, importance_value))
        emphasis += 0.10 * max(0.0, min(1.0, focal_value))
        if item.lifecycle is ObjectLifecycle.EMPHASIZED:
            emphasis += 0.10
        width *= emphasis
        height *= emphasis
        hint = item.metadata.get("layout_hint", {})
        if isinstance(hint, dict):
            scale = hint.get("scale", 1.0)
            if isinstance(scale, (int, float)):
                resolved_scale = max(0.25, min(4.0, float(scale)))
                width *= resolved_scale
                height *= resolved_scale
            size_class = hint.get("size_class")
            size_factors = {"small": 0.75, "medium": 1.0, "large": 1.4}
            if isinstance(size_class, str) and size_class in size_factors:
                width *= size_factors[size_class]
                height *= size_factors[size_class]
        return width, height

    def _layout_linear(
        self,
        children: list[_MeasuredNode],
        horizontal: bool,
        gap: float | None = None,
    ) -> tuple[float, float]:
        """Lay out children along one axis with stable spacing."""

        resolved_gap = self.DEFAULT_GAP if gap is None else max(0.0, gap)
        if horizontal:
            cursor = 0.0
            max_height = 0.0
            for child in children:
                child.x = cursor
                child.y = 0.0
                cursor += child.width + resolved_gap
                max_height = max(max_height, child.height)
            for child in children:
                child.y = (max_height - child.height) / 2
            return max(0.0, cursor - resolved_gap), max_height

        cursor = 0.0
        max_width = 0.0
        for child in children:
            child.x = 0.0
            child.y = cursor
            cursor += child.height + resolved_gap
            max_width = max(max_width, child.width)
        for child in children:
            child.x = (max_width - child.width) / 2
        return max_width, max(0.0, cursor - resolved_gap)

    def _layout_grid(
        self,
        children: list[_MeasuredNode],
        gap: float | None = None,
    ) -> tuple[float, float]:
        """Lay out children in a near-square deterministic grid."""

        columns = max(1, ceil(sqrt(len(children))))
        resolved_gap = self.DEFAULT_GAP if gap is None else max(0.0, gap)
        cell_width = max(child.width for child in children)
        cell_height = max(child.height for child in children)
        rows = ceil(len(children) / columns)
        for index, child in enumerate(children):
            row, column = divmod(index, columns)
            child.x = column * (cell_width + resolved_gap)
            child.y = row * (cell_height + resolved_gap)
        return (
            columns * cell_width + max(0, columns - 1) * resolved_gap,
            rows * cell_height + max(0, rows - 1) * resolved_gap,
        )

    def _layout_tree(self, children: list[_MeasuredNode]) -> tuple[float, float]:
        """Lay out level-order tree nodes by depth."""

        supplemental_kinds = {"annotation", "callout"}
        node_children = [
            child
            for child in children
            if child.kind != "connector"
            and child.kind not in supplemental_kinds
        ]
        supplemental = [
            child for child in children if child.kind in supplemental_kinds
        ]
        if not node_children:
            return self._layout_linear(supplemental or children, horizontal=True)
        max_width = max(child.width for child in node_children)
        max_height = max(child.height for child in node_children)
        depths = [
            (index + 1).bit_length() - 1
            for index in range(len(node_children))
        ]
        level_counts = {
            depth: depths.count(depth)
            for depth in set(depths)
        }
        levels = max(depths) + 1
        canvas_width = max_width * max(level_counts.values())
        for index, child in enumerate(node_children):
            depth = depths[index]
            index_in_level = index - (2**depth - 1)
            count = level_counts[depth]
            slot = canvas_width / count
            child.x = index_in_level * slot + (slot - child.width) / 2
            child.y = depth * (max_height + 72.0)
        height = levels * max_height + max(0, levels - 1) * 72.0
        if not supplemental:
            return canvas_width, height
        supplemental_width, supplemental_height = self._layout_linear(
            supplemental,
            horizontal=False,
            gap=20.0,
        )
        for child in supplemental:
            child.x += max(0.0, (canvas_width - supplemental_width) / 2)
            child.y += height + self.DEFAULT_GAP
        return (
            max(canvas_width, supplemental_width),
            height + self.DEFAULT_GAP + supplemental_height,
        )

    def repair(
        self,
        document: VisualDocument,
        assets: ResolvedAssetSet,
        viewport: Viewport,
        previous: LayoutPlan,
        repair: RepairPlan,
    ) -> LayoutPlan:
        """Apply a bounded geometry correction without rebuilding upstream state."""

        candidate = self.layout(document, assets, viewport)
        codes = set(repair.finding_codes)
        grow = bool(codes.intersection({
            "canvas_underused",
            "rendered_canvas_underused",
            "rendered_effective_font_too_small",
        }))
        shrink = bool(codes.intersection({
            "object_clipped",
            "layout_violation",
            "rendered_canvas_overfilled",
            "rendered_safe_area_clipped",
        }))
        if not grow and not shrink:
            return candidate
        factor = 1.12 if grow else 0.90
        repaired = previous.model_copy(deep=True)
        center_x = repaired.viewport.width / 2
        center_y = repaired.viewport.height / 2
        target_state_ids = {
            state.state_id
            for state in document.states
            if state.beat_id in repair.beat_ids
        }
        for state_id, root in repaired.state_roots.items():
            if target_state_ids and state_id not in target_state_ids:
                continue
            for child in root.children:
                self._scale_laid_out_node(child, factor, center_x, center_y)
        return repaired.model_copy(
            update={
                "diagnostics": [
                    item for item in candidate.diagnostics
                    if not any(code in item for code in repair.finding_codes)
                ]
            }
        )

    @classmethod
    def _scale_laid_out_node(
        cls,
        node: LaidOutNode,
        factor: float,
        center_x: float,
        center_y: float,
    ) -> None:
        """Scale absolute node geometry around the viewport center."""

        node_center_x = node.box.x + node.box.width / 2
        node_center_y = node.box.y + node.box.height / 2
        width = node.box.width * factor
        height = node.box.height * factor
        scaled_center_x = center_x + (node_center_x - center_x) * factor
        scaled_center_y = center_y + (node_center_y - center_y) * factor
        node.box = LayoutBox(
            x=scaled_center_x - width / 2,
            y=scaled_center_y - height / 2,
            width=width,
            height=height,
            rotation=node.box.rotation,
        )
        for child in node.children:
            cls._scale_laid_out_node(
                child,
                factor,
                center_x,
                center_y,
            )

    def _layout_graph(self, children: list[_MeasuredNode]) -> tuple[float, float]:
        """Lay out graph vertices on a circle and retain connector overlays."""

        nodes = [child for child in children if child.kind != "connector"]
        connectors = [child for child in children if child.kind == "connector"]
        if not nodes:
            return self._layout_linear(children, horizontal=True)
        radius = max(180.0, 70.0 * len(nodes))
        center = radius + max(child.width for child in nodes) / 2
        for index, child in enumerate(nodes):
            angle = -pi / 2 + 2 * pi * index / len(nodes)
            child.x = center + radius * cos(angle) - child.width / 2
            child.y = center + radius * sin(angle) - child.height / 2
        for connector in connectors:
            connector.x = center - connector.width / 2
            connector.y = center - connector.height / 2
        extent = 2 * center
        return extent, extent

    def _layout_radial(
        self,
        children: list[_MeasuredNode],
        connectors: list[_MeasuredNode],
    ) -> tuple[float, float]:
        """Place cycle members on a true radial track with a clear center."""

        if not children:
            return self._layout_linear(children, horizontal=True)
        radius = max(180.0, 82.0 * len(children))
        center = radius + max(child.width for child in children) / 2
        for index, child in enumerate(children):
            angle = -pi / 2 + 2 * pi * index / len(children)
            child.x = center + radius * cos(angle) - child.width / 2
            child.y = center + radius * sin(angle) - child.height / 2
        for connector in connectors:
            connector.x = center - connector.width / 2
            connector.y = center - connector.height / 2
        return 2 * center, 2 * center

    def _layout_split(
        self,
        children: list[_MeasuredNode],
        gap: float,
    ) -> tuple[float, float]:
        """Arrange comparison alternatives in explicit left/right columns."""

        if not children:
            return 0.0, 0.0
        columns = 2
        column_width = max(child.width for child in children)
        rows = ceil(len(children) / columns)
        row_height = max(child.height for child in children)
        for index, child in enumerate(children):
            row, column = divmod(index, columns)
            child.x = column * (column_width + gap)
            child.y = row * (row_height + gap)
        return (
            columns * column_width + gap,
            rows * row_height + max(0, rows - 1) * gap,
        )

    def _layout_timeline(
        self,
        children: list[_MeasuredNode],
        gap: float,
    ) -> tuple[float, float]:
        """Arrange events on a shared horizontal axis."""

        width, height = self._layout_linear(children, horizontal=True, gap=gap)
        axis_y = max(0.0, height / 2)
        for child in children:
            child.y = axis_y - child.height / 2
        return width, max(height, 96.0)

    def _layout_funnel(
        self,
        children: list[_MeasuredNode],
        gap: float,
    ) -> tuple[float, float]:
        """Stack narrowing funnel stages around one centered vertical axis."""

        if not children:
            return 0.0, 0.0
        width = max(child.width for child in children)
        cursor = 0.0
        for index, child in enumerate(children):
            factor = max(0.52, 1.0 - index * 0.10)
            child.x = (width - child.width * factor) / 2
            child.y = cursor
            cursor += child.height + gap
        return width, max(0.0, cursor - gap)

    def _layout_sankey(
        self,
        children: list[_MeasuredNode],
        gap: float,
    ) -> tuple[float, float]:
        """Use responsive flow lanes so many operands stay readable.

        A single horizontal lane makes a concept-rich flow shrink below the
        text minimum when it contains more than a few operands.  Keep short
        flows horizontal, but wrap larger flows into a deterministic,
        snake-ordered lane grid.  This preserves directional progression while
        giving each semantic card enough geometry for its label and detail.
        """

        if not children:
            return 0.0, 0.0
        lane_width = max(child.width for child in children)
        lane_gap = max(gap, 64.0)
        max_height = max(child.height for child in children)
        if len(children) <= 4:
            for index, child in enumerate(children):
                child.x = index * (lane_width + lane_gap)
                child.y = (index % 3) * (max_height + gap) / 2
            return (
                len(children) * lane_width
                + max(0, len(children) - 1) * lane_gap,
                max_height + max_height,
            )

        columns = min(3, len(children))
        rows = ceil(len(children) / columns)
        for index, child in enumerate(children):
            row, column = divmod(index, columns)
            # Alternate direction on each row so the visual still reads as a
            # continuous transfer rather than an unrelated card grid.
            if row % 2:
                column = columns - 1 - column
            child.x = column * (lane_width + lane_gap)
            child.y = row * (max_height + gap)
        return (
            columns * lane_width + max(0, columns - 1) * lane_gap,
            rows * max_height + max(0, rows - 1) * gap,
        )

    def _layout_layered(
        self,
        children: list[_MeasuredNode],
        gap: float,
    ) -> tuple[float, float]:
        """Stack architectural layers as broad bands with readable labels."""

        if not children:
            return 0.0, 0.0
        width = max(child.width for child in children)
        cursor = 0.0
        for child in children:
            child.x = (width - child.width) / 2
            child.y = cursor
            cursor += child.height + gap
        return width, max(0.0, cursor - gap)

    def _layout_callout(
        self,
        children: list[_MeasuredNode],
        gap: float,
    ) -> tuple[float, float]:
        """Give annotation/callout content a compact speech-bubble lane."""

        return self._layout_linear(children, horizontal=False, gap=min(gap, 20.0))

    def _layout_chart(
        self,
        children: list[_MeasuredNode],
        gap: float,
    ) -> tuple[float, float]:
        """Reserve a wide plot area and align marks to a common baseline."""

        width, height = self._layout_linear(children, horizontal=True, gap=gap)
        baseline = max(child.height for child in children) if children else 0.0
        for child in children:
            child.y = baseline - child.height
        return max(width, 720.0), max(height, 420.0)

    def _layout_code_trace(
        self,
        children: list[_MeasuredNode],
        gap: float,
    ) -> tuple[float, float]:
        """Split code lines from state snapshots for execution tracing."""

        if not children:
            return 0.0, 0.0
        left = children[: max(1, ceil(len(children) / 2))]
        right = children[len(left):]
        left_width, left_height = self._layout_linear(left, horizontal=False, gap=gap)
        right_width, right_height = self._layout_linear(right, horizontal=False, gap=gap)
        for child in right:
            child.x += left_width + gap
        return left_width + right_width + gap, max(left_height, right_height)

    def _layout_equation(
        self,
        children: list[_MeasuredNode],
        gap: float,
    ) -> tuple[float, float]:
        """Stack derivation steps with a stable left alignment."""

        width, height = self._layout_linear(children, horizontal=False, gap=gap)
        for child in children:
            child.x = 0.0
        return width, height

    def _stack_roots(
        self,
        state_id: str,
        roots: list[_MeasuredNode],
    ) -> _MeasuredNode:
        """Arrange multiple top-level objects as one stable scene document."""

        if not roots:
            return _MeasuredNode(f"{state_id}_root", "document", 1.0, 1.0)
        width, height = self._layout_linear(roots, horizontal=len(roots) <= 3)
        return _MeasuredNode(
            f"{state_id}_root",
            "document",
            width,
            height,
            children=roots,
        )

    def _materialize(
        self,
        node: _MeasuredNode,
        parent_x: float,
        parent_y: float,
        scale: float,
        z_index: int,
    ) -> LaidOutNode:
        """Convert relative measured geometry to absolute viewport geometry."""

        absolute_x = parent_x + node.x * scale
        absolute_y = parent_y + node.y * scale
        children = [
            self._materialize(
                child,
                absolute_x,
                absolute_y,
                scale,
                z_index + index + 1,
            )
            for index, child in enumerate(node.children)
        ]
        return LaidOutNode(
            object_id=node.object_id,
            kind=node.kind,
            operator=node.operator,
            box=LayoutBox(
                x=absolute_x,
                y=absolute_y,
                width=max(1.0, node.width * scale),
                height=max(1.0, node.height * scale),
            ),
            z_index=z_index,
            children=children,
        )

    def _diagnostics(
        self,
        state_id: str,
        root: LaidOutNode,
        viewport: Viewport,
    ) -> list[str]:
        """Report viewport violations that should be impossible after fitting."""

        diagnostics: list[str] = []
        for node in self._flatten(root):
            box = node.box
            if (
                box.x < -1e-6
                or box.y < -1e-6
                or box.x + box.width > viewport.width + 1e-6
                or box.y + box.height > viewport.height + 1e-6
            ):
                diagnostics.append(
                    f"{state_id}: object {node.object_id} exceeds viewport"
                )
            drawable_children = [
                child
                for child in node.children
                if child.kind != "connector"
            ]
            for index, first in enumerate(drawable_children):
                for second in drawable_children[index + 1:]:
                    if self._overlap(first.box, second.box):
                        diagnostics.append(
                            f"{state_id}: sibling objects {first.object_id} and "
                            f"{second.object_id} overlap"
                        )
        return diagnostics

    @staticmethod
    def _overlap(first: LayoutBox, second: LayoutBox) -> bool:
        """Return whether two boxes have a positive-area intersection."""

        return not (
            first.x + first.width <= second.x
            or second.x + second.width <= first.x
            or first.y + first.height <= second.y
            or second.y + second.height <= first.y
        )

    def _layout_preferences(self, item: ObjectState) -> tuple[str, float]:
        """Resolve declarative distribution and minimum-gap constraints."""

        layout = str(item.content.get("layout", "vertical"))
        gap = self.DEFAULT_GAP
        raw_constraints = item.metadata.get("constraints", [])
        if not isinstance(raw_constraints, list):
            return layout, gap
        for raw in raw_constraints:
            if not isinstance(raw, dict):
                continue
            parameters = raw.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}
            if raw.get("type") == "distribute":
                axis = parameters.get("axis")
                if isinstance(axis, str) and axis in {
                    "horizontal", "vertical", "grid", "tree", "graph",
                    "radial", "split", "timeline", "funnel", "sankey",
                    "layered", "callout", "chart", "code_trace", "equation",
                }:
                    layout = axis
            if raw.get("type") in {"distribute", "min_gap"}:
                value = parameters.get("gap")
                if isinstance(value, (int, float)):
                    gap = max(gap, float(value))
        return layout, gap

    def _ordered_child_ids(
        self,
        item: ObjectState,
        states: dict[str, ObjectState],
    ) -> list[str]:
        """Apply declarative ordering and relative layout hints."""

        child_ids = list(item.child_ids)
        raw_constraints = item.metadata.get("constraints", [])
        if isinstance(raw_constraints, list):
            for raw in raw_constraints:
                if not isinstance(raw, dict) or raw.get("type") != "order":
                    continue
                subjects = raw.get("subject_ids")
                if not isinstance(subjects, list):
                    continue
                ordered = [value for value in subjects if value in child_ids]
                child_ids = ordered + [value for value in child_ids if value not in ordered]
        for child_id in list(child_ids):
            child = states.get(child_id)
            hint = child.metadata.get("layout_hint", {}) if child else {}
            if not isinstance(hint, dict):
                continue
            before_id = hint.get("before_id")
            after_id = hint.get("after_id")
            if isinstance(before_id, str) and before_id in child_ids:
                child_ids.remove(child_id)
                child_ids.insert(child_ids.index(before_id), child_id)
            elif isinstance(after_id, str) and after_id in child_ids:
                child_ids.remove(child_id)
                child_ids.insert(child_ids.index(after_id) + 1, child_id)
        return child_ids

    def _flatten(self, root: LaidOutNode) -> list[LaidOutNode]:
        """Return a hierarchy in pre-order."""

        result = [root]
        for child in root.children:
            result.extend(self._flatten(child))
        return result
