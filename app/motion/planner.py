"""Distribute pedagogical animation events across phrase timing."""

from collections import defaultdict

from app.domain.layout import LaidOutNode, LayoutPlan
from app.domain.motion import MotionEvent, MotionPlan
from app.domain.narration import AlignedAudio
from app.domain.operations import OperationType, VisualOperation
from app.domain.storyboard import Storyboard


class SemanticAnimationPlanner:
    """Choose animation strategies from object semantics and operations."""

    MIN_DURATION = 0.15

    _KIND_STRATEGIES = {
        "text": "handwriting",
        "label": "handwriting",
        "annotation": "handwriting",
        "equation": "write_left_to_right",
        "array": "reveal_cell_by_cell",
        "array_cell": "reveal_cell",
        "graph": "reveal_nodes_then_edges",
        "graph_node": "node_pop",
        "tree": "recursive_expansion",
        "tree_node": "node_pop",
        "matrix": "draw_grid_then_values",
        "matrix_cell": "cell_highlight",
        "pipeline": "stage_flow",
        "connector": "grow_edge",
        "probability_distribution": "grow_distribution",
        "histogram_bar": "grow_bar",
        "transformer_block": "reveal_components",
        "component": "outline_then_label",
        "semantic_asset": "stroke_reveal",
        "comparison": "split_reveal",
        "timeline": "timeline_trace",
        "cycle": "cycle_trace",
        "cause_effect": "cause_effect_flow",
        "architecture": "layer_stack",
        "flowchart": "decision_flow",
        "venn": "overlap_reveal",
        "bar_chart": "grow_distribution",
        "line_chart": "stroke_reveal",
        "equation_derivation": "write_left_to_right",
        "code_trace": "trace_steps",
    }

    _OPERATOR_STRATEGIES = {
        "flow": "sankey_flow",
        "process": "sankey_flow",
        "cause_effect": "cause_effect_flow",
        "timeline": "timeline_trace",
        "cycle": "cycle_trace",
        "comparison": "split_reveal",
        "funnel": "funnel_collapse",
        "venn": "overlap_reveal",
        "layered": "layer_stack",
        "plot": "plot_trace",
        "bar_chart": "chart_growth",
        "line_chart": "plot_trace",
        "matrix": "matrix_cell_sequence",
        "molecule": "bond_trace",
        "circuit": "signal_trace",
        "map": "route_trace",
        "anatomy": "cutaway_reveal",
        "transform": "morph_state",
        "equation": "derivation_stack",
        "code_trace": "trace_steps",
    }

    _OPERATION_STRATEGIES = {
        OperationType.MOVE: "move",
        OperationType.RESIZE: "resize",
        OperationType.HIGHLIGHT: "pulse_highlight",
        OperationType.DIM: "dim",
        OperationType.MORPH: "morph",
        OperationType.DUPLICATE: "duplicate",
        OperationType.CONNECT: "grow_edge",
        OperationType.DISCONNECT: "erase_edge",
        OperationType.ERASE: "hand_erase",
        OperationType.SHOW: "fade_in",
        OperationType.HIDE: "fade_out",
        OperationType.UPDATE: "update_content",
        OperationType.GROUP: "group_focus",
        OperationType.UNGROUP: "group_expand",
    }

    def plan(
        self,
        storyboard: Storyboard,
        layout: LayoutPlan,
        alignment: AlignedAudio,
    ) -> MotionPlan:
        """Schedule operations and attention cues over full phrase windows."""

        beat_windows = self._beat_windows(alignment)
        kind_index = self._kind_index(layout)
        events: list[MotionEvent] = []
        for beat in storyboard.beats:
            try:
                start, end = beat_windows[beat.beat_id]
            except KeyError as exc:
                raise ValueError(
                    f"audio alignment is missing storyboard beat {beat.beat_id}"
                ) from exc
            work_items: list[
                tuple[str, str, VisualOperation | None, list[str]]
            ] = []
            for operation in beat.operations:
                target_groups = (
                    [[target] for target in self._expanded_targets(
                        operation.target_ids,
                        layout,
                    )]
                    if operation.operation is OperationType.CREATE
                    else [operation.target_ids]
                )
                work_items.extend(
                    (
                        f"{operation.operation_id}_{index:03d}",
                        operation.operation_id,
                        operation,
                        targets,
                    )
                    for index, targets in enumerate(target_groups, start=1)
                )
            work_items.extend(
                (
                    f"{beat.beat_id}_attention_{index:03d}",
                    f"{beat.beat_id}_attention_{index:03d}",
                    None,
                    cue.target_ids,
                )
                for index, cue in enumerate(beat.attention, start=1)
            )
            anchors = self._word_anchors(
                beat.beat_id,
                start,
                end,
                len(work_items),
                alignment,
            )
            for index, (
                event_suffix,
                operation_id,
                operation,
                target_ids,
            ) in enumerate(work_items):
                event_start = anchors[index]
                next_start = (
                    anchors[index + 1]
                    if index + 1 < len(anchors)
                    else end
                )
                slot = max(self.MIN_DURATION, next_start - event_start)
                available = max(1e-6, end - event_start)
                duration = min(
                    available,
                    max(min(self.MIN_DURATION, available), slot * 0.82),
                )
                strategy = (
                    "attention_focus"
                    if operation is None
                    else self._strategy(operation, target_ids, kind_index)
                )
                events.append(
                    MotionEvent(
                        event_id=f"motion_{event_suffix}",
                        beat_id=beat.beat_id,
                        operation_id=operation_id,
                        object_ids=target_ids,
                        strategy=strategy,
                        start_time=event_start,
                        duration=duration,
                        easing="ease_in_out",
                        parameters={
                            "beat_purpose": beat.purpose,
                            "sync": "word" if alignment.words else "phrase",
                            "operation_type": (
                                operation.operation.value
                                if operation is not None
                                else "attention"
                            ),
                        },
                    )
                )
        return MotionPlan(duration=alignment.duration, events=events)

    @staticmethod
    def _word_anchors(
        beat_id: str,
        start: float,
        end: float,
        count: int,
        alignment: AlignedAudio,
    ) -> list[float]:
        """Anchor visual actions to spoken word onsets when available."""

        if count <= 0:
            return []
        words = [word for word in alignment.words if word.beat_id == beat_id]
        if not words:
            slot = (end - start) / count
            return [start + index * slot for index in range(count)]
        if count == 1:
            return [words[0].audio_start]
        last_index = len(words) - 1
        return [
            words[round(index * last_index / (count - 1))].audio_start
            for index in range(count)
        ]

    def _expanded_targets(
        self,
        root_ids: list[str],
        layout: LayoutPlan,
    ) -> list[str]:
        """Expand created hierarchies for progressive disclosure."""

        children: dict[str, list[str]] = {}

        def visit(node: LaidOutNode) -> None:
            children[node.object_id] = [child.object_id for child in node.children]
            for child in node.children:
                visit(child)

        for root in layout.state_roots.values():
            visit(root)
        result: list[str] = []

        def expand(object_id: str) -> None:
            if object_id in result:
                return
            result.append(object_id)
            for child_id in children.get(object_id, []):
                expand(child_id)

        for root_id in root_ids:
            expand(root_id)
        return result

    def _strategy(
        self,
        operation: VisualOperation,
        target_ids: list[str],
        kind_index: dict[str, tuple[str, str]],
    ) -> str:
        """Choose an operation-specific or semantic-kind strategy."""

        if operation.operation is not OperationType.CREATE:
            return self._OPERATION_STRATEGIES.get(
                operation.operation,
                "state_transition",
            )
        first_kind, first_operator = next(
            (
                kind_index[target]
                for target in target_ids
                if target in kind_index
            ),
            ("semantic_asset", ""),
        )
        if first_kind == "connector":
            return self._KIND_STRATEGIES.get(first_kind, "grow_edge")
        return self._OPERATOR_STRATEGIES.get(
            first_operator,
            self._KIND_STRATEGIES.get(first_kind, "stroke_reveal"),
        )

    @staticmethod
    def _beat_windows(alignment: AlignedAudio) -> dict[str, tuple[float, float]]:
        """Merge one or more phrase intervals into each beat window."""

        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for phrase in alignment.phrases:
            grouped[phrase.beat_id].append((phrase.audio_start, phrase.audio_end))
        return {
            beat_id: (
                min(interval[0] for interval in intervals),
                max(interval[1] for interval in intervals),
            )
            for beat_id, intervals in grouped.items()
        }

    def _kind_index(self, layout: LayoutPlan) -> dict[str, tuple[str, str]]:
        """Index object kinds and operators across layout states."""

        index: dict[str, tuple[str, str]] = {}

        def visit(node: LaidOutNode, inherited_operator: str = "") -> None:
            operator = node.operator or inherited_operator
            index[node.object_id] = (node.kind, operator)
            for child in node.children:
                visit(child, operator)

        for root in layout.state_roots.values():
            visit(root)
        return index
