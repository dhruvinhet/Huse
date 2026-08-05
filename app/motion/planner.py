"""Distribute pedagogical animation events across phrase timing."""

from collections import defaultdict
import re

from app.domain.layout import LaidOutNode, LayoutBox, LayoutPlan
from app.domain.motion import MotionEvent, MotionPlan
from app.domain.narration import AlignedAudio
from app.domain.repair import RepairPlan
from app.domain.operations import OperationType, VisualOperation
from app.domain.storyboard import Storyboard


_STOP_WORDS = {
    "the", "and", "for", "from", "with", "into", "then", "this",
    "that", "item", "component", "visual", "operator", "relation",
}


def _tokens(value: object) -> set[str]:
    """Normalize labels and narration words for semantic matching."""

    return set(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _alias_word_score(aliases: set[str], word: object) -> int:
    """Score an exact readable alias match for one spoken word."""

    word_tokens = _tokens(word)
    if not word_tokens:
        return 0
    return sum(2 if token in aliases else 0 for token in word_tokens)


class SemanticAnimationPlanner:
    """Choose animation strategies from object semantics and operations."""

    MIN_DURATION = 0.15

    _KIND_STRATEGIES = {
        "text": "glyph_stroke",
        "label": "glyph_stroke",
        "annotation": "glyph_stroke",
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
        "component": "glyph_stroke",
        "semantic_asset": "svg_path_reveal",
        "comparison": "split_reveal",
        "timeline": "timeline_trace",
        "cycle": "cycle_trace",
        "cause_effect": "cause_effect_flow",
        "architecture": "layer_stack",
        "flowchart": "decision_flow",
        "venn": "overlap_reveal",
        "bar_chart": "grow_distribution",
        "line_chart": "plot_trace",
        "equation_derivation": "write_left_to_right",
        "code_trace": "trace_steps",
    }

    _OPERATOR_STRATEGIES = {
        "icon": "glyph_stroke",
        "group": "group_focus",
        "flow": "sankey_flow",
        "process": "process_trace",
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
        # Semantic families that inherit a parent operator need an explicit
        # adapter too; otherwise their children silently fall back to generic
        # glyph reveals and every diagram feels like the same card stack.
        "spatial": "cutaway_reveal",
        "binary_search": "trace_steps",
        "sorting": "stage_flow",
        "graph_traversal": "cycle_trace",
        "neural_network": "layer_stack",
        "protocol": "signal_trace",
        "system": "sankey_flow",
        "tree_index": "layer_stack",
        "scheduling": "timeline_trace",
        "memory_map": "layer_stack",
        "hash_map": "matrix_cell_sequence",
        "blockchain": "stage_flow",
        "semantic_structure": "group_focus",
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

    _ACTION_STRATEGIES = {
        "transfer": "semantic_transfer",
        "route": "route_trace",
        "split": "split_reveal",
        "merge": "funnel_collapse",
        "group": "group_focus",
        "compare": "split_reveal",
        "consume": "fade_out",
        "produce": "fade_in",
        "transform": "morph",
        "substitute": "morph",
        "accumulate": "grow_distribution",
        "trace": "trace_steps",
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
        box_index = self._box_index(layout)
        alias_index = self._alias_index(storyboard)
        events: list[MotionEvent] = []
        for beat in storyboard.beats:
            try:
                start, end = beat_windows[beat.beat_id]
            except KeyError as exc:
                raise ValueError(
                    f"audio alignment is missing storyboard beat {beat.beat_id}"
                ) from exc
            work_items: list[
                tuple[str, str, VisualOperation | None, list[str], set[str]]
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
                        aliases,
                    )
                    for index, targets in enumerate(target_groups, start=1)
                    for aliases in [self._target_aliases(targets, alias_index)]
                )
            work_items.extend(
                (
                    f"{beat.beat_id}_attention_{index:03d}",
                    f"{beat.beat_id}_attention_{index:03d}",
                    None,
                    cue.target_ids,
                    self._target_aliases(cue.target_ids, alias_index),
                )
                for index, cue in enumerate(beat.attention, start=1)
            )
            anchors = self._word_anchors(
                beat.beat_id,
                start,
                end,
                [item[4] for item in work_items],
                alignment,
            )
            for index, (
                event_suffix,
                operation_id,
                operation,
                target_ids,
                _aliases,
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
                    max(min(self.MIN_DURATION, available), slot),
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
                            "sync": (
                                "word"
                                if alignment.words
                                and any(word.confidence >= 0.8 for word in alignment.words)
                                else "estimated_word"
                                if alignment.words
                                else "phrase"
                            ),
                            "operation_type": (
                                operation.operation.value
                                if operation is not None
                                else "attention"
                            ),
                        },
                    )
                )
            action_count = len(beat.semantic_actions)
            for action_index, action in enumerate(beat.semantic_actions):
                action_start = start + (end - start) * action_index / max(
                    1, action_count
                )
                action_end = start + (end - start) * (action_index + 1) / max(
                    1, action_count
                )
                moving_ids = self._action_object_ids(action)
                path_ids = self._action_path_ids(action)
                path = [
                    [
                        box_index[object_id].x + box_index[object_id].width / 2,
                        box_index[object_id].y + box_index[object_id].height / 2,
                    ]
                    for object_id in path_ids
                    if object_id in box_index
                ]
                available = max(1e-6, end - action_start)
                slot = max(1e-6, action_end - action_start)
                duration = min(
                    available,
                    max(min(self.MIN_DURATION, available), min(action.duration_hint, slot)),
                )
                events.append(MotionEvent(
                    event_id=f"motion_semantic_{action.action_id}",
                    beat_id=beat.beat_id,
                    operation_id=f"semantic_{action.action_id}",
                    object_ids=moving_ids,
                    strategy=self._ACTION_STRATEGIES[action.action],
                    start_time=action_start,
                    duration=duration,
                    easing=action.easing,
                    parameters={
                        "operation_type": "semantic_action",
                        "semantic_action": action.action,
                        "operator": action.operator.value,
                        "direction": action.direction,
                        "relation": action.relation.value if action.relation else None,
                        "trajectory": path,
                        "state_delta": {
                            "operands": action.operand_ids,
                            "preconditions": len(action.preconditions),
                            "postconditions": len(action.postconditions),
                        },
                    },
                ))
        return MotionPlan(duration=alignment.duration, events=events)

    @staticmethod
    def _action_object_ids(action: object) -> list[str]:
        """Choose objects whose pixels visibly express the action delta."""

        for field in ("payload_ids", "item_ids", "path_ids", "member_ids"):
            value = getattr(action, field, None)
            if isinstance(value, list) and value:
                return list(value)
        source = getattr(action, "source_id", None)
        if isinstance(source, str):
            return [source]
        return list(getattr(action, "operand_ids"))

    @staticmethod
    def _action_path_ids(action: object) -> list[str]:
        """Return semantic endpoints/path order for geometry-derived travel."""

        declared = getattr(action, "path_ids", None)
        source = getattr(action, "source_id", None)
        target = getattr(action, "target_id", None)
        if isinstance(source, str) and isinstance(target, str):
            middle = list(declared) if isinstance(declared, list) else []
            return list(dict.fromkeys([source, *middle, target]))
        if isinstance(declared, list):
            return declared
        return list(getattr(action, "operand_ids"))

    def repair(
        self,
        storyboard: Storyboard,
        layout: LayoutPlan,
        alignment: AlignedAudio,
        previous: MotionPlan,
        repair: RepairPlan,
    ) -> MotionPlan:
        """Replan only motion and apply bounded timing corrections."""

        candidate = self.plan(storyboard, layout, alignment)
        codes = set(repair.finding_codes)
        if "rendered_opening_blank" in codes and candidate.events:
            first_beat = storyboard.beats[0].beat_id
            first_events = [
                event for event in candidate.events if event.beat_id == first_beat
            ]
            if first_events:
                first = min(first_events, key=lambda event: event.start_time)
                first.start_time = 0.0
                first.duration = max(first.duration, min(0.5, candidate.duration))
        if "static_gap_exceeded" in codes and candidate.events:
            ordered = sorted(candidate.events, key=lambda event: event.start_time)
            for index, event in enumerate(ordered):
                boundary = (
                    ordered[index + 1].start_time
                    if index + 1 < len(ordered)
                    else candidate.duration
                )
                event.duration = max(
                    event.duration,
                    min(boundary - event.start_time, 3.0),
                )
        return candidate

    @staticmethod
    def _word_anchors(
        beat_id: str,
        start: float,
        end: float,
        item_aliases: list[set[str]],
        alignment: AlignedAudio,
    ) -> list[float]:
        """Anchor visual actions to spoken word onsets when available."""

        count = len(item_aliases)
        if count <= 0:
            return []
        words = [word for word in alignment.words if word.beat_id == beat_id]
        if not words:
            slot = (end - start) / count
            return [start + index * slot for index in range(count)]
        if count == 1:
            return [words[0].audio_start]

        matched: dict[int, int] = {}
        used_words: set[int] = set()
        for item_index, aliases in enumerate(item_aliases):
            best: tuple[int, int] | None = None
            best_word_index: int | None = None
            for word_index, word in enumerate(words):
                if word_index in used_words:
                    continue
                score = _alias_word_score(aliases, word.text)
                candidate = (score, -word_index)
                if score > 0 and (best is None or candidate > best):
                    best = candidate
                    best_word_index = word_index
            if best_word_index is not None:
                matched[item_index] = best_word_index
                used_words.add(best_word_index)

        unused_word_indexes = [
            index for index in range(len(words)) if index not in used_words
        ]
        fallback_indexes = iter(unused_word_indexes)
        anchors: list[float] = []
        for item_index in range(count):
            word_index = matched.get(item_index)
            if word_index is None:
                word_index = next(
                    fallback_indexes,
                    round(item_index * (len(words) - 1) / (count - 1)),
                )
            anchor = min(end, max(start, words[word_index].audio_start))
            if anchors:
                anchor = max(anchor, anchors[-1])
            anchors.append(anchor)
        # A matched keyword can occur several seconds into a phrase.  Keep a
        # first visual action at the phrase onset so the shot never opens on a
        # static canvas while narration is already explaining it.  Subsequent
        # actions remain anchored to their matched word onsets.
        if anchors and anchors[0] - start > 1.0:
            anchors[0] = start
        return anchors

    @staticmethod
    def _alias_index(storyboard: Storyboard) -> dict[str, set[str]]:
        """Collect readable labels from deterministic CREATE object trees."""

        aliases: dict[str, set[str]] = {}

        def visit(raw: object) -> None:
            if not isinstance(raw, dict):
                return
            object_id = str(raw.get("object_id", "")).strip()
            if not object_id:
                return
            values: list[object] = [
                object_id,
                raw.get("accessibility_label", ""),
                raw.get("semantic_role", ""),
            ]
            content = raw.get("content")
            if isinstance(content, dict):
                values.extend(
                    content.get(key, "")
                    for key in ("label", "detail", "value", "operator")
                )
            aliases[object_id] = {
                token
                for value in values
                for token in _tokens(value)
                if len(token) > 2 and token not in _STOP_WORDS
            }
            children = raw.get("children")
            if isinstance(children, list):
                for child in children:
                    visit(child)

        for beat in storyboard.beats:
            for operation in beat.operations:
                objects = operation.arguments.get("objects")
                if isinstance(objects, list):
                    for raw in objects:
                        visit(raw)
        return aliases

    @staticmethod
    def _target_aliases(
        target_ids: list[str],
        alias_index: dict[str, set[str]],
    ) -> set[str]:
        """Return readable aliases for one motion event's targets."""

        return {
            token
            for target_id in target_ids
            for token in alias_index.get(target_id, set())
        }

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

    @staticmethod
    def _box_index(layout: LayoutPlan) -> dict[str, LayoutBox]:
        """Index final semantic geometry for action trajectories."""

        index: dict[str, LayoutBox] = {}

        def visit(node: LaidOutNode) -> None:
            index[node.object_id] = node.box
            for child in node.children:
                visit(child)

        for root in layout.state_roots.values():
            visit(root)
        return index
