"""Validate semantic operator actions and lower them to visual operations."""

from dataclasses import dataclass

from app.domain.operations import (
    SemanticAction,
    VisualOperation,
    OperationType,
)
from app.domain.visual_document import ObjectState
from app.domain.visual_intent import RendererOperator


class UnsupportedOperatorActionError(ValueError):
    """Raised before rendering when an operator cannot perform an action."""


@dataclass(frozen=True, slots=True)
class OperatorStateModel:
    """Closed action capability set for one semantic operator."""

    operator: RendererOperator
    supported_actions: frozenset[str]


_FLOW_ACTIONS = frozenset({
    "transfer", "route", "split", "merge", "consume", "produce",
    "transform", "accumulate", "trace",
})
_STRUCTURAL_ACTIONS = frozenset({
    "route", "split", "merge", "group", "compare", "trace",
})
_COMPARISON_ACTIONS = frozenset({
    "compare", "substitute", "transform", "trace",
})
_COMPUTATION_ACTIONS = frozenset({
    "transfer", "route", "split", "merge", "compare", "consume",
    "produce", "substitute", "accumulate", "trace", "transform",
})
_GENERIC_ACTIONS = frozenset({"group", "compare", "transform", "trace"})


def _operator_models() -> dict[RendererOperator, OperatorStateModel]:
    flow = {
        RendererOperator.FLOW,
        RendererOperator.PROCESS,
        RendererOperator.PROTOCOL,
        RendererOperator.SYSTEM,
        RendererOperator.CAUSE_EFFECT,
        RendererOperator.FLOWCHART,
        RendererOperator.FUNNEL,
        RendererOperator.CYCLE,
        RendererOperator.TRANSFORM,
    }
    structural = {
        RendererOperator.GROUP,
        RendererOperator.TREE,
        RendererOperator.GRAPH,
        RendererOperator.SPATIAL,
        RendererOperator.LAYERED,
        RendererOperator.TREE_INDEX,
        RendererOperator.MAP,
        RendererOperator.MOLECULE,
        RendererOperator.CIRCUIT,
    }
    comparison = {
        RendererOperator.COMPARISON,
        RendererOperator.VENN,
        RendererOperator.TABLE,
    }
    computation = {
        RendererOperator.ARRAY,
        RendererOperator.MATRIX,
        RendererOperator.SORTING,
        RendererOperator.BINARY_SEARCH,
        RendererOperator.GRAPH_TRAVERSAL,
        RendererOperator.CODE_TRACE,
        RendererOperator.SIMULATION,
        RendererOperator.HASH_MAP,
        RendererOperator.MEMORY_MAP,
        RendererOperator.SCHEDULING,
        RendererOperator.BLOCKCHAIN,
    }
    result: dict[RendererOperator, OperatorStateModel] = {}
    for operator in RendererOperator:
        actions = (
            _FLOW_ACTIONS if operator in flow
            else _STRUCTURAL_ACTIONS if operator in structural
            else _COMPARISON_ACTIONS if operator in comparison
            else _COMPUTATION_ACTIONS if operator in computation
            else _GENERIC_ACTIONS
        )
        result[operator] = OperatorStateModel(operator, actions)
    return result


OPERATOR_STATE_MODELS = _operator_models()


class SemanticActionCompiler:
    """Lower typed actions only after capability and operand validation."""

    def lower(
        self,
        action: SemanticAction,
        states: dict[str, ObjectState],
    ) -> list[VisualOperation]:
        model = OPERATOR_STATE_MODELS[action.operator]
        if action.action not in model.supported_actions:
            raise UnsupportedOperatorActionError(
                f"operator {action.operator.value!r} does not support "
                f"semantic action {action.action!r}; supported="
                f"{sorted(model.supported_actions)}"
            )
        referenced = self._referenced_ids(action)
        if not referenced.issubset(set(action.operand_ids)):
            missing = sorted(referenced - set(action.operand_ids))
            raise ValueError(
                f"semantic action {action.action_id!r} omits referenced "
                f"operands: {missing}"
            )
        new_ids = set(getattr(action, "output_ids", [])) if action.action == "split" else set()
        required = referenced - new_ids
        unknown = sorted(required - set(states))
        if unknown:
            raise ValueError(
                f"semantic action {action.action_id!r} references unknown "
                f"operands: {unknown}"
            )
        if len(new_ids) != len(getattr(action, "output_ids", [])):
            raise ValueError("split output IDs must be unique")
        return self._lower_valid(action, states)

    @staticmethod
    def reverse(action: SemanticAction) -> SemanticAction:
        """Build the explicit inverse for reversible directional actions."""

        if not action.reversible:
            raise ValueError(f"semantic action {action.action_id!r} is not reversible")
        direction = "reverse" if action.direction == "forward" else "forward"
        if action.action == "transfer":
            return action.model_copy(update={
                "action_id": f"{action.action_id}_reverse",
                "source_id": action.target_id,
                "target_id": action.source_id,
                "direction": direction,
                "preconditions": [],
                "postconditions": [],
            })
        if action.action == "substitute":
            return action.model_copy(update={
                "action_id": f"{action.action_id}_reverse",
                "source_id": action.replacement_id,
                "replacement_id": action.source_id,
                "direction": direction,
                "preconditions": [],
                "postconditions": [],
            })
        raise ValueError(
            f"semantic action {action.action!r} has no lossless inverse"
        )

    @staticmethod
    def _referenced_ids(action: SemanticAction) -> set[str]:
        ids = set(action.operand_ids)
        for name in (
            "source_id", "target_id", "group_id", "left_id", "right_id",
            "consumer_id", "producer_id", "replacement_id", "accumulator_id",
        ):
            value = getattr(action, name, None)
            if isinstance(value, str):
                ids.add(value)
        for name in (
            "payload_ids", "path_ids", "output_ids", "input_ids", "member_ids",
            "item_ids",
        ):
            value = getattr(action, name, None)
            if isinstance(value, list):
                ids.update(value)
        return ids

    @staticmethod
    def _operation(
        action: SemanticAction,
        suffix: str,
        operation: OperationType,
        targets: list[str],
        arguments: dict[str, object] | None = None,
    ) -> VisualOperation:
        return VisualOperation(
            operation_id=f"semantic_{action.action_id}_{suffix}",
            operation=operation,
            target_ids=targets,
            arguments=arguments or {},
            reversible=action.reversible,
        )

    def _lower_valid(
        self,
        action: SemanticAction,
        states: dict[str, ObjectState],
    ) -> list[VisualOperation]:
        marker = {
            "semantic_action": action.action,
            "action_id": action.action_id,
            "direction": action.direction,
            "relation": action.relation.value if action.relation else None,
        }
        if action.action == "transfer":
            return [self._operation(
                action, "move", OperationType.MOVE, action.payload_ids,
                {"parent_id": action.target_id, **marker},
            )]
        if action.action == "route":
            route = [action.source_id, *action.path_ids, action.target_id]
            return [self._operation(
                action, "route", OperationType.UPDATE, action.operand_ids,
                {"content": {**marker, "route": route}},
            )]
        if action.action == "split":
            existing = [item for item in action.output_ids if item in states]
            created = [item for item in action.output_ids if item not in states]
            result: list[VisualOperation] = []
            if created:
                result.append(self._operation(
                    action, "split", OperationType.DUPLICATE, created,
                    {"source_id": action.source_id, **marker},
                ))
            if existing:
                result.extend([
                    self._operation(
                        action, "split_state", OperationType.UPDATE, existing,
                        {"content": {**marker, "split_from": action.source_id}},
                    ),
                    self._operation(
                        action, "show_outputs", OperationType.SHOW, existing,
                    ),
                ])
            return result
        if action.action == "merge":
            hidden = [item for item in action.input_ids if item != action.target_id]
            result = [self._operation(
                action, "merge", OperationType.UPDATE, [action.target_id],
                {"content": {**marker, "merged_from": action.input_ids}},
            )]
            if hidden:
                result.append(self._operation(
                    action, "hide_inputs", OperationType.HIDE, hidden,
                ))
            return result
        if action.action == "group":
            return [self._operation(
                action, "group", OperationType.GROUP, action.member_ids,
                {"parent_id": action.group_id, **marker},
            )]
        if action.action == "compare":
            return [
                self._operation(
                    action, "compare_state", OperationType.UPDATE,
                    [action.left_id, action.right_id],
                    {"content": marker},
                ),
                self._operation(
                    action, "compare_focus", OperationType.HIGHLIGHT,
                    [action.left_id, action.right_id],
                ),
            ]
        if action.action == "consume":
            return [
                self._operation(
                    action, "consume_state", OperationType.UPDATE,
                    [action.consumer_id],
                    {"content": {**marker, "consumed": action.item_ids}},
                ),
                self._operation(
                    action, "consume_items", OperationType.HIDE, action.item_ids,
                ),
            ]
        if action.action == "produce":
            return [
                self._operation(
                    action, "produce_state", OperationType.UPDATE,
                    [action.producer_id],
                    {"content": {**marker, "produced": action.item_ids}},
                ),
                self._operation(
                    action, "produce_items", OperationType.SHOW, action.item_ids,
                ),
            ]
        if action.action == "transform":
            return [
                self._operation(
                    action, "transform_state", OperationType.UPDATE,
                    [action.target_id],
                    {"content": {**marker, "transformed_from": action.source_id}},
                ),
                self._operation(
                    action, "hide_source", OperationType.HIDE, [action.source_id],
                ),
                self._operation(
                    action, "show_target", OperationType.SHOW, [action.target_id],
                ),
            ]
        if action.action == "substitute":
            return [
                self._operation(
                    action, "hide_source", OperationType.HIDE, [action.source_id]
                ),
                self._operation(
                    action, "show_replacement", OperationType.SHOW,
                    [action.replacement_id], {"semantic_action": marker},
                ),
            ]
        if action.action == "accumulate":
            return [self._operation(
                action, "accumulate", OperationType.UPDATE,
                [action.accumulator_id],
                {"content": {
                    **marker,
                    "accumulated_items": action.item_ids,
                    "accumulated_count": len(action.item_ids),
                }},
            )]
        if action.action == "trace":
            return [
                self._operation(
                    action, f"trace_{index:02d}", OperationType.UPDATE, [object_id],
                    {"content": {**marker, "trace_order": index}},
                )
                for index, object_id in enumerate(action.path_ids)
            ]
        raise UnsupportedOperatorActionError(
            f"semantic action {action.action!r} has no lowering implementation"
        )
