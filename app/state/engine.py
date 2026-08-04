"""Apply storyboard operations to immutable persistent visual states."""

from copy import deepcopy

from pydantic import TypeAdapter

from app.domain.operations import OperationType, VisualOperation
from app.domain.storyboard import Storyboard, VisualObjectSpec
from app.domain.visual_document import (
    ObjectLifecycle,
    ObjectState,
    VisualDocument,
    VisualState,
)


class VisualStateTransitionEngine:
    """Materialize deterministic state checkpoints for every visual beat."""

    _OBJECT_LIST = TypeAdapter(list[VisualObjectSpec])

    def materialize(self, storyboard: Storyboard) -> VisualDocument:
        """Apply ordered operations while preserving object identity."""

        current = self._initial_state(storyboard.initial_objects)
        states: list[VisualState] = []
        parent_state_id: str | None = None
        for index, beat in enumerate(storyboard.beats, start=1):
            for operation in beat.operations:
                self._apply(current, operation)
            if beat.shot_plan is not None:
                self._apply_shot_plan(current, beat)
            state_id = f"{storyboard.document_id}_state_{index:04d}"
            states.append(
                VisualState(
                    state_id=state_id,
                    beat_id=beat.beat_id,
                    parent_state_id=parent_state_id,
                    object_states=deepcopy(current),
                )
            )
            parent_state_id = state_id
        return VisualDocument(
            document_id=storyboard.document_id,
            initial_objects=storyboard.initial_objects,
            states=states,
        )

    def _apply_shot_plan(
        self,
        states: dict[str, ObjectState],
        beat: object,
    ) -> None:
        """Keep each shot composition focused and bounded in active geometry."""

        shot_plan = beat.shot_plan
        if shot_plan is None or shot_plan.cleanup_policy == "retain":
            return

        explicit_targets = {
            target_id
            for operation in beat.operations
            for target_id in operation.target_ids
            if target_id in states
        }
        concept_targets = {
            object_id
            for object_id, state in states.items()
            if set(beat.concept_ids).intersection(
                state.metadata.get("concept_ids", [])
                if isinstance(state.metadata.get("concept_ids", []), list)
                else []
            )
        }
        focus = concept_targets or explicit_targets
        entering_whole = any(
            operation.operation.value == "create"
            for operation in beat.operations
        ) and beat.purpose == "introduce"
        if (not concept_targets or entering_whole) and explicit_targets:
            # A root-level create/show target means "orient to the whole
            # diagram" for the entering shot. Preserve its declared children
            # until a later shot supplies a narrower semantic focus.
            for target_id in list(explicit_targets):
                stack = list(states[target_id].child_ids)
                while stack:
                    child_id = stack.pop()
                    if child_id not in states or child_id in focus:
                        continue
                    focus.add(child_id)
                    stack.extend(states[child_id].child_ids)
        if not focus:
            focus = {
                object_id
                for object_id, state in states.items()
                if state.parent_id is None
            }

        # A connector is useful only when one of its endpoints is in focus.
        for object_id, state in states.items():
            if state.kind != "connector":
                continue
            source_id = state.content.get("source_id")
            target_id = state.content.get("target_id")
            if source_id in focus or target_id in focus:
                focus.add(object_id)

        roots = {
            object_id
            for object_id, state in states.items()
            if state.parent_id is None
        }
        focus.update(roots)

        # A recap is a fresh, deliberately sparse view: keep the operator
        # boundary and a few high-importance concept objects, not every past
        # object scaled down to fit.
        if shot_plan.cleanup_policy == "replace" or beat.purpose == "summarize":
            candidates = [
                (object_id, state)
                for object_id, state in states.items()
                if object_id not in roots
                and state.kind != "connector"
                and state.lifecycle is not ObjectLifecycle.REMOVED
            ]
            candidates.sort(
                key=lambda item: (
                    -float(item[1].metadata.get("importance", 0.5)),
                    item[0],
                )
            )
            focus = set(roots)
            focus.update(
                object_id
                for object_id, _state in candidates[: max(1, shot_plan.max_active_objects - len(roots))]
            )

        # A resolved illustration is part of its concept's visual identity.
        # Carry descendants whenever a parent is focused; otherwise shot
        # cleanup can leave a small text card behind while hiding its asset.
        # Asset children are then exempted from the teaching-object budget
        # below: they occupy pixels, but do not represent an additional idea.
        descendants: list[str] = []
        for object_id in list(focus):
            if states[object_id].parent_id is not None:
                descendants.extend(states[object_id].child_ids)
        while descendants:
            child_id = descendants.pop()
            if child_id in states and child_id not in focus:
                focus.add(child_id)
                descendants.extend(states[child_id].child_ids)

        # Preserve ancestors so a focused child remains attached to its
        # semantic container, then trim the least important leaves if the
        # shot's active-object budget is exceeded.
        for object_id in list(focus):
            parent_id = states[object_id].parent_id
            while parent_id is not None and parent_id in states:
                focus.add(parent_id)
                parent_id = states[parent_id].parent_id

        non_connectors = [
            object_id
            for object_id in focus
            if states[object_id].kind != "connector"
            and states[object_id].kind != "semantic_asset"
            and states[object_id].parent_id is not None
        ]
        budget = max(1, shot_plan.max_active_objects - len(roots))
        if len(non_connectors) > budget:
            ranked = sorted(
                non_connectors,
                key=lambda object_id: (
                    -float(states[object_id].metadata.get("importance", 0.5)),
                    object_id,
                ),
            )
            focus.difference_update(ranked[budget:])

        # Connectors consume the same active-object budget. Keep only the
        # first useful relation paths after the focused nodes are admitted.
        connector_ids = sorted(
            object_id
            for object_id in focus
            if states[object_id].kind == "connector"
        )
        active_core = len(
            [
                object_id
                for object_id in focus
                if states[object_id].kind != "connector"
                and states[object_id].kind != "semantic_asset"
                and object_id not in roots
            ]
        )
        connector_budget = max(0, shot_plan.max_active_objects - len(roots) - active_core)
        focus.difference_update(connector_ids[connector_budget:])

        for object_id, state in states.items():
            if object_id in focus:
                if state.lifecycle is ObjectLifecycle.HIDDEN:
                    state.lifecycle = ObjectLifecycle.VISIBLE
                continue
            if state.lifecycle is ObjectLifecycle.REMOVED:
                continue
            state.lifecycle = (
                ObjectLifecycle.DIMMED
                if shot_plan.cleanup_policy == "dim_non_target"
                else ObjectLifecycle.HIDDEN
            )

    def _initial_state(
        self,
        roots: list[VisualObjectSpec],
    ) -> dict[str, ObjectState]:
        """Flatten initial hierarchy while preserving parent relationships."""

        states: dict[str, ObjectState] = {}

        def visit(item: VisualObjectSpec, parent_id: str | None) -> None:
            child_ids = [child.object_id for child in item.children]
            metadata = {
                "semantic_role": item.semantic_role,
                "concept_ids": item.concept_ids,
                "accessibility_label": item.accessibility_label,
                "constraints": [
                    constraint.model_dump(mode="json")
                    for constraint in item.constraints
                ],
            }
            if item.asset_query is not None:
                metadata["asset_id"] = f"asset_{item.object_id}"
                metadata["asset_query"] = item.asset_query.model_dump(mode="json")
            importance = item.content.get("importance")
            if isinstance(importance, (int, float)):
                metadata["importance"] = max(0.0, min(1.0, float(importance)))
            focal_weight = item.content.get("focal_weight")
            if isinstance(focal_weight, (int, float)):
                metadata["focal_weight"] = max(0.0, min(1.0, float(focal_weight)))
            states[item.object_id] = ObjectState(
                object_id=item.object_id,
                kind=item.kind,
                lifecycle=ObjectLifecycle.VISIBLE,
                content=deepcopy(item.content),
                style_token=item.style_token,
                parent_id=parent_id,
                child_ids=child_ids,
                metadata=metadata,
            )
            for child in item.children:
                visit(child, item.object_id)

        for root in roots:
            visit(root, None)
        return states

    def _apply(
        self,
        states: dict[str, ObjectState],
        operation: VisualOperation,
    ) -> None:
        """Apply one validated semantic operation in place."""

        if operation.operation is OperationType.CREATE:
            self._create(states, operation)
            return
        if operation.operation is OperationType.DUPLICATE:
            self._duplicate(states, operation)
            return

        targets = [states[target_id] for target_id in operation.target_ids]
        if operation.operation is OperationType.UPDATE:
            for target in targets:
                content = operation.arguments.get("content")
                if isinstance(content, dict):
                    target.content.update(deepcopy(content))
                style_token = operation.arguments.get("style_token")
                if isinstance(style_token, str) and style_token.strip():
                    target.style_token = style_token
        elif operation.operation in {OperationType.MOVE, OperationType.RESIZE}:
            for target in targets:
                target.metadata["layout_hint"] = deepcopy(operation.arguments)
            if operation.operation is OperationType.MOVE:
                parent_id = operation.arguments.get("parent_id")
                if isinstance(parent_id, str):
                    if parent_id not in states:
                        raise ValueError("move parent_id must reference an existing object")
                    for target in targets:
                        if target.parent_id and target.object_id in states[target.parent_id].child_ids:
                            states[target.parent_id].child_ids.remove(target.object_id)
                        target.parent_id = parent_id
                        if target.object_id not in states[parent_id].child_ids:
                            states[parent_id].child_ids.append(target.object_id)
        elif operation.operation is OperationType.HIGHLIGHT:
            for target in targets:
                target.lifecycle = ObjectLifecycle.EMPHASIZED
        elif operation.operation is OperationType.DIM:
            for target in targets:
                target.lifecycle = ObjectLifecycle.DIMMED
        elif operation.operation is OperationType.SHOW:
            for target in targets:
                target.lifecycle = ObjectLifecycle.VISIBLE
        elif operation.operation is OperationType.HIDE:
            for target in targets:
                target.lifecycle = ObjectLifecycle.HIDDEN
        elif operation.operation is OperationType.ERASE:
            for target in targets:
                target.lifecycle = ObjectLifecycle.REMOVED
        elif operation.operation is OperationType.MORPH:
            for target in targets:
                kind = operation.arguments.get("kind")
                if isinstance(kind, str) and kind.strip():
                    target.kind = kind
                content = operation.arguments.get("content")
                if isinstance(content, dict):
                    target.content = deepcopy(content)
        elif operation.operation in {OperationType.CONNECT, OperationType.DISCONNECT}:
            for target in targets:
                target.content.update(deepcopy(operation.arguments))
                target.lifecycle = (
                    ObjectLifecycle.VISIBLE
                    if operation.operation is OperationType.CONNECT
                    else ObjectLifecycle.HIDDEN
                )
        elif operation.operation is OperationType.GROUP:
            self._group(states, operation)
        elif operation.operation is OperationType.UNGROUP:
            for target in targets:
                target.parent_id = None

    def _create(
        self,
        states: dict[str, ObjectState],
        operation: VisualOperation,
    ) -> None:
        """Create declared object definitions and descendants."""

        definitions = self._OBJECT_LIST.validate_python(
            operation.arguments.get("objects")
        )
        flattened = [item for root in definitions for item in root.flatten()]
        declared_ids = set(operation.target_ids)
        defined_root_ids = {item.object_id for item in definitions}
        if declared_ids != defined_root_ids:
            raise ValueError("create target IDs must match root object definitions")
        if any(item.object_id in states for item in flattened):
            raise ValueError("create cannot replace an existing object")
        created = self._initial_state(definitions)
        states.update(created)

    @staticmethod
    def _duplicate(
        states: dict[str, ObjectState],
        operation: VisualOperation,
    ) -> None:
        """Duplicate one existing object into new persistent IDs."""

        source_id = str(operation.arguments["source_id"])
        source = states[source_id]
        for target_id in operation.target_ids:
            duplicate = source.model_copy(deep=True)
            duplicate.object_id = target_id
            duplicate.parent_id = None
            duplicate.child_ids = []
            duplicate.metadata["duplicated_from"] = source_id
            states[target_id] = duplicate

    @staticmethod
    def _group(
        states: dict[str, ObjectState],
        operation: VisualOperation,
    ) -> None:
        """Assign targets to an existing semantic parent."""

        parent_id = operation.arguments.get("parent_id")
        if not isinstance(parent_id, str) or parent_id not in states:
            raise ValueError("group operations require a known parent_id")
        parent = states[parent_id]
        for target_id in operation.target_ids:
            states[target_id].parent_id = parent_id
            if target_id not in parent.child_ids:
                parent.child_ids.append(target_id)
