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

    def _initial_state(
        self,
        roots: list[VisualObjectSpec],
    ) -> dict[str, ObjectState]:
        """Flatten initial hierarchy while preserving parent relationships."""

        states: dict[str, ObjectState] = {}

        def visit(item: VisualObjectSpec, parent_id: str | None) -> None:
            child_ids = [child.object_id for child in item.children]
            states[item.object_id] = ObjectState(
                object_id=item.object_id,
                kind=item.kind,
                lifecycle=ObjectLifecycle.VISIBLE,
                content=deepcopy(item.content),
                style_token=item.style_token,
                parent_id=parent_id,
                child_ids=child_ids,
                metadata={
                    "semantic_role": item.semantic_role,
                    "concept_ids": item.concept_ids,
                    "accessibility_label": item.accessibility_label,
                    "constraints": [
                        constraint.model_dump(mode="json")
                        for constraint in item.constraints
                    ],
                },
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
