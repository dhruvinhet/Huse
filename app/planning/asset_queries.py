"""Deterministically attach semantic asset requests to concept visuals."""

from __future__ import annotations

from app.domain.assets import AssetKind, AssetQuery
from app.domain.lesson import LessonPlan
from app.domain.operations import OperationType
from app.domain.storyboard import Storyboard, VisualObjectSpec


class SemanticAssetQueryPlanner:
    """Ensure visual concepts actually use the offline asset capability."""

    _ELIGIBLE_KINDS = frozenset({
        "browser", "chip", "cloud", "component", "document", "icon", "database",
        "graph_node", "phone", "server", "tree_node",
    })

    def enrich(self, storyboard: Storyboard, lesson: LessonPlan) -> Storyboard:
        """Return a copy with grounded, topic-independent asset queries."""

        result = storyboard.model_copy(deep=True)
        nodes = {
            node.concept_id: node
            for node in lesson.concept_graph.nodes
        }

        def enrich_object(item: VisualObjectSpec) -> None:
            for child in item.children:
                enrich_object(child)
            if item.asset_query is not None or item.kind not in self._ELIGIBLE_KINDS:
                return
            if len(item.concept_ids) != 1:
                return
            node = nodes.get(item.concept_ids[0])
            if node is None:
                return
            semantics = list(dict.fromkeys([
                *node.visual_affordances[:3],
                node.definition,
            ]))
            item.asset_query = AssetQuery(
                concept=node.label,
                asset_kind=AssetKind.LINE_ART,
                style_id="whiteboard.default",
                required_semantics=semantics,
            )
            item.content["asset_slot"] = "left"
            item.content["focal_weight"] = (
                0.9 if node.importance >= 0.75 else 0.6
            )

        for root in result.initial_objects:
            enrich_object(root)
        for beat in result.beats:
            for operation in beat.operations:
                if operation.operation is not OperationType.CREATE:
                    continue
                raw_objects = operation.arguments.get("objects")
                if not isinstance(raw_objects, list):
                    continue
                roots = [
                    VisualObjectSpec.model_validate(item)
                    for item in raw_objects
                    if isinstance(item, dict)
                ]
                for root in roots:
                    enrich_object(root)
                operation.arguments["objects"] = [
                    root.model_dump(mode="json") for root in roots
                ]
        return result
