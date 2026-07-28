"""Flatten supported V2 state/layout into the current RenderScene contract."""

from app.domain.assets import ResolvedAssetSet
from app.domain.layout import LaidOutNode, LayoutPlan
from app.domain.visual_document import ObjectLifecycle, VisualDocument
from app.models.render import RenderScene, RenderableObject


class V2ToLegacyRenderAdapter:
    """Provide an explicit, lossy bridge during renderer migration."""

    def convert(
        self,
        document: VisualDocument,
        layout: LayoutPlan,
        assets: ResolvedAssetSet,
        state_duration: float = 1.0,
    ) -> list[RenderScene]:
        """Flatten V2 states without silently creating empty placeholders."""

        if state_duration <= 0:
            raise ValueError("state_duration must be positive")
        asset_by_object = {
            asset.asset_id.removeprefix("asset_"): asset
            for asset in assets.assets
        }
        scenes: list[RenderScene] = []
        for scene_number, state in enumerate(document.states, start=1):
            boxes = {
                node.object_id: node
                for node in self._flatten(layout.state_roots[state.state_id])
            }
            objects: list[RenderableObject] = []
            for object_id, item in state.object_states.items():
                if item.lifecycle in {ObjectLifecycle.HIDDEN, ObjectLifecycle.REMOVED}:
                    continue
                node = boxes.get(object_id)
                if node is None:
                    continue
                label = str(
                    item.content.get("text")
                    or item.content.get("label")
                    or item.content.get("value")
                    or item.metadata.get("accessibility_label")
                    or item.kind
                )
                asset = asset_by_object.get(object_id)
                if asset is not None and asset.mime_type == "image/svg+xml":
                    object_type = "svg"
                    content = asset.path
                    animation = "draw"
                else:
                    object_type = "text"
                    content = label
                    animation = "write"
                objects.append(
                    RenderableObject(
                        object_id=object_id,
                        type=object_type,
                        content=content,
                        x=round(node.box.x + node.box.width / 2),
                        y=round(node.box.y + node.box.height / 2),
                        width=max(1, round(node.box.width)),
                        height=max(1, round(node.box.height)),
                        animation=animation,
                        start_time=0,
                        end_time=state_duration,
                    )
                )
            scenes.append(RenderScene(scene_number=scene_number, objects=objects))
        return scenes

    def _flatten(self, root: LaidOutNode) -> list[LaidOutNode]:
        """Flatten one layout hierarchy."""

        result = [root]
        for child in root.children:
            result.extend(self._flatten(child))
        return result
