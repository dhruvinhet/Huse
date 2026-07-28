"""Convert a V1 Script into an explicit legacy-sourced V2 storyboard."""

from app.domain.operations import OperationType, VisualOperation
from app.domain.storyboard import Storyboard, VisualBeat, VisualObjectSpec
from app.models.script import Script


class LegacyScriptToStoryboardAdapter:
    """Preserve V1 inputs while marking their limited visual semantics."""

    _KIND_MAPPING = {
        "title": "label",
        "text": "text",
        "body": "text",
        "body-text": "text",
        "arrow": "legacy_svg",
        "box": "component",
        "circle": "component",
        "icon": "semantic_asset",
        "image": "semantic_asset",
    }

    def convert(self, script: Script) -> Storyboard:
        """Convert ordered scenes to persistent beats without hidden inference."""

        beats: list[VisualBeat] = []
        previous_ids: list[str] = []
        for scene in script.scenes:
            objects = [
                self._object(scene.scene_number, index, visual)
                for index, visual in enumerate(scene.visuals, start=1)
            ]
            operations: list[VisualOperation] = []
            if previous_ids:
                operations.append(
                    VisualOperation(
                        operation_id=f"legacy_scene_{scene.scene_number}_hide_previous",
                        operation=OperationType.HIDE,
                        target_ids=previous_ids,
                        reversible=False,
                    )
                )
            if objects:
                operations.append(
                    VisualOperation(
                        operation_id=f"legacy_scene_{scene.scene_number}_create",
                        operation=OperationType.CREATE,
                        target_ids=[item.object_id for item in objects],
                        arguments={
                            "objects": [item.model_dump(mode="json") for item in objects]
                        },
                    )
                )
            else:
                fallback = VisualObjectSpec(
                    object_id=f"legacy_scene_{scene.scene_number}_title",
                    kind="label",
                    semantic_role="legacy_scene_title",
                    content={"text": scene.title},
                    style_token="concept.primary",
                    accessibility_label=scene.title,
                )
                objects = [fallback]
                operations.append(
                    VisualOperation(
                        operation_id=f"legacy_scene_{scene.scene_number}_fallback",
                        operation=OperationType.CREATE,
                        target_ids=[fallback.object_id],
                        arguments={"objects": [fallback.model_dump(mode="json")]},
                    )
                )
            beats.append(
                VisualBeat(
                    beat_id=f"legacy_beat_{scene.scene_number}",
                    section_id=f"legacy_section_{scene.scene_number}",
                    concept_ids=[f"legacy_concept_{scene.scene_number}"],
                    teaching_intent=scene.title,
                    phrase_intent=scene.narration,
                    estimated_duration=scene.estimated_duration,
                    operations=operations,
                )
            )
            previous_ids = [item.object_id for item in objects]
        return Storyboard(
            document_id="legacy_document",
            title=script.title,
            beats=beats,
            final_learning_summary=[scene.title for scene in script.scenes],
        )

    def _object(self, scene_number: int, index: int, visual: object) -> VisualObjectSpec:
        """Convert one limited V1 visual with explicit legacy provenance."""

        visual_type = str(getattr(visual, "type")).strip().lower()
        content = str(getattr(visual, "content")).strip()
        kind = self._KIND_MAPPING.get(visual_type, "semantic_asset")
        label = content
        if visual_type in {"box", "circle", "arrow"} and content == visual_type:
            label = visual_type.title()
        return VisualObjectSpec(
            object_id=f"legacy_scene_{scene_number}_object_{index}",
            kind=kind,
            semantic_role=f"legacy_{visual_type}",
            content={
                "label": label,
                "legacy_type": visual_type,
                "legacy_position": str(getattr(visual, "position")),
            },
            style_token="legacy.default",
            accessibility_label=label,
        )
