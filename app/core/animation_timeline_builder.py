"""Build non-overlapping animation instructions from render scenes."""

from loguru import logger

from app.models.animation import (
    AnimationInstruction,
    AnimationTimeline,
    AnimationType,
    SceneTimeline,
)
from app.models.render import RenderScene, RenderableObject


class AnimationTimelineBuilder:
    """Convert renderable objects into a sequential animation timeline."""

    DEFAULT_ANIMATION_DURATION = 1.0
    ANIMATION_MAPPING = {
        "text": AnimationType.WRITE,
        "svg": AnimationType.DRAW,
        "icon": AnimationType.FADE,
        "image": AnimationType.FADE,
    }

    def build(self, scenes: list[RenderScene]) -> AnimationTimeline:
        """Build per-scene, non-overlapping animation instructions."""

        scene_timelines: list[SceneTimeline] = []
        animation_count = 0

        for scene in scenes:
            next_start_time = 0.0
            instructions: list[AnimationInstruction] = []

            for obj in scene.objects:
                instruction = self._build_instruction(obj, next_start_time)
                instructions.append(instruction)
                next_start_time = instruction.start_time + instruction.duration

            animation_count += len(instructions)
            scene_timelines.append(
                SceneTimeline(
                    scene_number=scene.scene_number,
                    animations=instructions,
                )
            )

        logger.info(
            "Animation timeline built (scenes_processed={}, "
            "animations_created={}).",
            len(scene_timelines),
            animation_count,
        )
        return AnimationTimeline(scenes=scene_timelines)

    def _build_instruction(
        self,
        obj: RenderableObject,
        next_start_time: float,
    ) -> AnimationInstruction:
        """Map and schedule one object without overlapping its predecessor."""

        object_duration = obj.end_time - obj.start_time
        animation_duration = min(
            self.DEFAULT_ANIMATION_DURATION,
            object_duration,
        )
        animation_type = self.ANIMATION_MAPPING.get(
            obj.type.strip().lower(),
            AnimationType.NONE,
        )
        return AnimationInstruction(
            object_id=obj.object_id,
            animation=animation_type,
            start_time=max(next_start_time, obj.start_time),
            duration=animation_duration,
        )
