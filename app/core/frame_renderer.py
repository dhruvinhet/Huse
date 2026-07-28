"""Render animation timelines as sequential PNG frames using Pillow."""

from math import ceil
from pathlib import Path
from typing import Protocol

from loguru import logger
from PIL import Image

from app.config.settings import PROJECT_ROOT, settings
from app.models.animation import (
    AnimationInstruction,
    AnimationType,
    SceneTimeline,
)
from app.models.render import RenderScene, RenderableObject
from app.renderers.icon_renderer import IconRenderer
from app.renderers.image_renderer import ImageRenderer
from app.renderers.svg_renderer import SVGRenderer
from app.renderers.text_renderer import TextRenderer
from app.utils.debug_recorder import DebugRecorder


class ObjectRenderer(Protocol):
    """Interface shared by specialized static object renderers."""

    def render(self, canvas: Image.Image, obj: RenderableObject) -> bool:
        """Render one object onto a Pillow image."""

        ...


class FrameRenderer:
    """Execute one scene timeline into a deterministic PNG frame sequence."""

    CANVAS_SIZE = (1920, 1080)
    BACKGROUND_COLOR = (255, 255, 255, 255)
    RENDER_ORDER = {
        "text": 0,
        "svg": 1,
        "icon": 2,
        "image": 3,
    }

    def __init__(self, debug_recorder: DebugRecorder | None = None) -> None:
        """Initialize reusable object renderers for cached full layers."""

        self._renderers: dict[str, ObjectRenderer] = {
            "text": TextRenderer(),
            "svg": SVGRenderer(),
            "icon": IconRenderer(),
            "image": ImageRenderer(),
        }
        self._debug_recorder = debug_recorder

    def render(
        self,
        scene: RenderScene,
        timeline: SceneTimeline,
        fps: int = 30,
        frame_start: int = 1,
        frame_count: int | None = None,
        clear_output: bool = True,
    ) -> None:
        """Render a scene into a global, non-overwriting PNG sequence."""

        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        if scene.scene_number != timeline.scene_number:
            raise ValueError("scene and timeline scene numbers must match")
        if frame_start < 1:
            raise ValueError("frame_start must be at least one")

        natural_duration = self._duration(scene, timeline)
        resolved_frame_count = (
            ceil(natural_duration * fps)
            if frame_count is None
            else frame_count
        )
        if resolved_frame_count <= 0:
            raise ValueError("frame_count must be greater than zero")
        duration = resolved_frame_count / fps
        if duration <= 0:
            raise ValueError("scene duration must be greater than zero")

        frames_directory = Path(settings.TEMP_DIR) / "frames"
        working_directory = self._working_path(frames_directory)
        working_directory.mkdir(parents=True, exist_ok=True)
        if clear_output:
            self._remove_stale_frames(working_directory)
        self._validate_destinations(
            working_directory,
            frame_start,
            resolved_frame_count,
        )

        instructions = {
            instruction.object_id: instruction
            for instruction in timeline.animations
        }
        ordered_objects = sorted(
            scene.objects,
            key=lambda item: self.RENDER_ORDER.get(item.type.lower(), 99),
        )
        cached_layers = {
            obj.object_id: self._render_full_layer(obj)
            for obj in ordered_objects
        }
        base_canvas = Image.new(
            "RGBA",
            self.CANVAS_SIZE,
            self.BACKGROUND_COLOR,
        )
        sample_indexes = {
            0,
            resolved_frame_count // 2,
            resolved_frame_count - 1,
        }
        for frame_index in range(resolved_frame_count):
            timestamp = min((frame_index + 1) / fps, duration)
            canvas = base_canvas.copy()
            for obj in ordered_objects:
                instruction = instructions.get(obj.object_id)
                self._composite_object(
                    canvas=canvas,
                    obj=obj,
                    instruction=instruction,
                    timestamp=timestamp,
                    full_layer=cached_layers[obj.object_id],
                )

            frame_number = frame_start + frame_index
            frame_path = working_directory / f"frame_{frame_number:06d}.png"
            canvas.convert("RGB").save(frame_path, format="PNG")
            self._record_frame_debug(
                scene=scene,
                ordered_objects=ordered_objects,
                instructions=instructions,
                frame_number=frame_number,
                frame_index=frame_index,
                fps=fps,
                timestamp=timestamp,
                frame_path=frame_path,
                is_sample=frame_index in sample_indexes,
            )

        logger.info(
            "Frames rendered (range={}-{}, frames={}, fps={}, duration={}, "
            "output_folder={}).",
            frame_start,
            frame_start + resolved_frame_count - 1,
            resolved_frame_count,
            fps,
            duration,
            frames_directory.as_posix(),
        )

    def _composite_object(
        self,
        canvas: Image.Image,
        obj: RenderableObject,
        instruction: AnimationInstruction | None,
        timestamp: float,
        full_layer: Image.Image,
    ) -> None:
        """Apply one instruction state and composite its object layer."""

        if instruction is None or instruction.animation is AnimationType.NONE:
            canvas.alpha_composite(full_layer)
            return

        progress = self._progress(instruction, timestamp)
        if progress <= 0:
            return
        if progress >= 1:
            canvas.alpha_composite(full_layer)
            return

        if instruction.animation is AnimationType.WRITE:
            self._composite_write(canvas, obj, progress)
        elif instruction.animation is AnimationType.DRAW:
            self._composite_draw(canvas, obj, full_layer, progress)
        elif instruction.animation is AnimationType.FADE:
            self._composite_fade(canvas, obj, full_layer, progress)

    def _composite_write(
        self,
        canvas: Image.Image,
        obj: RenderableObject,
        progress: float,
    ) -> None:
        """Render a prefix of a text object based on animation progress."""

        character_count = max(1, ceil(len(obj.content) * progress))
        partial_object = obj.model_copy(
            update={"content": obj.content[:character_count]}
        )
        layer = Image.new("RGBA", self.CANVAS_SIZE, (0, 0, 0, 0))
        self._renderers["text"].render(layer, partial_object)
        canvas.alpha_composite(layer)

    def _composite_draw(
        self,
        canvas: Image.Image,
        obj: RenderableObject,
        full_layer: Image.Image,
        progress: float,
    ) -> None:
        """Reveal a primitive layer from left to right using linear progress."""

        left, top, right, bottom = self._object_box(obj)
        visible_right = left + max(1, round((right - left) * progress))
        region = full_layer.crop((left, top, visible_right, bottom))
        canvas.alpha_composite(region, dest=(left, top))

    def _composite_fade(
        self,
        canvas: Image.Image,
        obj: RenderableObject,
        full_layer: Image.Image,
        progress: float,
    ) -> None:
        """Scale a cached object's alpha channel using linear progress."""

        left, top, right, bottom = self._object_box(obj)
        region = full_layer.crop((left, top, right, bottom))
        alpha = region.getchannel("A").point(
            lambda value: round(value * progress)
        )
        region.putalpha(alpha)
        canvas.alpha_composite(region, dest=(left, top))

    def _render_full_layer(self, obj: RenderableObject) -> Image.Image:
        """Render and cache one complete object on a transparent layer."""

        layer = Image.new("RGBA", self.CANVAS_SIZE, (0, 0, 0, 0))
        renderer = self._renderers.get(obj.type.strip().lower())
        if renderer is None:
            logger.warning(
                "Frame renderer does not support object {} of type {}.",
                obj.object_id,
                obj.type,
            )
            return layer
        renderer.render(layer, obj)
        return layer

    def _record_frame_debug(
        self,
        scene: RenderScene,
        ordered_objects: list[RenderableObject],
        instructions: dict[str, AnimationInstruction],
        frame_number: int,
        frame_index: int,
        fps: int,
        timestamp: float,
        frame_path: Path,
        is_sample: bool,
    ) -> None:
        """Record object animation state for one rendered frame."""

        if self._debug_recorder is None:
            return
        object_states: list[dict[str, object]] = []
        for obj in ordered_objects:
            instruction = instructions.get(obj.object_id)
            if instruction is None:
                animation = AnimationType.NONE
                progress = 1.0
            else:
                animation = instruction.animation
                progress = (
                    1.0
                    if animation is AnimationType.NONE
                    else self._progress(instruction, timestamp)
                )
            state: dict[str, object] = {
                "object_id": obj.object_id,
                "type": obj.type,
                "animation": animation.value,
                "progress": round(progress, 6),
                "visible": progress > 0,
                "position": {"x": obj.x, "y": obj.y},
                "size": {"width": obj.width, "height": obj.height},
            }
            if animation is AnimationType.WRITE:
                character_count = min(
                    len(obj.content),
                    max(0, ceil(len(obj.content) * progress)),
                )
                state["visible_text"] = obj.content[:character_count]
            object_states.append(state)

        self._debug_recorder.append_jsonl(
            "frames/frame_trace.jsonl",
            {
                "frame_number": frame_number,
                "scene_number": scene.scene_number,
                "scene_frame_index": frame_index,
                "global_time_seconds": round((frame_number - 1) / fps, 6),
                "scene_time_seconds": round(timestamp, 6),
                "output_path": frame_path.as_posix(),
                "objects": object_states,
            },
        )
        if is_sample:
            self._debug_recorder.copy_file(
                frame_path,
                (
                    "frames/samples/"
                    f"scene_{scene.scene_number:03d}_"
                    f"frame_{frame_number:06d}.png"
                ),
            )

    def _object_box(self, obj: RenderableObject) -> tuple[int, int, int, int]:
        """Return a canvas-clamped box around a center-positioned object."""

        left = max(0, obj.x - obj.width // 2)
        top = max(0, obj.y - obj.height // 2)
        right = min(self.CANVAS_SIZE[0], left + obj.width)
        bottom = min(self.CANVAS_SIZE[1], top + obj.height)
        return left, top, right, bottom

    @staticmethod
    def _progress(
        instruction: AnimationInstruction,
        timestamp: float,
    ) -> float:
        """Calculate clamped linear progress for a frame timestamp."""

        if timestamp <= instruction.start_time:
            return 0.0
        elapsed = timestamp - instruction.start_time
        return min(1.0, elapsed / instruction.duration)

    @staticmethod
    def _duration(scene: RenderScene, timeline: SceneTimeline) -> float:
        """Determine the duration required by objects and instructions."""

        object_end = max(
            (obj.end_time for obj in scene.objects),
            default=0.0,
        )
        animation_end = max(
            (
                instruction.start_time + instruction.duration
                for instruction in timeline.animations
            ),
            default=0.0,
        )
        return max(object_end, animation_end)

    @staticmethod
    def _working_path(configured_path: Path) -> Path:
        """Resolve a configured relative frame directory from project root."""

        if configured_path.is_absolute():
            return configured_path
        return PROJECT_ROOT / configured_path

    @staticmethod
    def _remove_stale_frames(frames_directory: Path) -> None:
        """Remove only prior sequential PNG frames from the output folder."""

        for stale_frame in frames_directory.glob("frame_*.png"):
            stale_frame.unlink()

    @staticmethod
    def _validate_destinations(
        frames_directory: Path,
        frame_start: int,
        frame_count: int,
    ) -> None:
        """Refuse to overwrite any frame in the requested global range."""

        for frame_number in range(
            frame_start,
            frame_start + frame_count,
        ):
            frame_path = frames_directory / f"frame_{frame_number:06d}.png"
            if frame_path.exists():
                raise FileExistsError(
                    f"Frame already exists and will not be overwritten: "
                    f"{frame_path.name}"
                )
