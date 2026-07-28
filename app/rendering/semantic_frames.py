"""Render hierarchical semantic states and motion into PNG frames."""

import os
import shutil
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from app.config.settings import PROJECT_ROOT
from app.design import WhiteboardDesignSystem
from app.domain.camera import CameraCue, CameraOperation
from app.domain.layout import LaidOutNode, LayoutBox, Viewport
from app.domain.motion import MotionEvent
from app.domain.rendering import FrameSequence, RenderJob
from app.domain.visual_document import ObjectLifecycle, ObjectState, VisualState
from app.models.render import RenderableObject
from app.renderers.svg_renderer import SVGRenderer
from app.renderers.image_renderer import ImageRenderer
from app.utils.debug_recorder import DebugRecorder


@dataclass(frozen=True)
class _CachedLayer:
    """Store a tightly cropped object layer and its canvas position."""

    image: Image.Image
    position: tuple[int, int]


class SemanticFrameRenderer:
    """Execute V2 state, layout, motion, attention, and camera plans."""

    _DESIGN = WhiteboardDesignSystem()
    BACKGROUND = _DESIGN.background
    INK = _DESIGN.ink
    MUTED = _DESIGN.muted
    ACCENT = _DESIGN.blue
    HIGHLIGHT = _DESIGN.yellow
    TRACE_BATCH_SIZE = 128
    MAX_SAVE_WORKERS = 4
    MIN_FONT_SIZE = 16

    def __init__(self, debug_recorder: DebugRecorder | None = None) -> None:
        """Initialize reusable asset renderer and optional trace recorder."""

        self._svg_renderer = SVGRenderer()
        self._image_renderer = ImageRenderer()
        self._debug = debug_recorder
        self._design = WhiteboardDesignSystem()
        self._font_cache: dict[int, ImageFont.ImageFont] = {}
        self._layer_cache: dict[str, _CachedLayer] = {}

    def render(self, job: RenderJob) -> FrameSequence:
        """Render all manifest frames as a continuous deterministic sequence."""

        frames_directory = self._working_path(Path(job.output_folder))
        frames_directory.mkdir(parents=True, exist_ok=True)
        for stale in frames_directory.glob("frame_*.png"):
            stale.unlink()

        state_by_beat = {state.beat_id: state for state in job.document.states}
        state_order = [state.beat_id for state in job.document.states]
        starts = self._beat_starts(job)
        events_by_beat: dict[str, list[MotionEvent]] = defaultdict(list)
        for event in job.motion.events:
            events_by_beat[event.beat_id].append(event)
        cues_by_beat = {cue.beat_id: cue for cue in job.camera.cues}
        samples: list[str] = []
        sample_frames = self._sample_frames(job)
        flattened_by_beat = {
            beat_id: self._flatten(job.layout.state_roots[state.state_id])
            for beat_id, state in state_by_beat.items()
        }
        raw_boxes_by_beat = {
            beat_id: {node.object_id: node.box for node in nodes}
            for beat_id, nodes in flattened_by_beat.items()
        }
        visible_ids_by_beat = {
            beat_id: self._visible_context_ids(
                state_by_beat[beat_id],
                cues_by_beat.get(beat_id),
            )
            for beat_id in state_by_beat
        }
        boxes_by_beat = {
            beat_id: self._fit_view_boxes(
                state_by_beat[beat_id],
                flattened_by_beat[beat_id],
                raw_boxes_by_beat[beat_id],
                visible_ids_by_beat[beat_id],
                job.layout.viewport,
                cues_by_beat.get(beat_id),
            )
            for beat_id in state_by_beat
        }
        ordered_by_beat = {
            beat_id: sorted(
                (
                    node
                    for node in nodes
                    if node.object_id in state_by_beat[beat_id].object_states
                    and node.object_id in visible_ids_by_beat[beat_id]
                ),
                key=lambda node: (
                    self._render_priority(
                        node.kind,
                        bool(
                            state_by_beat[beat_id]
                            .object_states[node.object_id]
                            .child_ids
                        ),
                    ),
                    node.z_index,
                ),
            )
            for beat_id, nodes in flattened_by_beat.items()
        }
        save_workers = max(
            1,
            min(self.MAX_SAVE_WORKERS, os.cpu_count() or 1),
        )
        pending_saves: deque[Future[None]] = deque()
        trace_batch: list[dict[str, Any]] = []
        last_signature: tuple[Any, ...] | None = None
        last_rendered_path: Path | None = None
        last_save: Future[None] | None = None
        previous_beat_id: str | None = None
        beat_index = 0
        rendered_frames = 0
        reused_frames = 0
        if self._debug is not None:
            self._debug.write_text("v2/frames/frame_trace.jsonl", "")

        with ThreadPoolExecutor(
            max_workers=save_workers,
            thread_name_prefix="png-writer",
        ) as executor:
            for frame_number in range(1, job.manifest.total_frames + 1):
                timestamp = (frame_number - 1) / job.manifest.fps
                while (
                    beat_index + 1 < len(state_order)
                    and starts[state_order[beat_index + 1]]
                    <= timestamp + 1e-9
                ):
                    beat_index += 1
                beat_id = state_order[beat_index]
                state = state_by_beat[beat_id]
                previous_state = (
                    state_by_beat[state_order[beat_index - 1]]
                    if beat_index > 0
                    else None
                )
                previous_boxes = (
                    boxes_by_beat[previous_state.beat_id]
                    if previous_state is not None
                    else {}
                )
                events = events_by_beat.get(beat_id, [])
                cue = cues_by_beat.get(beat_id)
                signature = self._frame_signature(
                    beat_id,
                    events,
                    cue,
                    timestamp,
                )
                frame_path = (
                    frames_directory / f"frame_{frame_number:06d}.png"
                )

                if (
                    signature == last_signature
                    and last_rendered_path is not None
                ):
                    if last_save is not None:
                        last_save.result()
                        last_save = None
                    self._link_or_copy(last_rendered_path, frame_path)
                    reused_frames += 1
                else:
                    if beat_id != previous_beat_id:
                        self._layer_cache.clear()
                        previous_beat_id = beat_id
                    canvas = self._render_state(
                        job,
                        state,
                        previous_state,
                        ordered_by_beat[beat_id],
                        boxes_by_beat[beat_id],
                        previous_boxes,
                        events,
                        timestamp,
                    )
                    if cue is not None and cue.operation not in {
                        CameraOperation.FIT,
                        CameraOperation.HOLD,
                    }:
                        canvas = self._apply_camera(
                            canvas,
                            cue,
                            timestamp,
                        )
                    last_save = executor.submit(
                        self._save_png,
                        canvas,
                        frame_path,
                    )
                    pending_saves.append(last_save)
                    last_rendered_path = frame_path
                    last_signature = signature
                    rendered_frames += 1
                    if len(pending_saves) > save_workers * 2:
                        pending_saves.popleft().result()

                if frame_number in sample_frames:
                    samples.append(frame_path.as_posix())
                if self._debug is not None:
                    trace_batch.append(
                        self._trace_payload(
                            frame_number,
                            timestamp,
                            beat_id,
                            state,
                            events,
                            frame_path,
                        )
                    )
                    if len(trace_batch) >= self.TRACE_BATCH_SIZE:
                        self._debug.append_jsonl_many(
                            "v2/frames/frame_trace.jsonl",
                            trace_batch,
                        )
                        trace_batch.clear()

            while pending_saves:
                pending_saves.popleft().result()
        if self._debug is not None and trace_batch:
            self._debug.append_jsonl_many(
                "v2/frames/frame_trace.jsonl",
                trace_batch,
            )
        logger.info(
            "Semantic frames complete (rendered={}, reused={}, "
            "png_workers={}).",
            rendered_frames,
            reused_frames,
            save_workers,
        )
        return FrameSequence(
            folder=Path(job.output_folder).as_posix(),
            total_frames=job.manifest.total_frames,
            fps=job.manifest.fps,
            sample_paths=samples,
        )

    def _render_state(
        self,
        job: RenderJob,
        state: VisualState,
        previous_state: VisualState | None,
        ordered: list[LaidOutNode],
        boxes: dict[str, LayoutBox],
        previous_boxes: dict[str, LayoutBox],
        events: list[MotionEvent],
        timestamp: float,
    ) -> Image.Image:
        """Render one state with active transition effects."""

        size = (job.layout.viewport.width, job.layout.viewport.height)
        canvas = Image.new("RGBA", size, self.BACKGROUND)
        event_by_object = {
            object_id: event
            for event in events
            for object_id in event.object_ids
        }
        previous_ids = set(previous_state.object_states) if previous_state else set()

        for node in ordered:
            object_state = state.object_states[node.object_id]
            if object_state.lifecycle in {ObjectLifecycle.HIDDEN, ObjectLifecycle.REMOVED}:
                continue
            event = event_by_object.get(node.object_id)
            progress = self._event_progress(event, timestamp)
            is_new = node.object_id not in previous_ids
            if is_new and event is not None and timestamp < event.start_time:
                continue
            final_box = boxes[node.object_id]
            box = final_box
            if (
                event is not None
                and event.strategy in {"move", "resize", "morph"}
                and node.object_id in previous_boxes
            ):
                box = self._interpolate_box(
                    previous_boxes[node.object_id],
                    node.box,
                    progress,
                )
            cache_key = f"{state.state_id}:{node.object_id}"
            dynamic_connector = (
                object_state.kind == "connector"
                and event is not None
                and event.strategy == "grow_edge"
                and progress < 1.0
            )
            if dynamic_connector:
                cached = self._create_object_layer(
                    size,
                    object_state,
                    box,
                    boxes,
                    job,
                    progress=progress,
                )
            elif box == final_box:
                cached = self._layer_cache.get(cache_key)
                if cached is None:
                    cached = self._create_object_layer(
                        size,
                        object_state,
                        box,
                        boxes,
                        job,
                    )
                    self._layer_cache[cache_key] = cached
            else:
                cached = self._create_object_layer(
                    size,
                    object_state,
                    box,
                    boxes,
                    job,
                )
            layer = cached.image
            opacity = self._opacity(object_state, event, progress, is_new)
            reveal_box = (
                LayoutBox(
                    x=cached.position[0],
                    y=cached.position[1],
                    width=max(1, layer.width),
                    height=max(1, layer.height),
                )
                if object_state.kind == "connector"
                else box
            )
            if event is not None and event.strategy in {
                "handwriting", "write_left_to_right", "grow_edge",
                "stroke_reveal", "reveal_cell", "reveal_cell_by_cell",
                "timeline_trace", "cycle_trace", "cause_effect_flow",
                "trace_steps", "decision_flow",
            } and progress < 1 and object_state.kind != "connector":
                layer = self._horizontal_reveal(
                    layer.copy(),
                    cached.position,
                    reveal_box,
                    progress,
                )
            if event is not None and event.strategy in {
                "grow_bar", "grow_distribution"
            } and progress < 1:
                layer = self._vertical_reveal(
                    layer.copy(),
                    cached.position,
                    reveal_box,
                    progress,
                )
            if opacity < 1:
                if layer is cached.image:
                    layer = layer.copy()
                alpha = layer.getchannel("A").point(
                    lambda value: round(value * opacity)
                )
                layer.putalpha(alpha)
            canvas.alpha_composite(layer, dest=cached.position)
        return canvas

    def _create_object_layer(
        self,
        size: tuple[int, int],
        state: ObjectState,
        box: LayoutBox,
        boxes: dict[str, LayoutBox],
        job: RenderJob,
        progress: float = 1.0,
    ) -> _CachedLayer:
        """Draw one object once and retain only its nontransparent region."""

        full_layer = Image.new("RGBA", size, (0, 0, 0, 0))
        self._draw_object(full_layer, state, box, boxes, job, progress)
        bounds = full_layer.getbbox()
        if bounds is None:
            return _CachedLayer(Image.new("RGBA", (1, 1)), (0, 0))
        return _CachedLayer(
            image=full_layer.crop(bounds),
            position=(bounds[0], bounds[1]),
        )

    def _draw_object(
        self,
        layer: Image.Image,
        state: ObjectState,
        box: LayoutBox,
        boxes: dict[str, LayoutBox],
        job: RenderJob,
        progress: float = 1.0,
    ) -> None:
        """Draw one semantic kind without inferring educational meaning."""

        draw = ImageDraw.Draw(layer)
        coordinates = self._coords(box)
        style = self._design.resolve(
            state.kind,
            state.style_token,
            dimmed=state.lifecycle is ObjectLifecycle.DIMMED,
        )
        ink = style.ink
        if state.kind == "connector":
            self._draw_connector(draw, state, boxes, ink, progress)
            return
        if state.kind == "semantic_asset":
            self._draw_semantic_asset(layer, state, box, job)
            return
        if state.kind == "histogram_bar":
            draw.rounded_rectangle(
                coordinates,
                radius=10,
                outline=ink,
                width=style.stroke_width,
            )
            inner = (coordinates[0] + 5, coordinates[1] + 5, coordinates[2] - 5, coordinates[3] - 5)
            draw.rectangle(inner, fill=style.accent[:3] + (190,))
            self._draw_centered_text(
                draw,
                coordinates,
                str(state.content.get("label", "")),
                26,
                ink,
            )
            return
        if state.kind in {"label", "text", "annotation", "equation"}:
            self._draw_centered_text(
                draw,
                coordinates,
                self._label(state),
                style.font_size,
                ink,
            )
            if state.kind == "annotation":
                draw.line(
                    (coordinates[0], coordinates[3] - 4, coordinates[2], coordinates[3] - 4),
                    fill=ink,
                    width=3,
                )
            return

        fill = style.fill
        if state.lifecycle is ObjectLifecycle.EMPHASIZED:
            draw.rounded_rectangle(
                self._expand(coordinates, 12),
                radius=22,
                fill=self.HIGHLIGHT,
            )
        draw.rounded_rectangle(
            coordinates,
            radius=style.corner_radius,
            fill=fill,
            outline=ink,
            width=style.stroke_width,
        )
        label = self._label(state)
        if label:
            if state.child_ids:
                header_height = min(
                    86,
                    max(52, round(box.height * 0.2)),
                )
                header = (
                    coordinates[0] + 16,
                    coordinates[1] + 8,
                    coordinates[2] - 16,
                    min(coordinates[3] - 8, coordinates[1] + header_height),
                )
                self._draw_centered_text(
                    draw,
                    header,
                    label,
                    style.font_size,
                    ink,
                )
            else:
                self._draw_centered_text(
                    draw,
                    coordinates,
                    label,
                    style.font_size,
                    ink,
                )

    def _draw_connector(
        self,
        draw: ImageDraw.ImageDraw,
        state: ObjectState,
        boxes: dict[str, LayoutBox],
        ink: tuple[int, int, int, int],
        progress: float = 1.0,
    ) -> None:
        """Draw a routed arrow between declared semantic endpoints."""

        source_id = state.content.get("source_id")
        target_id = state.content.get("target_id")
        if not isinstance(source_id, str) or not isinstance(target_id, str):
            return
        if source_id not in boxes or target_id not in boxes:
            return
        source = boxes[source_id]
        target = boxes[target_id]
        start, end, horizontal = self._connector_anchors(source, target)
        obstacles = [
            box
            for object_id, box in boxes.items()
            if object_id not in {source_id, target_id, state.object_id}
            and box.width > 4
            and box.height > 4
            and not self._contains_point(box, self._center(source))
            and not self._contains_point(box, self._center(target))
        ]
        points = self._orthogonal_route(
            start,
            end,
            horizontal,
            obstacles,
            boxes,
        )
        points = self._partial_polyline(points, progress)
        if len(points) < 2:
            return
        draw.line(points, fill=ink, width=5, joint="curve")
        direction_x = points[-1][0] - points[-2][0]
        direction_y = points[-1][1] - points[-2][1]
        length = max(1.0, (direction_x**2 + direction_y**2) ** 0.5)
        unit_x, unit_y = direction_x / length, direction_y / length
        perpendicular_x, perpendicular_y = -unit_y, unit_x
        arrow = [
            points[-1],
            (
                points[-1][0] - unit_x * 18 + perpendicular_x * 9,
                points[-1][1] - unit_y * 18 + perpendicular_y * 9,
            ),
            (
                points[-1][0] - unit_x * 18 - perpendicular_x * 9,
                points[-1][1] - unit_y * 18 - perpendicular_y * 9,
            ),
        ]
        if progress >= 0.985:
            draw.polygon(arrow, fill=ink)
        else:
            tip = points[-1]
            draw.ellipse(
                (tip[0] - 7, tip[1] - 7, tip[0] + 7, tip[1] + 7),
                fill=self.ACCENT,
            )

    @staticmethod
    def _partial_polyline(
        points: list[tuple[float, float]],
        progress: float,
    ) -> list[tuple[float, float]]:
        """Return the distance-proportional prefix of a routed connector."""

        clamped = min(1.0, max(0.0, progress))
        lengths = [
            ((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2) ** 0.5
            for first, second in zip(points, points[1:])
        ]
        remaining = sum(lengths) * clamped
        result = [points[0]]
        for first, second, length in zip(points, points[1:], lengths):
            if remaining >= length:
                result.append(second)
                remaining -= length
                continue
            if length > 0 and remaining > 0:
                ratio = remaining / length
                result.append(
                    (
                        first[0] + (second[0] - first[0]) * ratio,
                        first[1] + (second[1] - first[1]) * ratio,
                    )
                )
            break
        return result

    @staticmethod
    def _center(box: LayoutBox) -> tuple[float, float]:
        """Return the center point of a layout box."""

        return box.x + box.width / 2, box.y + box.height / 2

    def _connector_anchors(
        self,
        source: LayoutBox,
        target: LayoutBox,
    ) -> tuple[tuple[float, float], tuple[float, float], bool]:
        """Attach a connector to facing object boundaries, never centers."""

        source_center = self._center(source)
        target_center = self._center(target)
        delta_x = target_center[0] - source_center[0]
        delta_y = target_center[1] - source_center[1]
        vertically_separated = (
            source.y + source.height <= target.y
            or target.y + target.height <= source.y
        )
        horizontally_separated = (
            source.x + source.width <= target.x
            or target.x + target.width <= source.x
        )
        if vertically_separated:
            horizontal = False
        elif horizontally_separated:
            horizontal = True
        else:
            horizontal = abs(delta_x) >= abs(delta_y)
        if horizontal:
            if delta_x >= 0:
                return (
                    (source.x + source.width, source_center[1]),
                    (target.x, target_center[1]),
                    True,
                )
            return (
                (source.x, source_center[1]),
                (target.x + target.width, target_center[1]),
                True,
            )
        if delta_y >= 0:
            return (
                (source_center[0], source.y + source.height),
                (target_center[0], target.y),
                False,
            )
        return (
            (source_center[0], source.y),
            (target_center[0], target.y + target.height),
            False,
        )

    def _orthogonal_route(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        horizontal: bool,
        obstacles: list[LayoutBox],
        boxes: dict[str, LayoutBox],
    ) -> list[tuple[float, float]]:
        """Choose the shortest orthogonal route that avoids visible nodes."""

        minimum_x = min(box.x for box in boxes.values())
        maximum_x = max(box.x + box.width for box in boxes.values())
        minimum_y = min(box.y for box in boxes.values())
        maximum_y = max(box.y + box.height for box in boxes.values())
        if horizontal:
            middle_x = (start[0] + end[0]) / 2
            candidates = [
                [start, (middle_x, start[1]), (middle_x, end[1]), end],
                [
                    start,
                    (start[0], max(12.0, minimum_y - 28.0)),
                    (end[0], max(12.0, minimum_y - 28.0)),
                    end,
                ],
                [
                    start,
                    (start[0], maximum_y + 28.0),
                    (end[0], maximum_y + 28.0),
                    end,
                ],
            ]
        else:
            middle_y = (start[1] + end[1]) / 2
            candidates = [
                [start, (start[0], middle_y), (end[0], middle_y), end],
                [
                    start,
                    (max(12.0, minimum_x - 28.0), start[1]),
                    (max(12.0, minimum_x - 28.0), end[1]),
                    end,
                ],
                [
                    start,
                    (maximum_x + 28.0, start[1]),
                    (maximum_x + 28.0, end[1]),
                    end,
                ],
            ]
        normalized = [self._deduplicate_points(route) for route in candidates]
        return min(
            normalized,
            key=lambda route: self._route_score(route, obstacles),
        )

    def _route_score(
        self,
        points: list[tuple[float, float]],
        obstacles: list[LayoutBox],
    ) -> float:
        """Penalize obstacle crossings before route length and bend count."""

        crossings = sum(
            self._segment_intersects_box(first, second, obstacle)
            for first, second in zip(points, points[1:])
            for obstacle in obstacles
        )
        length = sum(
            abs(second[0] - first[0]) + abs(second[1] - first[1])
            for first, second in zip(points, points[1:])
        )
        return crossings * 1_000_000 + length + max(0, len(points) - 2) * 20

    @staticmethod
    def _segment_intersects_box(
        first: tuple[float, float],
        second: tuple[float, float],
        box: LayoutBox,
    ) -> bool:
        """Detect an orthogonal segment crossing a padded obstacle."""

        padding = 10.0
        left = box.x - padding
        right = box.x + box.width + padding
        top = box.y - padding
        bottom = box.y + box.height + padding
        if abs(first[1] - second[1]) < 1e-6:
            segment_left, segment_right = sorted((first[0], second[0]))
            return (
                top <= first[1] <= bottom
                and segment_right >= left
                and segment_left <= right
            )
        segment_top, segment_bottom = sorted((first[1], second[1]))
        return (
            left <= first[0] <= right
            and segment_bottom >= top
            and segment_top <= bottom
        )

    @staticmethod
    def _contains_point(
        box: LayoutBox,
        point: tuple[float, float],
    ) -> bool:
        """Return whether a point is inside a layout box."""

        return (
            box.x <= point[0] <= box.x + box.width
            and box.y <= point[1] <= box.y + box.height
        )

    @staticmethod
    def _deduplicate_points(
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Remove zero-length route segments while retaining both anchors."""

        result: list[tuple[float, float]] = []
        for point in points:
            if not result or point != result[-1]:
                result.append(point)
        return result if len(result) >= 2 else [points[0], points[-1]]

    def _draw_semantic_asset(
        self,
        layer: Image.Image,
        state: ObjectState,
        box: LayoutBox,
        job: RenderJob,
    ) -> None:
        """Render a resolved SVG semantic asset when available."""

        asset = next(
            (
                item for item in job.assets.assets
                if item.asset_id == f"asset_{state.object_id}"
            ),
            None,
        )
        if asset is None or asset.path.startswith("template://"):
            draw = ImageDraw.Draw(layer)
            draw.rounded_rectangle(self._coords(box), radius=18, outline=self.INK, width=5)
            self._draw_centered_text(draw, self._coords(box), self._label(state), 30, self.INK)
            return
        is_svg = asset.mime_type == "image/svg+xml"
        renderable = RenderableObject(
            object_id=state.object_id,
            type="svg" if is_svg else "image",
            content=asset.path,
            x=round(box.x + box.width / 2),
            y=round(box.y + box.height / 2),
            width=max(1, round(box.width)),
            height=max(1, round(box.height)),
            animation="none",
            start_time=0,
            end_time=1,
        )
        renderer = self._svg_renderer if is_svg else self._image_renderer
        renderer.render(layer, renderable)

    def _apply_camera(
        self,
        canvas: Image.Image,
        cue: CameraCue,
        timestamp: float,
    ) -> Image.Image:
        """Apply restrained, eased camera choreography after view fitting."""

        progress = self._cue_progress(cue, timestamp)
        progress = progress * progress * (3.0 - 2.0 * progress)
        zoom = {
            CameraOperation.PAN: 0.015,
            CameraOperation.TRACK: 0.025,
            CameraOperation.FOCUS: 0.045,
            CameraOperation.ZOOM: 0.060,
        }.get(cue.operation, 0.0)
        inset_x = canvas.width * zoom * progress
        inset_y = canvas.height * zoom * progress
        target_center_x = float(
            cue.parameters.get("target_x", canvas.width / 2)
        ) + float(cue.parameters.get("target_width", 0)) / 2
        direction = -1.0 if target_center_x < canvas.width / 2 else 1.0
        pan = (
            canvas.width * 0.012 * progress * direction
            if cue.operation in {CameraOperation.PAN, CameraOperation.TRACK}
            else 0.0
        )
        left = max(0.0, min(canvas.width - 2 * inset_x, inset_x + pan))
        top = inset_y
        right = min(float(canvas.width), left + canvas.width - 2 * inset_x)
        bottom = canvas.height - inset_y
        crop = canvas.crop((round(left), round(top), round(right), round(bottom)))
        return crop.resize(canvas.size, Image.Resampling.LANCZOS)

    def _trace_payload(
        self,
        frame_number: int,
        timestamp: float,
        beat_id: str,
        state: VisualState,
        events: list[MotionEvent],
        frame_path: Path,
    ) -> dict[str, Any]:
        """Build one V2 frame trace record for batched persistence."""

        return {
            "frame_number": frame_number,
            "time_seconds": round(timestamp, 6),
            "beat_id": beat_id,
            "state_id": state.state_id,
            "output_path": frame_path.as_posix(),
            "objects": [
                {
                    "object_id": item.object_id,
                    "kind": item.kind,
                    "lifecycle": item.lifecycle.value,
                }
                for item in state.object_states.values()
            ],
            "events": [
                {
                    "event_id": event.event_id,
                    "strategy": event.strategy,
                    "progress": round(
                        self._event_progress(event, timestamp),
                        6,
                    ),
                }
                for event in events
            ],
        }

    @staticmethod
    def _save_png(canvas: Image.Image, frame_path: Path) -> None:
        """Save one lossless RGB frame outside the rendering thread."""

        canvas.convert("RGB").save(frame_path, "PNG")

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> None:
        """Reuse an identical encoded frame without changing sequence layout."""

        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)

    def _visible_context_ids(
        self,
        state: VisualState,
        cue: CameraCue | None,
    ) -> set[str]:
        """Select active semantic groups plus the context needed to explain them."""

        known = set(state.object_states)
        if cue is None or not cue.target_ids:
            return known
        anchors = {target for target in cue.target_ids if target in known}
        for target_id in list(anchors):
            target = state.object_states[target_id]
            if target.kind != "connector":
                continue
            for endpoint_key in ("source_id", "target_id"):
                endpoint = target.content.get(endpoint_key)
                if isinstance(endpoint, str) and endpoint in known:
                    anchors.add(endpoint)

        selected: set[str] = set()
        for anchor in anchors:
            root_id = anchor
            while state.object_states[root_id].parent_id is not None:
                parent_id = state.object_states[root_id].parent_id
                if parent_id is None or parent_id not in known:
                    break
                root_id = parent_id
            selected.update(self._state_descendants(state, root_id))

        changed = True
        while changed:
            changed = False
            for object_id, item in state.object_states.items():
                if item.kind != "connector" or object_id in selected:
                    continue
                source_id = item.content.get("source_id")
                target_id = item.content.get("target_id")
                if source_id in selected and target_id in selected:
                    selected.add(object_id)
                    changed = True
        return selected or known

    @staticmethod
    def _state_descendants(state: VisualState, root_id: str) -> set[str]:
        """Return one declared semantic subtree, including its root."""

        result: set[str] = set()
        pending = [root_id]
        while pending:
            object_id = pending.pop()
            if object_id in result or object_id not in state.object_states:
                continue
            result.add(object_id)
            pending.extend(state.object_states[object_id].child_ids)
        return result

    def _fit_view_boxes(
        self,
        state: VisualState,
        nodes: list[LaidOutNode],
        raw_boxes: dict[str, LayoutBox],
        visible_ids: set[str],
        viewport: Viewport,
        cue: CameraCue | None,
    ) -> dict[str, LayoutBox]:
        """Fit active semantic context before drawing to retain native detail."""

        kinds = {node.object_id: node.kind for node in nodes}
        packed_boxes = self._pack_context_boxes(
            state,
            raw_boxes,
            visible_ids,
        )
        content_ids = [
            object_id
            for object_id in visible_ids
            if object_id in packed_boxes and kinds.get(object_id) != "connector"
        ]
        if not content_ids:
            content_ids = [
                object_id
                for object_id in visible_ids
                if object_id in packed_boxes
            ]
        if not content_ids:
            return {}
        left = min(packed_boxes[item].x for item in content_ids)
        top = min(packed_boxes[item].y for item in content_ids)
        right = max(
            packed_boxes[item].x + packed_boxes[item].width
            for item in content_ids
        )
        bottom = max(
            packed_boxes[item].y + packed_boxes[item].height
            for item in content_ids
        )
        safe_ratio = 0.08
        if cue is not None:
            configured = cue.parameters.get("safe_margin", safe_ratio)
            if isinstance(configured, (int, float)):
                safe_ratio = max(0.04, min(0.2, float(configured)))
        margin = max(
            float(viewport.margin),
            min(viewport.width, viewport.height) * safe_ratio,
        )
        available_width = max(1.0, viewport.width - 2 * margin)
        available_height = max(1.0, viewport.height - 2 * margin)
        content_width = max(1.0, right - left)
        content_height = max(1.0, bottom - top)
        scale = min(
            available_width / content_width,
            available_height / content_height,
        )
        offset_x = margin + (available_width - content_width * scale) / 2
        offset_y = margin + (available_height - content_height * scale) / 2
        return {
            object_id: LayoutBox(
                x=offset_x + (box.x - left) * scale,
                y=offset_y + (box.y - top) * scale,
                width=max(1.0, box.width * scale),
                height=max(1.0, box.height * scale),
                rotation=box.rotation,
            )
            for object_id, box in packed_boxes.items()
            if object_id in visible_ids
        }

    def _pack_context_boxes(
        self,
        state: VisualState,
        raw_boxes: dict[str, LayoutBox],
        visible_ids: set[str],
    ) -> dict[str, LayoutBox]:
        """Compact connected top-level groups into a readable teaching flow."""

        roots = [
            object_id
            for object_id in visible_ids
            if object_id in state.object_states
            and state.object_states[object_id].parent_id is None
            and state.object_states[object_id].kind != "connector"
            and object_id in raw_boxes
        ]
        if len(roots) <= 1:
            return {
                object_id: box
                for object_id, box in raw_boxes.items()
                if object_id in visible_ids
            }
        root_by_object: dict[str, str] = {}
        for object_id in visible_ids:
            if object_id not in state.object_states:
                continue
            root_id = object_id
            while state.object_states[root_id].parent_id is not None:
                parent_id = state.object_states[root_id].parent_id
                if parent_id is None or parent_id not in state.object_states:
                    break
                root_id = parent_id
            root_by_object[object_id] = root_id
        edges: set[tuple[str, str]] = set()
        for object_id in visible_ids:
            item = state.object_states.get(object_id)
            if item is None or item.kind != "connector":
                continue
            source_id = item.content.get("source_id")
            target_id = item.content.get("target_id")
            if not isinstance(source_id, str) or not isinstance(target_id, str):
                continue
            source_root = root_by_object.get(source_id)
            target_root = root_by_object.get(target_id)
            if (
                source_root in roots
                and target_root in roots
                and source_root != target_root
            ):
                edges.add((source_root, target_root))
        order = self._topological_roots(roots, edges)
        maximum_width = max(raw_boxes[root].width for root in roots)
        cursor_y = 0.0
        translations: dict[str, tuple[float, float]] = {}
        for root_id in order:
            root_box = raw_boxes[root_id]
            target_x = (maximum_width - root_box.width) / 2
            translations[root_id] = (
                target_x - root_box.x,
                cursor_y - root_box.y,
            )
            cursor_y += root_box.height + 96.0
        packed: dict[str, LayoutBox] = {}
        for object_id in visible_ids:
            if object_id not in raw_boxes:
                continue
            box = raw_boxes[object_id]
            root_id = root_by_object.get(object_id)
            if root_id in translations:
                delta_x, delta_y = translations[root_id]
                packed[object_id] = LayoutBox(
                    x=box.x + delta_x,
                    y=box.y + delta_y,
                    width=box.width,
                    height=box.height,
                    rotation=box.rotation,
                )
            else:
                packed[object_id] = LayoutBox(
                    x=maximum_width / 2,
                    y=max(0.0, cursor_y / 2),
                    width=1.0,
                    height=1.0,
                    rotation=box.rotation,
                )
        return packed

    @staticmethod
    def _topological_roots(
        roots: list[str],
        edges: set[tuple[str, str]],
    ) -> list[str]:
        """Order connected semantic groups from prerequisite to result."""

        incoming = {root: 0 for root in roots}
        outgoing = {root: set() for root in roots}
        for source, target in edges:
            if target not in outgoing[source]:
                outgoing[source].add(target)
                incoming[target] += 1
        ready = sorted(root for root, count in incoming.items() if count == 0)
        result: list[str] = []
        while ready:
            root = ready.pop(0)
            result.append(root)
            for target in sorted(outgoing[root]):
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
                    ready.sort()
        return result + sorted(root for root in roots if root not in result)

    def _frame_signature(
        self,
        beat_id: str,
        events: list[MotionEvent],
        cue: CameraCue | None,
        timestamp: float,
    ) -> tuple[Any, ...]:
        """Describe every time-varying input that can alter rendered pixels."""

        event_progress = tuple(
            round(self._event_progress(event, timestamp), 8)
            for event in events
        )
        camera_progress = (
            round(self._cue_progress(cue, timestamp), 8)
            if cue is not None and cue.operation not in {
                CameraOperation.FIT,
                CameraOperation.HOLD,
            }
            else 1.0
        )
        return beat_id, event_progress, camera_progress

    def _font(self, size: int) -> ImageFont.ImageFont:
        """Load and cache a consistent TrueType font with fallback."""

        if size in self._font_cache:
            return self._font_cache[size]
        candidates = [
            PROJECT_ROOT / "assets" / "fonts" / "DejaVuSans.ttf",
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        font: ImageFont.ImageFont = ImageFont.load_default()
        for path in candidates:
            if path.is_file():
                font = ImageFont.truetype(str(path), size=size)
                break
        self._font_cache[size] = font
        return font

    def _draw_centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        text: str,
        size: int,
        fill: tuple[int, int, int, int],
    ) -> None:
        """Fit, wrap, and center text without crossing its layout box."""

        if not text:
            return
        horizontal_padding = 12
        vertical_padding = 8
        available_width = max(
            1,
            box[2] - box[0] - 2 * horizontal_padding,
        )
        available_height = max(
            1,
            box[3] - box[1] - 2 * vertical_padding,
        )
        resolved: tuple[
            ImageFont.ImageFont,
            list[str],
            tuple[int, int, int, int],
        ] | None = None
        for font_size in range(size, self.MIN_FONT_SIZE - 1, -2):
            font = self._font(font_size)
            lines = self._wrap_text(draw, text, font, available_width)
            display = "\n".join(lines)
            bounds = draw.multiline_textbbox(
                (0, 0),
                display,
                font=font,
                spacing=4,
                align="center",
            )
            if (
                bounds[2] - bounds[0] <= available_width
                and bounds[3] - bounds[1] <= available_height
            ):
                resolved = font, lines, bounds
                break
        if resolved is None:
            font = self._font(self.MIN_FONT_SIZE)
            lines = self._wrap_text(draw, text, font, available_width)
            line_bounds = draw.textbbox((0, 0), "Ag", font=font)
            line_height = max(1, line_bounds[3] - line_bounds[1] + 4)
            maximum_lines = max(1, available_height // line_height)
            was_truncated = len(lines) > maximum_lines
            lines = lines[:maximum_lines]
            if was_truncated and lines:
                lines[-1] = self._ellipsize(
                    draw,
                    lines[-1],
                    font,
                    available_width,
                )
            display = "\n".join(lines)
            bounds = draw.multiline_textbbox(
                (0, 0),
                display,
                font=font,
                spacing=4,
                align="center",
            )
            resolved = font, lines, bounds
        font, lines, bounds = resolved
        display = "\n".join(lines)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        draw.multiline_text(
            (
                box[0] + (box[2] - box[0] - text_width) / 2 - bounds[0],
                box[1] + (box[3] - box[1] - text_height) / 2 - bounds[1],
            ),
            display,
            font=font,
            fill=fill,
            spacing=4,
            align="center",
        )

    @staticmethod
    def _wrap_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
        maximum_width: int,
    ) -> list[str]:
        """Greedily wrap words, splitting only tokens wider than the box."""

        words = " ".join(text.split()).split(" ")
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if draw.textlength(candidate, font=font) <= maximum_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            if draw.textlength(word, font=font) <= maximum_width:
                current = word
                continue
            chunk = ""
            for character in word:
                candidate_chunk = chunk + character
                if (
                    chunk
                    and draw.textlength(candidate_chunk, font=font)
                    > maximum_width
                ):
                    lines.append(chunk)
                    chunk = character
                else:
                    chunk = candidate_chunk
            current = chunk
        if current:
            lines.append(current)
        return lines or [""]

    @staticmethod
    def _ellipsize(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
        maximum_width: int,
    ) -> str:
        """Shorten one final line to a measured typographic boundary."""

        ellipsis = "…"
        candidate = text.rstrip()
        while (
            candidate
            and draw.textlength(candidate + ellipsis, font=font)
            > maximum_width
        ):
            candidate = candidate[:-1].rstrip()
        return candidate + ellipsis

    @staticmethod
    def _label(state: ObjectState) -> str:
        """Return declared display content for a semantic object."""

        return str(
            state.content.get("text")
            or state.content.get("label")
            or state.content.get("value")
            or ""
        )

    @staticmethod
    def _coords(box: LayoutBox) -> tuple[int, int, int, int]:
        """Convert a layout box to Pillow coordinates."""

        return (
            round(box.x),
            round(box.y),
            round(box.x + box.width),
            round(box.y + box.height),
        )

    @staticmethod
    def _expand(box: tuple[int, int, int, int], amount: int) -> tuple[int, int, int, int]:
        """Expand a Pillow box equally in each direction."""

        return (box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount)

    @staticmethod
    def _render_priority(kind: str, has_children: bool = False) -> int:
        """Draw group backgrounds, connectors, cards, then text overlays."""

        if has_children:
            return 0
        if kind == "connector":
            return 1
        if kind in {"label", "text", "annotation", "equation"}:
            return 3
        return 2

    @staticmethod
    def _event_progress(event: MotionEvent | None, timestamp: float) -> float:
        """Return clamped linear progress for one motion event."""

        if event is None:
            return 1.0
        if timestamp <= event.start_time:
            return 0.0
        return min(1.0, (timestamp - event.start_time) / event.duration)

    @staticmethod
    def _cue_progress(cue: CameraCue, timestamp: float) -> float:
        """Return clamped progress for camera interpolation."""

        if timestamp <= cue.start_time:
            return 0.0
        return min(1.0, (timestamp - cue.start_time) / cue.duration)

    @staticmethod
    def _opacity(
        state: ObjectState,
        event: MotionEvent | None,
        progress: float,
        is_new: bool,
    ) -> float:
        """Calculate opacity for lifecycle and transition semantics."""

        opacity = 0.28 if state.lifecycle is ObjectLifecycle.DIMMED else 1.0
        if event is None:
            return opacity
        if event.strategy in {"fade_out", "hand_erase", "dim"}:
            return opacity * (1.0 - 0.72 * progress)
        if is_new or event.strategy in {
            "fade_in", "node_pop", "outline_then_label", "duplicate",
            "reveal_components", "stage_flow",
        }:
            return opacity * progress
        return opacity

    @staticmethod
    def _horizontal_reveal(
        layer: Image.Image,
        position: tuple[int, int],
        box: LayoutBox,
        progress: float,
    ) -> Image.Image:
        """Mask a layer from left to right."""

        mask = Image.new("L", layer.size, 0)
        draw = ImageDraw.Draw(mask)
        left = box.x - position[0]
        top = box.y - position[1]
        right = left + box.width * progress
        draw.rectangle((left, top, right, top + box.height), fill=255)
        alpha = Image.composite(layer.getchannel("A"), Image.new("L", layer.size, 0), mask)
        layer.putalpha(alpha)
        return layer

    @staticmethod
    def _vertical_reveal(
        layer: Image.Image,
        position: tuple[int, int],
        box: LayoutBox,
        progress: float,
    ) -> Image.Image:
        """Mask a layer upward from its baseline."""

        mask = Image.new("L", layer.size, 0)
        draw = ImageDraw.Draw(mask)
        left = box.x - position[0]
        top = box.y - position[1] + box.height * (1.0 - progress)
        bottom = box.y - position[1] + box.height
        draw.rectangle((left, top, left + box.width, bottom), fill=255)
        alpha = Image.composite(layer.getchannel("A"), Image.new("L", layer.size, 0), mask)
        layer.putalpha(alpha)
        return layer

    @staticmethod
    def _interpolate_box(start: LayoutBox, end: LayoutBox, progress: float) -> LayoutBox:
        """Interpolate geometry for move, resize, and morph transitions."""

        return LayoutBox(
            x=start.x + (end.x - start.x) * progress,
            y=start.y + (end.y - start.y) * progress,
            width=start.width + (end.width - start.width) * progress,
            height=start.height + (end.height - start.height) * progress,
            rotation=start.rotation + (end.rotation - start.rotation) * progress,
        )

    @staticmethod
    def _active_beat(
        timestamp: float,
        beat_order: list[str],
        starts: dict[str, float],
    ) -> str:
        """Select the latest beat that has started."""

        active = beat_order[0]
        for beat_id in beat_order:
            if starts[beat_id] <= timestamp + 1e-9:
                active = beat_id
            else:
                break
        return active

    @staticmethod
    def _beat_starts(job: RenderJob) -> dict[str, float]:
        """Derive deterministic beat starts from camera and motion plans."""

        starts: dict[str, float] = {}
        for cue in job.camera.cues:
            starts[cue.beat_id] = min(starts.get(cue.beat_id, cue.start_time), cue.start_time)
        for event in job.motion.events:
            starts[event.beat_id] = min(starts.get(event.beat_id, event.start_time), event.start_time)
        for state in job.document.states:
            starts.setdefault(state.beat_id, 0.0)
        return starts

    @staticmethod
    def _sample_frames(job: RenderJob) -> set[int]:
        """Select first, middle, and last frames for each manifest scene."""

        samples: set[int] = set()
        for scene in job.manifest.scenes:
            samples.update(
                {
                    scene.frame_start,
                    (scene.frame_start + scene.frame_end) // 2,
                    scene.frame_end,
                }
            )
        return samples

    def _flatten(self, root: LaidOutNode | None) -> list[LaidOutNode]:
        """Flatten a layout tree in stable pre-order."""

        if root is None:
            return []
        result = [root]
        for child in root.children:
            result.extend(self._flatten(child))
        return result

    @staticmethod
    def _aspect_crop(
        left: float,
        top: float,
        right: float,
        bottom: float,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        """Expand and clamp a target crop to the canvas aspect ratio."""

        left = max(0.0, left)
        top = max(0.0, top)
        right = min(float(width), right)
        bottom = min(float(height), bottom)
        target_ratio = width / height
        crop_width = max(1.0, right - left)
        crop_height = max(1.0, bottom - top)
        if crop_width / crop_height < target_ratio:
            expanded = crop_height * target_ratio
            center = (left + right) / 2
            left, right = center - expanded / 2, center + expanded / 2
        else:
            expanded = crop_width / target_ratio
            center = (top + bottom) / 2
            top, bottom = center - expanded / 2, center + expanded / 2
        shift_x = max(0.0, -left) - max(0.0, right - width)
        shift_y = max(0.0, -top) - max(0.0, bottom - height)
        return (
            max(0.0, left + shift_x),
            max(0.0, top + shift_y),
            min(float(width), right + shift_x),
            min(float(height), bottom + shift_y),
        )

    @staticmethod
    def _working_path(path: Path) -> Path:
        """Resolve configured relative output folders from project root."""

        return path if path.is_absolute() else PROJECT_ROOT / path
