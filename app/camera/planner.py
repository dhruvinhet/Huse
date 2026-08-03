"""Create stable fit, focus, and tracking cues from storyboard intent."""

from collections import defaultdict

from app.domain.camera import CameraCue, CameraOperation, CameraPlan
from app.domain.layout import LayoutPlan
from app.domain.narration import AlignedAudio
from app.domain.storyboard import Storyboard


class SemanticCameraPlanner:
    """Plan restrained camera cues aligned with educational beats."""

    def plan(
        self,
        storyboard: Storyboard,
        layout: LayoutPlan,
        alignment: AlignedAudio,
    ) -> CameraPlan:
        """Create one camera cue per beat using declared intent."""

        geometry = self._geometry_index(layout, storyboard)
        windows: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for phrase in alignment.phrases:
            windows[phrase.beat_id].append((phrase.audio_start, phrase.audio_end))
        cues: list[CameraCue] = []
        for beat in storyboard.beats:
            intervals = windows.get(beat.beat_id)
            if not intervals:
                raise ValueError(
                    f"audio alignment is missing storyboard beat {beat.beat_id}"
                )
            start = min(item[0] for item in intervals)
            end = max(item[1] for item in intervals)
            intent = beat.camera_intent
            beat_geometry = geometry.get(beat.beat_id, {})
            targets = self._authoritative_targets(beat, beat_geometry)
            operation = CameraOperation(
                intent.operation
                if intent
                else self._operation_for_purpose(beat.purpose)
            )
            if operation in {CameraOperation.FIT, CameraOperation.HOLD}:
                targets = self._fit_targets(beat_geometry, targets)
            target_bounds = self._target_bounds(targets, beat_geometry)
            target_bounds.update(
                self._source_rect(
                    target_bounds,
                    layout.viewport.width,
                    layout.viewport.height,
                    beat.shot_plan.occupancy_target
                    if beat.shot_plan is not None
                    else (0.35, 0.70),
                )
            )
            cues.append(
                CameraCue(
                    cue_id=f"camera_{beat.beat_id}",
                    beat_id=beat.beat_id,
                    start_time=start,
                    duration=end - start,
                    operation=operation,
                    target_ids=targets,
                    parameters={
                        "emphasis": intent.emphasis if intent else 0.5,
                        "safe_margin": 0.08,
                        "purpose": beat.purpose,
                        "occupancy_target": list(
                            beat.shot_plan.occupancy_target
                            if beat.shot_plan is not None
                            else (0.35, 0.70)
                        ),
                        **target_bounds,
                        "ease": "smoothstep",
                    },
                )
            )
        return CameraPlan(duration=alignment.duration, cues=cues)

    @staticmethod
    def _operation_for_purpose(purpose: str) -> str:
        """Choose restrained camera choreography from teaching intent."""

        return {
            "introduce": "fit",
            "demonstrate": "track",
            "compare": "fit",
            "transform": "pan",
            "connect": "track",
            "emphasize": "zoom",
            "summarize": "fit",
        }.get(purpose, "fit")

    @staticmethod
    def _geometry_index(
        layout: LayoutPlan,
        storyboard: Storyboard,
    ) -> dict[str, dict[str, object]]:
        """Index geometry per beat so camera planning never uses stale history."""

        index: dict[str, dict[str, object]] = {}

        def visit(node: object, current: dict[str, object]) -> None:
            object_id = getattr(node, "object_id")
            current[object_id] = getattr(node, "box")
            for child in getattr(node, "children"):
                visit(child, current)

        for beat, root in zip(storyboard.beats, layout.state_roots.values()):
            current: dict[str, object] = {}
            visit(root, current)
            index[beat.beat_id] = current
        return index

    @staticmethod
    def _authoritative_targets(
        beat: object,
        geometry: dict[str, object],
    ) -> list[str]:
        """Prefer real focused objects over a model-supplied root target."""

        candidates = [
            target_id
            for cue in beat.attention
            for target_id in cue.target_ids
            if target_id in geometry
        ]
        if not candidates:
            candidates = [
                target_id
                for operation in reversed(beat.operations)
                for target_id in operation.target_ids
                if target_id in geometry
            ]
        if not candidates:
            candidates = [target_id for target_id in (beat.camera_intent.target_ids if beat.camera_intent else []) if target_id in geometry]
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _fit_targets(
        geometry: dict[str, object],
        targets: list[str],
    ) -> list[str]:
        """Use the active beat bounds for fit/hold instead of a single tiny cue."""

        if targets and len(targets) > 1:
            return targets
        return list(geometry)

    @staticmethod
    def _target_bounds(
        targets: list[str],
        geometry: dict[str, object],
    ) -> dict[str, float]:
        """Return the union of target boxes for renderer camera interpolation."""

        boxes = [geometry[target] for target in targets if target in geometry]
        if not boxes:
            return {}
        left = min(float(getattr(box, "x")) for box in boxes)
        top = min(float(getattr(box, "y")) for box in boxes)
        right = max(
            float(getattr(box, "x")) + float(getattr(box, "width"))
            for box in boxes
        )
        bottom = max(
            float(getattr(box, "y")) + float(getattr(box, "height"))
            for box in boxes
        )
        return {
            "target_x": left,
            "target_y": top,
            "target_width": right - left,
            "target_height": bottom - top,
        }

    @staticmethod
    def _source_rect(
        bounds: dict[str, float],
        width: int,
        height: int,
        occupancy_target: tuple[float, float],
    ) -> dict[str, float]:
        """Derive an aspect-correct source rectangle around target geometry."""

        if not bounds:
            return {
                "source_left": 0.0,
                "source_top": 0.0,
                "source_right": float(width),
                "source_bottom": float(height),
            }
        desired = max(0.35, min(0.70, sum(occupancy_target) / 2))
        target_width = max(1.0, bounds["target_width"])
        target_height = max(1.0, bounds["target_height"])
        crop_width = min(float(width), target_width / desired)
        crop_height = min(float(height), target_height / desired)
        aspect = width / height
        if crop_width / crop_height < aspect:
            crop_width = min(float(width), crop_height * aspect)
        else:
            crop_height = min(float(height), crop_width / aspect)
        center_x = bounds["target_x"] + target_width / 2
        center_y = bounds["target_y"] + target_height / 2
        left = max(0.0, min(float(width) - crop_width, center_x - crop_width / 2))
        top = max(0.0, min(float(height) - crop_height, center_y - crop_height / 2))
        return {
            "source_left": left,
            "source_top": top,
            "source_right": left + crop_width,
            "source_bottom": top + crop_height,
        }
