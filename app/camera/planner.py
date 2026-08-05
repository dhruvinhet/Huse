"""Create stable fit, focus, and tracking cues from storyboard intent."""

from collections import defaultdict

from app.domain.camera import CameraCue, CameraOperation, CameraPlan
from app.domain.layout import LayoutPlan
from app.domain.narration import AlignedAudio
from app.domain.storyboard import Storyboard


class SemanticCameraPlanner:
    """Plan restrained camera cues aligned with educational beats."""

    MINIMUM_VISIBLE_CHANGE_PX = 2.0

    def __init__(self, minimum_visible_change_px: float = 2.0) -> None:
        """Configure the minimum meaningful crop-edge movement in pixels."""

        if minimum_visible_change_px < 0:
            raise ValueError("minimum camera change cannot be negative")
        self._minimum_visible_change_px = minimum_visible_change_px

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
        previous_source = {
            "source_left": 0.0,
            "source_top": 0.0,
            "source_right": float(layout.viewport.width),
            "source_bottom": float(layout.viewport.height),
        }
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
            else:
                targets = self._focus_targets(
                    targets,
                    beat_geometry,
                    viewport_width=layout.viewport.width,
                )
            target_bounds = self._target_bounds(targets, beat_geometry)
            target_bounds.update(
                self._source_rect(
                    target_bounds,
                    layout.viewport.width,
                    layout.viewport.height,
                    beat.shot_plan.occupancy_target
                    if beat.shot_plan is not None
                    else (0.35, 0.70),
                    operation,
                )
            )
            measured_change = self._source_delta(target_bounds, previous_source)
            exempt = operation in {CameraOperation.FIT, CameraOperation.HOLD}
            source_width = max(
                1.0,
                target_bounds["source_right"] - target_bounds["source_left"],
            )
            source_height = max(
                1.0,
                target_bounds["source_bottom"] - target_bounds["source_top"],
            )
            final_target_occupancy = max(
                target_bounds.get("target_width", 0.0) / source_width,
                target_bounds.get("target_height", 0.0) / source_height,
            )
            target_edges_retained = self._target_edges_retained(target_bounds)
            source_start = {
                f"source_start_{key.removeprefix('source_')}": value
                for key, value in previous_source.items()
            }
            target_bounds.update(source_start)
            previous_source = {
                key: target_bounds[key]
                for key in (
                    "source_left",
                    "source_top",
                    "source_right",
                    "source_bottom",
                )
            }
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
                        "fit_hold_exempt": exempt,
                        "minimum_visible_change_px": self._minimum_visible_change_px,
                        "measured_visible_change_px": measured_change,
                        "final_target_occupancy": final_target_occupancy,
                        "target_edges_retained": target_edges_retained,
                        **target_bounds,
                        "ease": "smoothstep",
                    },
                )
            )
        return CameraPlan(duration=alignment.duration, cues=cues)

    @staticmethod
    def _source_delta(
        current: dict[str, float],
        previous: dict[str, float],
    ) -> float:
        """Return the maximum crop-edge displacement in output pixels."""

        return max(
            abs(float(current[key]) - float(previous[key]))
            for key in (
                "source_left", "source_top", "source_right", "source_bottom"
            )
        )

    @staticmethod
    def _target_edges_retained(bounds: dict[str, float]) -> bool:
        """Confirm the final camera crop contains every required target edge."""

        required = {
            "target_x", "target_y", "target_width", "target_height",
            "source_left", "source_top", "source_right", "source_bottom",
        }
        if not required.issubset(bounds):
            return True
        return (
            bounds["target_x"] >= bounds["source_left"] - 1e-6
            and bounds["target_y"] >= bounds["source_top"] - 1e-6
            and bounds["target_x"] + bounds["target_width"]
            <= bounds["source_right"] + 1e-6
            and bounds["target_y"] + bounds["target_height"]
            <= bounds["source_bottom"] + 1e-6
        )

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
    def _focus_targets(
        targets: list[str],
        geometry: dict[str, object],
        viewport_width: float | None = None,
    ) -> list[str]:
        """Keep focus shots on a readable local cluster, not the whole rail.

        Attention planning quite correctly names every related object for
        highlighting, but a camera should frame the smallest useful teaching
        cluster.  Connector and evidence objects are retained only when they
        are the sole available target; otherwise they consume crop area without
        improving readability.
        """

        ordered = list(dict.fromkeys(targets))
        semantic = [
            target
            for target in ordered
            if "connector" not in target.casefold()
            and "evidence" not in target.casefold()
            and target in geometry
        ]
        if semantic:
            candidates = semantic
        else:
            # Some focused states intentionally hide the content cards while
            # retaining their connectors and an explanatory callout.  Do not
            # interpret those connector-only attention cues as camera content:
            # recover the visible non-structural object from the laid-out
            # state, and use connectors only when nothing else is available.
            visible_content = [
                object_id
                for object_id in geometry
                if object_id.casefold() not in {"root", "scene", "nested_group"}
                and "connector" not in object_id.casefold()
                and not object_id.casefold().endswith("_root")
                and not object_id.casefold().endswith("_scene")
                and object_id.casefold() not in {"nested_group"}
            ]
            candidates = visible_content or [
                target for target in ordered if target in geometry
            ]
        if len(candidates) <= 1:
            return candidates
        if len(candidates) <= 3:
            # Three cards split across rows often make the union taller than
            # the 16:9 viewport.  A focus shot that keeps all three then
            # degenerates to a full-frame fit.  Select the smallest
            # contiguous pair so the source rectangle can produce a real
            # detail crop while retaining a local relationship.
            if len(candidates) == 3:
                pairs = [candidates[index:index + 2] for index in range(2)]
                candidates = min(
                    pairs,
                    key=lambda pair: (
                        SemanticCameraPlanner._target_bounds(pair, geometry).get(
                            "target_width", 0.0
                        )
                        * SemanticCameraPlanner._target_bounds(pair, geometry).get(
                            "target_height", 0.0
                        ),
                        pair,
                    ),
                )
            return candidates
        # Layout preserves deterministic teaching order in object IDs.  A
        # contiguous three-object window gives the camera a coherent local
        # explanation while subsequent beats move to the next window.
        best_window = candidates[:3]
        best_area = float("inf")
        for index in range(len(candidates) - 2):
            window = candidates[index:index + 3]
            bounds = SemanticCameraPlanner._target_bounds(window, geometry)
            area = bounds.get("target_width", 0.0) * bounds.get("target_height", 0.0)
            if area < best_area:
                best_area = area
                best_window = window
        return best_window

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
        operation: CameraOperation = CameraOperation.FIT,
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
        if operation in {
            CameraOperation.FOCUS,
            CameraOperation.ZOOM,
            CameraOperation.TRACK,
            CameraOperation.PAN,
        }:
            # Focus shots should make a visible compositional change even when
            # the requested occupancy range is permissive.  Use the declared
            # upper occupancy bound so a wide callout still gets a real crop
            # instead of falling back to a full-frame source rectangle.
            desired = max(desired, 0.70)
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
        result = {
            "source_left": left,
            "source_top": top,
            "source_right": left + crop_width,
            "source_bottom": top + crop_height,
        }
        result["effective_font_scale"] = min(
            4.0,
            max(1.0, width / max(1.0, crop_width)),
        )
        result["minimum_effective_font_size"] = 16.0
        return result
