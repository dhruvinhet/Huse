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

        geometry = self._geometry_index(layout)
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
            targets = (
                intent.target_ids
                if intent and intent.target_ids
                else beat.operations[-1].target_ids
            )
            operation = CameraOperation(
                intent.operation
                if intent
                else self._operation_for_purpose(beat.purpose)
            )
            target_bounds = self._target_bounds(targets, geometry)
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
    def _geometry_index(layout: LayoutPlan) -> dict[str, object]:
        """Index latest known geometry for every semantic object."""

        index: dict[str, object] = {}

        def visit(node: object) -> None:
            object_id = getattr(node, "object_id")
            index[object_id] = getattr(node, "box")
            for child in getattr(node, "children"):
                visit(child)

        for root in layout.state_roots.values():
            visit(root)
        return index

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
