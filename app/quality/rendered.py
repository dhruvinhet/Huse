"""Deterministic pixel checks that run even without a vision model."""

from dataclasses import dataclass
from pathlib import Path
from statistics import median

from PIL import Image, ImageChops

from app.domain.camera import CameraPlan
from app.domain.layout import LaidOutNode, LayoutPlan
from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityFinding,
    QualityReport,
)
from app.domain.rendering import FrameSequence
from app.domain.storyboard import Storyboard
from app.domain.visual_document import VisualDocument
from app.models.video_manifest import VideoManifest
from app.quality.policy import QualityPolicy


@dataclass(frozen=True, slots=True)
class _PixelMetrics:
    """Measured foreground geometry for one rendered frame."""

    ink_ratio: float
    occupancy: float
    bounds: tuple[int, int, int, int] | None
    width: int
    height: int
    edge_ink_ratio: float


class RenderedFrameQualityEvaluator:
    """Inspect opening, transition, and final pixels plus planned focal geometry."""

    def __init__(self, policy: QualityPolicy | None = None) -> None:
        self._policy = policy or QualityPolicy()

    def evaluate(
        self,
        frames: FrameSequence,
        context: dict[str, object] | None = None,
    ) -> QualityReport:
        """Inspect bounded deterministic samples from the rendered sequence."""

        context = context or {}
        findings: list[QualityFinding] = []
        for diagnostic in frames.diagnostics:
            if not diagnostic.startswith("semantic_asset_render_failed:"):
                continue
            parts = diagnostic.split(":", maxsplit=2)
            if len(parts) < 3 or not parts[1]:
                continue
            object_id = parts[1]
            findings.append(QualityFinding(
                code="semantic_asset_render_failed",
                severity=FindingSeverity.ERROR,
                artifact_id="rendered_frames",
                message=(
                    f"Semantic artwork for {object_id!r} could not be rendered; "
                    "a visible diagnostic fallback was painted instead."
                ),
                repair_target="assets",
                repair_scope="object",
                object_ids=[object_id],
                measured_value=0.0,
                required_value=1.0,
                patch_paths=["/assets"],
            ))
        folder = Path(frames.folder)
        opening_limit = min(frames.total_frames, max(1, round(frames.fps * 2)))
        opening_numbers = list(range(1, opening_limit + 1, max(1, frames.fps // 4)))
        opening_metrics = self._numbered_metrics(folder, frames, opening_numbers)
        first_visible = next(
            (
                (number - 1) / frames.fps
                for number, metrics in opening_metrics
                if self._substantive(metrics)
            ),
            None,
        )
        if opening_metrics and first_visible is None:
            findings.append(QualityFinding(
                code="rendered_opening_blank",
                severity=FindingSeverity.ERROR,
                artifact_id="rendered_frames",
                message="No substantive teaching visual appears in the first two seconds.",
                repair_target="renderer",
                repair_scope="frame",
                frame_numbers=[number for number, _ in opening_metrics],
                measured_value=0.0,
                required_value=1.0,
                patch_paths=["/render/opening"],
            ))

        manifest = context.get("manifest")
        scene_samples = self._scene_sample_numbers(
            manifest if isinstance(manifest, VideoManifest) else None
        )
        explicit_samples = [
            Path(path) for path in frames.sample_paths if Path(path).exists()
        ]
        sample_metrics = [
            (self._frame_number(path), self._metrics(path))
            for path in explicit_samples
        ]
        already_sampled = {number for number, _ in sample_metrics if number is not None}
        sample_metrics.extend(
            (number, metrics)
            for number, metrics in self._numbered_metrics(
                folder,
                frames,
                [number for number in scene_samples if number not in already_sampled],
            )
        )
        substantive = [
            (number, item)
            for number, item in sample_metrics
            if self._substantive(item)
        ]
        if sample_metrics and len(substantive) < max(1, len(sample_metrics) * 2 // 3):
            findings.append(QualityFinding(
                code="rendered_samples_blank",
                severity=FindingSeverity.ERROR,
                artifact_id="rendered_frames",
                message=(
                    f"Only {len(substantive)} of {len(sample_metrics)} opening, "
                    "transition, and final samples contain substantive content."
                ),
                repair_target="renderer",
                repair_scope="frame",
                frame_numbers=[
                    number for number, item in sample_metrics
                    if number is not None and not self._substantive(item)
                ],
                measured_value=(
                    len(substantive) / len(sample_metrics) if sample_metrics else 0.0
                ),
                required_value=2 / 3,
                patch_paths=["/render/samples"],
            ))

        occupancies = [item.occupancy for _, item in substantive]
        if occupancies:
            middle = median(occupancies)
            if middle < self._policy.minimum_rendered_occupancy:
                findings.append(QualityFinding(
                    code="rendered_canvas_underused",
                    severity=FindingSeverity.ERROR,
                    artifact_id="rendered_frames",
                    message=(
                        f"Median teaching-pixel occupancy {middle:.1%} is below "
                        f"{self._policy.minimum_rendered_occupancy:.1%}."
                    ),
                    repair_target="layout",
                    repair_scope="stage",
                    measured_value=middle,
                    required_value=self._policy.minimum_rendered_occupancy,
                    patch_paths=["/layout"],
                ))
            if middle > self._policy.maximum_rendered_occupancy:
                findings.append(QualityFinding(
                    code="rendered_canvas_overfilled",
                    severity=FindingSeverity.ERROR,
                    artifact_id="rendered_frames",
                    message=(
                        f"Median teaching-pixel occupancy {middle:.1%} exceeds "
                        f"{self._policy.maximum_rendered_occupancy:.1%}."
                    ),
                    repair_target="layout",
                    repair_scope="stage",
                    measured_value=middle,
                    required_value=self._policy.maximum_rendered_occupancy,
                    patch_paths=["/layout"],
                ))

        clipped = [
            number
            for number, item in substantive
            if item.edge_ink_ratio > 0.002
        ]
        if clipped:
            findings.append(QualityFinding(
                code="rendered_safe_area_clipped",
                severity=FindingSeverity.ERROR,
                artifact_id="rendered_frames",
                message=(
                    f"Foreground pixels touch the outer frame band in "
                    f"{len(clipped)} sampled frames."
                ),
                repair_target="layout",
                repair_scope="frame",
                frame_numbers=[number for number in clipped if number is not None],
                measured_value=float(len(clipped)),
                required_value=0.0,
                patch_paths=["/layout", "/camera"],
            ))

        if sample_metrics:
            self._planned_geometry_findings(context, findings)
        score = max(0.0, 1.0 - 0.18 * len(findings))
        return QualityReport(
            overall_score=score,
            scores={"rendered_pixel_quality": score},
            findings=findings,
            decision=(
                EvaluationDecision.REPAIR
                if findings
                else EvaluationDecision.PASS
            ),
        )

    def _planned_geometry_findings(
        self,
        context: dict[str, object],
        findings: list[QualityFinding],
    ) -> None:
        """Use renderer inputs to explain pixel risks that do not require OCR."""

        layout = context.get("layout")
        document = context.get("document")
        storyboard = context.get("storyboard")
        camera = context.get("camera")
        if not isinstance(layout, LayoutPlan) or not isinstance(document, VisualDocument):
            return
        states = {state.state_id: state for state in document.states}
        unreadable: list[str] = []
        output_scale = min(
            1.0,
            layout.viewport.width / 1920.0,
            layout.viewport.height / 1080.0,
        )
        required_font = self._policy.minimum_effective_font_px * output_scale
        for state_id, root in layout.state_roots.items():
            state = states.get(state_id)
            if state is None:
                continue
            for node in self._flatten(root):
                content = state.object_states.get(node.object_id)
                if content is None or node.kind == "connector":
                    continue
                paints_text = any(
                    isinstance(content.content.get(key), str)
                    and bool(str(content.content[key]).strip())
                    for key in ("label", "text", "value", "detail")
                )
                if not paints_text:
                    continue
                estimated_font = min(36.0, max(12.0, node.box.height * 0.22))
                if estimated_font < required_font:
                    unreadable.append(node.object_id)
        if unreadable:
            findings.append(QualityFinding(
                code="rendered_effective_font_too_small",
                severity=FindingSeverity.ERROR,
                artifact_id="rendered_frames",
                message=(
                    f"{len(set(unreadable))} text-bearing objects have an "
                    f"estimated effective font below "
                    f"{required_font:.1f}px for this output resolution."
                ),
                repair_target="layout",
                repair_scope="object",
                object_ids=sorted(set(unreadable)),
                measured_value=float(len(set(unreadable))),
                required_value=0.0,
                patch_paths=["/layout"],
            ))

        if not isinstance(camera, CameraPlan) or not isinstance(storyboard, Storyboard):
            return
        beat_ids = {beat.beat_id for beat in storyboard.beats}
        missing_targets = [
            target
            for cue in camera.cues
            if cue.beat_id in beat_ids
            for target in cue.target_ids
            if all(target not in state.object_states for state in document.states)
        ]
        if missing_targets:
            findings.append(QualityFinding(
                code="rendered_focal_target_missing",
                severity=FindingSeverity.ERROR,
                artifact_id="rendered_frames",
                message=f"Camera focal targets do not exist: {sorted(set(missing_targets))}.",
                repair_target="camera",
                repair_scope="object",
                object_ids=sorted(set(missing_targets)),
                patch_paths=["/camera/cues"],
            ))

    @staticmethod
    def _flatten(root: LaidOutNode) -> list[LaidOutNode]:
        result = [root]
        for child in root.children:
            result.extend(RenderedFrameQualityEvaluator._flatten(child))
        return result

    @staticmethod
    def _scene_sample_numbers(manifest: VideoManifest | None) -> list[int]:
        """Sample both transition midpoints and beat-final frames."""

        if manifest is None:
            return []
        numbers: list[int] = []
        for scene in manifest.scenes:
            midpoint = (scene.frame_start + scene.frame_end) // 2
            numbers.extend([midpoint, scene.frame_end])
        return list(dict.fromkeys(numbers))

    @staticmethod
    def _path(folder: Path, frames: FrameSequence, number: int) -> Path:
        try:
            name = frames.pattern % number
        except (TypeError, ValueError):
            name = f"frame_{number:06d}.png"
        return folder / name

    def _numbered_metrics(
        self,
        folder: Path,
        frames: FrameSequence,
        numbers: list[int],
    ) -> list[tuple[int, _PixelMetrics]]:
        return [
            (number, self._metrics(path))
            for number in numbers
            for path in [self._path(folder, frames, number)]
            if path.exists()
        ]

    @staticmethod
    def _frame_number(path: Path) -> int | None:
        digits = "".join(character for character in path.stem if character.isdigit())
        return int(digits) if digits else None

    @staticmethod
    def _metrics(path: Path) -> _PixelMetrics:
        """Return foreground ratios, bounds, and unsafe-edge ink."""

        with Image.open(path) as source:
            image = source.convert("RGB")
        background_color = image.getpixel((0, 0))
        background = Image.new("RGB", image.size, background_color)
        difference = ImageChops.difference(image, background).convert("L")
        mask = difference.point(lambda value: 255 if value > 18 else 0)
        histogram = mask.histogram()
        ink_pixels = sum(histogram[1:])
        pixel_count = max(1, image.width * image.height)
        bounds = mask.getbbox()
        occupancy = 0.0
        if bounds is not None:
            occupancy = (
                (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                / pixel_count
            )
        edge = max(1, round(min(image.size) * 0.015))
        edge_pixels = (
            sum(mask.crop((0, 0, image.width, edge)).histogram()[1:])
            + sum(mask.crop((0, image.height - edge, image.width, image.height)).histogram()[1:])
            + sum(mask.crop((0, edge, edge, image.height - edge)).histogram()[1:])
            + sum(mask.crop((image.width - edge, edge, image.width, image.height - edge)).histogram()[1:])
        )
        edge_area = max(1, 2 * image.width * edge + 2 * (image.height - 2 * edge) * edge)
        return _PixelMetrics(
            ink_ratio=ink_pixels / pixel_count,
            occupancy=occupancy,
            bounds=bounds,
            width=image.width,
            height=image.height,
            edge_ink_ratio=edge_pixels / edge_area,
        )

    @staticmethod
    def _substantive(metrics: _PixelMetrics) -> bool:
        """Distinguish real diagram content from stray antialiased pixels."""

        return metrics.ink_ratio >= 0.0015 and metrics.occupancy >= 0.02
