"""Deterministic pixel checks that run even without a vision model."""

from pathlib import Path
from statistics import median

from PIL import Image, ImageChops

from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityFinding,
    QualityReport,
)
from app.domain.rendering import FrameSequence


class RenderedFrameQualityEvaluator:
    """Reject blank openings and materially underused rendered canvases."""

    def evaluate(self, frames: FrameSequence) -> QualityReport:
        """Inspect bounded deterministic samples from the encoded PNG sequence."""

        findings: list[QualityFinding] = []
        folder = Path(frames.folder)
        opening_limit = min(frames.total_frames, max(1, round(frames.fps * 2)))
        stride = max(1, frames.fps // 4)
        opening_paths = [
            folder / f"frame_{number:06d}.png"
            for number in range(1, opening_limit + 1, stride)
        ]
        opening_metrics = [
            self._metrics(path) for path in opening_paths if path.exists()
        ]
        first_visible = next(
            (
                index * stride / frames.fps
                for index, metrics in enumerate(opening_metrics)
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
            ))

        sample_metrics = [
            self._metrics(Path(path))
            for path in frames.sample_paths
            if Path(path).exists()
        ]
        substantive = [item for item in sample_metrics if self._substantive(item)]
        if sample_metrics and len(substantive) < max(1, len(sample_metrics) * 2 // 3):
            findings.append(QualityFinding(
                code="rendered_samples_blank",
                severity=FindingSeverity.ERROR,
                artifact_id="rendered_frames",
                message="Too many representative frames contain no substantive visual content.",
                repair_target="renderer",
            ))
        occupancies = [item[1] for item in substantive]
        if occupancies and median(occupancies) < 0.08:
            findings.append(QualityFinding(
                code="rendered_canvas_underused",
                severity=FindingSeverity.ERROR,
                artifact_id="rendered_frames",
                message="Representative teaching visuals occupy less than 8% of the canvas.",
                repair_target="layout",
            ))

        score = max(0.0, 1.0 - 0.34 * len(findings))
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

    @staticmethod
    def _metrics(path: Path) -> tuple[float, float]:
        """Return ink ratio and ink bounding-box occupancy for one frame."""

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
        return ink_pixels / pixel_count, occupancy

    @staticmethod
    def _substantive(metrics: tuple[float, float]) -> bool:
        """Distinguish real diagram content from a few stray antialiased pixels."""

        ink_ratio, occupancy = metrics
        return ink_ratio >= 0.0015 and occupancy >= 0.02
