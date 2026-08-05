"""Configurable gates for educational visual quality."""

from dataclasses import dataclass

from app.domain.assets import AssetSource, ResolvedAssetSet
from app.domain.quality import FindingSeverity, QualityReport
from app.domain.strategy import TemplateMatch


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Thresholds used by deterministic and AI-assisted evaluators."""

    minimum_concept_coverage: float = 0.90
    maximum_static_gap: float = 3.0
    maximum_repair_attempts: int = 2
    minimum_text_width: float = 72.0
    fail_on_placeholder: bool = True
    minimum_rendered_occupancy: float = 0.08
    maximum_rendered_occupancy: float = 0.96
    minimum_effective_font_px: float = 24.0
    minimum_camera_delta_px: float = 2.0

    def __post_init__(self) -> None:
        """Validate policy values at construction time."""

        if not 0 <= self.minimum_concept_coverage <= 1:
            raise ValueError("minimum_concept_coverage must be between zero and one")
        if self.maximum_static_gap <= 0:
            raise ValueError("maximum_static_gap must be positive")
        if self.maximum_repair_attempts < 0:
            raise ValueError("maximum_repair_attempts cannot be negative")
        if self.minimum_text_width <= 0:
            raise ValueError("minimum_text_width must be positive")
        if not 0 < self.minimum_rendered_occupancy < 1:
            raise ValueError("minimum_rendered_occupancy must be between zero and one")
        if not 0 < self.maximum_rendered_occupancy <= 1:
            raise ValueError("maximum_rendered_occupancy must be between zero and one")
        if self.maximum_rendered_occupancy <= self.minimum_rendered_occupancy:
            raise ValueError("maximum rendered occupancy must exceed the minimum")
        if self.minimum_effective_font_px <= 0:
            raise ValueError("minimum_effective_font_px must be positive")
        if self.minimum_camera_delta_px < 0:
            raise ValueError("minimum_camera_delta_px cannot be negative")


@dataclass(frozen=True, slots=True)
class QualityReviewPolicy:
    """Select costly semantic review only for explicitly risky artifacts."""

    minimum_template_confidence: float = 0.65
    minimum_deterministic_score: float = 0.92
    review_generated_assets: bool = True

    def storyboard_reasons(
        self,
        report: QualityReport,
        matches: list[TemplateMatch],
    ) -> list[str]:
        """Return stable reasons for an optional storyboard critic."""

        reasons: list[str] = []
        if not matches:
            reasons.append("no_reviewed_template_match")
        elif max(match.score for match in matches) < self.minimum_template_confidence:
            reasons.append("low_template_confidence")
        if report.overall_score < self.minimum_deterministic_score:
            reasons.append("deterministic_score_borderline")
        if any(
            finding.severity is FindingSeverity.WARNING
            for finding in report.findings
        ):
            reasons.append("deterministic_warning")
        return reasons

    def rendered_reasons(
        self,
        compiled: QualityReport,
        rendered: QualityReport,
        assets: ResolvedAssetSet,
        matches: list[TemplateMatch],
    ) -> list[str]:
        """Return stable reasons for optional multimodal frame review."""

        reasons = self.storyboard_reasons(compiled, matches)
        if rendered.overall_score < self.minimum_deterministic_score:
            reasons.append("rendered_score_borderline")
        if self.review_generated_assets and any(
            asset.source is AssetSource.GENERATED for asset in assets.assets
        ):
            reasons.append("generated_semantic_asset")
        return list(dict.fromkeys(reasons))
