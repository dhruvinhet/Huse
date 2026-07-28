"""Configurable gates for educational visual quality."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Thresholds used by deterministic and AI-assisted evaluators."""

    minimum_concept_coverage: float = 0.90
    maximum_static_gap: float = 3.0
    maximum_repair_attempts: int = 2
    minimum_text_width: float = 72.0
    fail_on_placeholder: bool = True

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
