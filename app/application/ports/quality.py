"""Port for deterministic and multimodal quality evaluation."""

from typing import Protocol

from app.domain.quality import QualityReport


class QualityEvaluator(Protocol):
    """Evaluate a pipeline artifact with structured context."""

    def evaluate(
        self,
        artifact_id: str,
        artifact: object,
        context: dict[str, object],
    ) -> QualityReport:
        """Return scores, findings, and a pass/repair/fail decision."""
