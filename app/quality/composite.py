"""Combine deterministic and AI-assisted quality evaluators."""

from app.application.ports.quality import QualityEvaluator
from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityReport,
)


class CompositeQualityEvaluator:
    """Aggregate evaluators while preserving the strictest decision."""

    def __init__(self, evaluators: list[QualityEvaluator]) -> None:
        """Require at least one ordered evaluator."""

        if not evaluators:
            raise ValueError("at least one quality evaluator is required")
        self._evaluators = evaluators

    def evaluate(
        self,
        artifact_id: str,
        artifact: object,
        context: dict[str, object],
    ) -> QualityReport:
        """Merge scores, findings, and the most restrictive decision."""

        reports = [
            evaluator.evaluate(artifact_id, artifact, context)
            for evaluator in self._evaluators
        ]
        score_values: dict[str, list[float]] = {}
        for report in reports:
            for key, value in report.scores.items():
                score_values.setdefault(key, []).append(value)
        scores = {
            key: sum(values) / len(values)
            for key, values in score_values.items()
        }
        findings = [finding for report in reports for finding in report.findings]
        if any(report.decision is EvaluationDecision.FAIL for report in reports):
            decision = EvaluationDecision.FAIL
        elif any(report.decision is EvaluationDecision.REPAIR for report in reports):
            decision = EvaluationDecision.REPAIR
        else:
            decision = EvaluationDecision.PASS
        # A fatal finding always dominates an inconsistent provider decision.
        if any(item.severity is FindingSeverity.FATAL for item in findings):
            decision = EvaluationDecision.FAIL
        return QualityReport(
            overall_score=sum(report.overall_score for report in reports)
            / len(reports),
            scores=scores,
            findings=findings,
            decision=decision,
        )
