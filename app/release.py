"""Evaluate deterministic release gates and apply explicit expiring waivers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.benchmarking import compare_reports
from app.benchmarking.models import BenchmarkReport
from app.domain.release import ReleaseGateResult, ReleaseReport, ReleaseWaiver


_SEMANTIC_FAILURE_CODES = {
    "connector_relation_reversed",
    "connector_relation_ungrounded",
    "semantic_state_delta_missing",
    "factual_claim_unsupported",
    "factual_relation_reversed",
    "factual_relation_unsupported",
    "recap_objective_missing",
}


class ReleaseGateEvaluator:
    """Turn benchmark evidence into a deterministic release decision."""

    def evaluate(
        self,
        benchmark: BenchmarkReport,
        *,
        performance: dict[str, object] | None = None,
        baseline: BenchmarkReport | None = None,
        waivers: list[ReleaseWaiver] | None = None,
        artifact_paths: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> ReleaseReport:
        current_time = now or datetime.now(timezone.utc)
        waiver_by_gate = {item.gate_id: item for item in waivers or []}
        completed = [case for case in benchmark.cases if case.status == "completed"]

        def values(name: str) -> list[float]:
            return [
                float(value)
                for case in completed
                if (value := getattr(case.metrics, name)) is not None
            ]

        gates: list[ReleaseGateResult] = []

        def gate(
            gate_id: str,
            measured: float,
            required: float,
            passes: bool,
            message: str,
        ) -> None:
            waiver = waiver_by_gate.get(gate_id)
            valid_waiver = (
                waiver is not None
                and waiver.expires_at.astimezone(timezone.utc) > current_time
            )
            gates.append(ReleaseGateResult(
                gate_id=gate_id,
                status="passed" if passes else "waived" if valid_waiver else "failed",
                measured_value=measured,
                required_value=required,
                message=message,
                waiver=waiver if not passes and valid_waiver else None,
            ))

        gate(
            "reliability",
            benchmark.reliability,
            0.95,
            benchmark.reliability >= 0.95,
            "At least 95% of benchmark cases must complete.",
        )
        for gate_id, metric, threshold in (
            ("action_coverage", "action_coverage", 0.90),
            ("topic_specificity", "topic_specificity", 0.75),
            ("state_delta_coverage", "state_delta_coverage", 1.0),
            ("validated_word_timing", "audio_timing_coverage", 0.80),
        ):
            measured_values = values(metric)
            measured = min(measured_values, default=0.0)
            gate(
                gate_id,
                measured,
                threshold,
                bool(measured_values) and measured >= threshold,
                f"Every measured case must satisfy {metric} >= {threshold:.0%}.",
            )
        clipping = sum(case.metrics.clipping_violations for case in completed)
        readability = sum(case.metrics.readability_violations for case in completed)
        gate("safe_area_clipping", float(clipping), 0.0, clipping == 0,
             "No safe-area clipping violations are allowed.")
        gate("readability", float(readability), 0.0, readability == 0,
             "No deterministic readability violations are allowed.")
        semantic_failures = sum(
            code in _SEMANTIC_FAILURE_CODES
            for case in completed
            for code in case.quality_findings
        )
        gate("semantic_validity", float(semantic_failures), 0.0,
             semantic_failures == 0,
             "Required relations, claims, recaps, and state deltas must be valid.")

        if performance is not None:
            median = float(performance.get("median_seconds", float("inf")))
            p90 = float(performance.get("p90_seconds", float("inf")))
            gate("runtime_median", median, 180.0, median <= 180.0,
                 "Two-minute renderer median must be at most 180 seconds.")
            gate("runtime_p90", p90, 240.0, p90 <= 240.0,
                 "Two-minute renderer p90 must be at most 240 seconds.")
        else:
            gate("runtime_profile", 0.0, 1.0, False,
                 "A documented renderer performance report is required.")

        similarities = [
            case.novelty_similarity
            for case in benchmark.cases
            if case.novelty_similarity is not None
        ]
        gates.append(ReleaseGateResult(
            gate_id="novelty_similarity",
            status="reported",
            measured_value=(sum(similarities) / len(similarities) if similarities else None),
            message="Novelty remains report-only until repeatability is calibrated.",
        ))
        comparison = compare_reports(baseline, benchmark) if baseline else None
        passed = all(item.status in {"passed", "waived", "reported"} for item in gates)
        return ReleaseReport(
            created_at=current_time,
            suite_id=benchmark.suite_id,
            passed=passed,
            gates=gates,
            comparison=comparison,
            artifact_paths=artifact_paths or {},
        )
