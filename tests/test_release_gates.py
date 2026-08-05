"""Tests for deterministic release gates and explicit expiring waivers."""

from datetime import datetime, timedelta, timezone

from app.benchmarking.models import (
    BenchmarkCaseResult,
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkReport,
)
from app.domain.release import ReleaseWaiver
from app.release import ReleaseGateEvaluator


def _benchmark(*, action_coverage: float = 1.0) -> BenchmarkReport:
    now = datetime.now(timezone.utc)
    return BenchmarkReport(
        suite_id="release-suite",
        config=BenchmarkConfig(),
        started_at=now,
        finished_at=now,
        environment={"python": "3.12"},
        cases=[BenchmarkCaseResult(
            case_id="case",
            status="completed",
            wall_time_seconds=1,
            metrics=BenchmarkMetrics(
                topic_specificity=1,
                action_coverage=action_coverage,
                state_delta_coverage=1,
                audio_timing_coverage=1,
            ),
            novelty_similarity=0.9,
        )],
    )


def _performance() -> dict[str, object]:
    return {"median_seconds": 100.0, "p90_seconds": 150.0}


def test_release_passes_all_deterministic_gates_and_reports_novelty() -> None:
    report = ReleaseGateEvaluator().evaluate(
        _benchmark(), performance=_performance()
    )

    assert report.passed is True
    assert all(
        gate.status == "passed"
        for gate in report.gates
        if gate.gate_id != "novelty_similarity"
    )
    novelty = next(
        gate for gate in report.gates if gate.gate_id == "novelty_similarity"
    )
    assert novelty.status == "reported"


def test_failed_gate_requires_owner_and_unexpired_waiver() -> None:
    now = datetime.now(timezone.utc)
    failed = ReleaseGateEvaluator().evaluate(
        _benchmark(action_coverage=0.5),
        performance=_performance(),
        now=now,
    )
    assert failed.passed is False

    waiver = ReleaseWaiver(
        gate_id="action_coverage",
        owner="quality-owner",
        reason="Known benchmark migration with tracked remediation.",
        expires_at=now + timedelta(days=7),
    )
    waived = ReleaseGateEvaluator().evaluate(
        _benchmark(action_coverage=0.5),
        performance=_performance(),
        waivers=[waiver],
        now=now,
    )
    assert waived.passed is True
    gate = next(
        item for item in waived.gates if item.gate_id == "action_coverage"
    )
    assert gate.status == "waived"
    assert gate.waiver.owner == "quality-owner"

    expired = waiver.model_copy(update={"expires_at": now - timedelta(seconds=1)})
    rejected = ReleaseGateEvaluator().evaluate(
        _benchmark(action_coverage=0.5),
        performance=_performance(),
        waivers=[expired],
        now=now,
    )
    assert rejected.passed is False


def test_missing_runtime_artifact_fails_release() -> None:
    report = ReleaseGateEvaluator().evaluate(_benchmark())

    assert report.passed is False
    runtime = next(
        gate for gate in report.gates if gate.gate_id == "runtime_profile"
    )
    assert runtime.status == "failed"
