"""Comparison of compatible benchmark reports."""

from app.benchmarking.models import (
    BenchmarkComparison,
    BenchmarkReport,
    MetricDelta,
)


def compare_reports(
    baseline: BenchmarkReport,
    current: BenchmarkReport,
) -> BenchmarkComparison:
    """Compare common numeric metrics by stable case ID."""

    if baseline.suite_id != current.suite_id:
        raise ValueError("benchmark reports use different suite IDs")
    baseline_cases = {case.case_id: case for case in baseline.cases}
    current_cases = {case.case_id: case for case in current.cases}
    common = sorted(set(baseline_cases).intersection(current_cases))
    deltas: list[MetricDelta] = []
    for case_id in common:
        old_metrics = baseline_cases[case_id].metrics.model_dump()
        new_metrics = current_cases[case_id].metrics.model_dump()
        for metric, baseline_value in old_metrics.items():
            current_value = new_metrics.get(metric)
            if isinstance(baseline_value, (int, float)) and isinstance(
                current_value,
                (int, float),
            ):
                deltas.append(MetricDelta(
                    case_id=case_id,
                    metric=metric,
                    baseline=float(baseline_value),
                    current=float(current_value),
                    delta=float(current_value) - float(baseline_value),
                ))
    return BenchmarkComparison(
        suite_id=current.suite_id,
        reliability_delta=current.reliability - baseline.reliability,
        metric_deltas=deltas,
        added_cases=sorted(set(current_cases) - set(baseline_cases)),
        missing_cases=sorted(set(baseline_cases) - set(current_cases)),
    )
