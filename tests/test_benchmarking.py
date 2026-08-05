"""Reproducibility and comparison tests for the benchmark harness."""

from pathlib import Path

from PIL import Image

from app.benchmarking import (
    BenchmarkConfig,
    BenchmarkRunner,
    OfflineCaseExecutor,
    compare_reports,
    load_benchmark_suite,
    perceptual_distance,
)


SUITE_PATH = Path("tests/benchmark/cases.v1.json")
GOLDEN_PATH = Path("tests/benchmark/golden/simple_card.ppm")


def test_versioned_suite_covers_thirty_diverse_cases() -> None:
    suite = load_benchmark_suite(SUITE_PATH)

    assert len(suite.cases) == 30
    assert len({case.case_id for case in suite.cases}) == 30
    assert {
        "process",
        "hierarchy",
        "comparison",
        "mechanism",
        "timeline",
        "math",
        "code",
        "biology",
        "finance",
        "unfamiliar",
    }.issubset({case.category for case in suite.cases})


def test_offline_tier_is_network_free_and_structurally_repeatable(
    tmp_path: Path,
) -> None:
    suite = load_benchmark_suite(SUITE_PATH)
    config = BenchmarkConfig(width=960, height=540, fps=2)

    first = BenchmarkRunner(OfflineCaseExecutor()).run(
        suite,
        config,
        tmp_path / "first",
        limit=3,
    )
    second = BenchmarkRunner(OfflineCaseExecutor()).run(
        suite,
        config,
        tmp_path / "second",
        limit=3,
    )

    assert first.reliability == 1
    assert second.reliability == 1
    assert (tmp_path / "first" / "report.json").exists()
    assert [case.metrics for case in first.cases] == [
        case.metrics for case in second.cases
    ]
    assert all(case.stage_timings_seconds for case in first.cases)


def test_report_comparison_identifies_case_and_metric_delta(tmp_path: Path) -> None:
    suite = load_benchmark_suite(SUITE_PATH)
    baseline = BenchmarkRunner(OfflineCaseExecutor()).run(
        suite,
        BenchmarkConfig(),
        tmp_path / "baseline",
        limit=1,
    )
    degraded_case = baseline.cases[0].model_copy(
        update={
            "metrics": baseline.cases[0].metrics.model_copy(
                update={"topic_specificity": 0.0}
            )
        }
    )
    current = baseline.model_copy(update={"cases": [degraded_case]})

    comparison = compare_reports(baseline, current)

    specificity = next(
        item for item in comparison.metric_deltas
        if item.metric == "topic_specificity"
    )
    assert specificity.case_id == baseline.cases[0].case_id
    assert specificity.delta <= 0


def test_golden_frame_uses_perceptual_threshold(tmp_path: Path) -> None:
    with Image.open(GOLDEN_PATH) as source:
        same_pixels = tmp_path / "same.png"
        source.convert("RGB").resize((80, 80)).save(same_pixels)
        changed = source.convert("RGB").resize((80, 80))
        changed.paste((255, 0, 0), (20, 20, 60, 60))
        changed_path = tmp_path / "changed.png"
        changed.save(changed_path)

    assert perceptual_distance(GOLDEN_PATH, same_pixels) < 0.02
    assert perceptual_distance(GOLDEN_PATH, changed_path) > 0.02
