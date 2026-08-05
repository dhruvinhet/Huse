"""Run the versioned Huse benchmark suite and optionally compare a baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if REPOSITORY_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT.as_posix())

from app.benchmarking import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkMode,
    BenchmarkRunner,
    OfflineCaseExecutor,
    ProviderCaseExecutor,
    compare_reports,
    load_benchmark_suite,
    load_report,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--suite",
        type=Path,
        default=REPOSITORY_ROOT / "tests" / "benchmark" / "cases.v1.json",
    )
    result.add_argument("--mode", choices=[item.value for item in BenchmarkMode], default="offline")
    result.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "outputs" / "benchmark")
    result.add_argument("--baseline", type=Path)
    result.add_argument("--limit", type=int)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--provider-id")
    result.add_argument("--model-id")
    result.add_argument("--width", type=int, default=1920)
    result.add_argument("--height", type=int, default=1080)
    result.add_argument("--fps", type=int, default=30)
    return result


def main() -> int:
    args = parser().parse_args()
    suite = load_benchmark_suite(args.suite)
    config = BenchmarkConfig(
        mode=BenchmarkMode(args.mode),
        suite_version=suite.schema_version,
        provider_id=args.provider_id,
        model_id=args.model_id,
        seed=args.seed,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    executor = (
        ProviderCaseExecutor()
        if config.mode is BenchmarkMode.PROVIDER
        else OfflineCaseExecutor()
    )
    report = BenchmarkRunner(executor).run(
        suite,
        config,
        args.output,
        limit=args.limit,
    )
    print(f"Report: {(args.output / 'report.json').resolve()}")
    print(f"Reliability: {report.reliability:.1%} ({len(report.cases)} cases)")
    if args.baseline:
        comparison = compare_reports(load_report(args.baseline), report)
        comparison_path = args.output / "comparison.json"
        comparison_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
        print(f"Comparison: {comparison_path.resolve()}")
        print(f"Reliability delta: {comparison.reliability_delta:+.1%}")
    return 0 if report.reliability == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
