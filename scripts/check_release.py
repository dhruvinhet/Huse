"""Evaluate benchmark and performance artifacts as release gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if REPOSITORY_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT.as_posix())

from app.benchmarking import load_report  # noqa: E402
from app.domain.release import ReleaseWaiver  # noqa: E402
from app.release import ReleaseGateEvaluator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--waivers", type=Path, default=Path("release-waivers.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    performance = json.loads(args.performance.read_text(encoding="utf-8"))
    waiver_payload = (
        json.loads(args.waivers.read_text(encoding="utf-8"))
        if args.waivers.is_file()
        else {"waivers": []}
    )
    waivers = [
        ReleaseWaiver.model_validate(item)
        for item in waiver_payload.get("waivers", [])
    ]
    report = ReleaseGateEvaluator().evaluate(
        load_report(args.benchmark),
        performance=performance,
        baseline=load_report(args.baseline) if args.baseline else None,
        waivers=waivers,
        artifact_paths={
            "benchmark": args.benchmark.as_posix(),
            "performance": args.performance.as_posix(),
            **({"baseline": args.baseline.as_posix()} if args.baseline else {}),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    for gate in report.gates:
        print(f"{gate.status.upper():8} {gate.gate_id}: {gate.message}")
    print(f"Release decision: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
