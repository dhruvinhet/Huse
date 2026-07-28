"""Tests for run-scoped structured debug artifacts."""

import json
from pathlib import Path

from app.utils.debug_recorder import DebugRecorder


def test_debug_bundle_records_jsonl_and_result(tmp_path: Path) -> None:
    """An enabled recorder creates a unique, inspectable run bundle."""

    recorder = DebugRecorder(enabled=True, root_dir=tmp_path / "debug")
    run_dir = recorder.start_run("Test Topic")
    assert run_dir is not None

    recorder.write_text("llm/prompt.txt", "Exact prompt")
    recorder.write_json("tts/input.json", {"text": "Narration"})
    recorder.append_jsonl("frames/frame_trace.jsonl", {"frame": 1})
    recorder.append_jsonl("frames/frame_trace.jsonl", {"frame": 2})
    recorder.append_jsonl_many(
        "frames/frame_trace.jsonl",
        [{"frame": 3}, {"frame": 4}],
    )
    recorder.finish_run(
        status="complete",
        execution_time=1.25,
        stage_timings={"Generate Script": 0.5},
    )

    latest = json.loads(
        (tmp_path / "debug" / "latest_run.json").read_text(
            encoding="utf-8"
        )
    )
    assert Path(latest["path"]) == run_dir
    assert (run_dir / "llm" / "prompt.txt").read_text() == "Exact prompt"
    trace_lines = (
        run_dir / "frames" / "frame_trace.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["frame"] for line in trace_lines] == [1, 2, 3, 4]
    result = json.loads(
        (run_dir / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "complete"


def test_disabled_recorder_creates_no_artifacts(tmp_path: Path) -> None:
    """Debug recording can be disabled without changing pipeline callers."""

    debug_root = tmp_path / "debug"
    recorder = DebugRecorder(enabled=False, root_dir=debug_root)

    assert recorder.start_run("Ignored") is None
    recorder.write_text("ignored.txt", "ignored")

    assert not debug_root.exists()
