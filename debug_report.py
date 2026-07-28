"""Inspect the latest pipeline debug bundle from the command line."""

import argparse
import json
from pathlib import Path
from typing import Any

from app.config.settings import PROJECT_ROOT, settings


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON artifact."""

    return json.loads(path.read_text(encoding="utf-8"))


def latest_run_directory() -> Path:
    """Resolve the latest recorded run from its stable pointer file."""

    debug_root = (
        settings.DEBUG_DIR
        if settings.DEBUG_DIR.is_absolute()
        else PROJECT_ROOT / settings.DEBUG_DIR
    )
    pointer_path = debug_root / "latest_run.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(
            "No debug run exists. Run python run_demo.py with "
            "DEBUG_ARTIFACTS=true first."
        )
    return Path(load_json(pointer_path)["path"])


def find_frame(trace_path: Path, frame_number: int) -> dict[str, Any]:
    """Stream the frame trace until the requested global frame is found."""

    with trace_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["frame_number"] == frame_number:
                return record
    raise ValueError(f"Frame {frame_number} is not present in the trace.")


def print_summary(run_dir: Path) -> None:
    """Print stage status plus narration-to-visual mapping by scene."""

    result = load_json(run_dir / "result.json")
    script = load_json(run_dir / "pipeline" / "script.json")
    audio = load_json(run_dir / "pipeline" / "audio.json")
    manifest = load_json(run_dir / "pipeline" / "video_manifest.json")
    audio_scenes = {
        scene["scene_number"]: scene
        for scene in audio["scenes"]
    }
    manifest_scenes = {
        scene["scene_number"]: scene
        for scene in manifest["scenes"]
    }

    print(f"Debug run: {run_dir}")
    print(f"Status: {result['status']}")
    print(f"Execution time: {result['execution_time_seconds']:.2f}s")
    print(f"Frames: {manifest['total_frames']} at {manifest['fps']} FPS")
    print()
    for scene in script["scenes"]:
        scene_number = scene["scene_number"]
        scene_audio = audio_scenes[scene_number]
        scene_manifest = manifest_scenes[scene_number]
        print(
            f"Scene {scene_number}: audio "
            f"{scene_audio['start_time']:.2f}-"
            f"{scene_audio['end_time']:.2f}s, frames "
            f"{scene_manifest['frame_start']}-"
            f"{scene_manifest['frame_end']}"
        )
        print(f"  Narration: {scene['narration']}")
        for visual in scene["visuals"]:
            print(
                "  Visual: "
                f"{visual['type']} | {visual['content']} | "
                f"{visual['position']} | {visual['animation']}"
            )
        print()


def main() -> None:
    """Print the latest run summary and optionally one frame trace record."""

    parser = argparse.ArgumentParser(
        description="Inspect whiteboard pipeline debug artifacts."
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Print the structured trace for one global frame number.",
    )
    args = parser.parse_args()
    run_dir = latest_run_directory()
    print_summary(run_dir)
    if args.frame is not None:
        record = find_frame(
            run_dir / "frames" / "frame_trace.jsonl",
            args.frame,
        )
        print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
