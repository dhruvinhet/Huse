"""Measure the bounded-memory raw-frame encoder used by production rendering."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw


def _frame(width: int, height: int, variant: int) -> bytes:
    image = Image.new("RGB", (width, height), "#fbfaf4")
    draw = ImageDraw.Draw(image)
    offset = 24 + (variant % 4) * max(8, width // 20)
    draw.rounded_rectangle(
        (offset, height // 3, min(width - 24, offset + width // 3), 2 * height // 3),
        radius=max(4, width // 100),
        fill="#ffffff",
        outline="#2563eb",
        width=max(2, width // 320),
    )
    return image.tobytes()


def run_once(
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    fps: int,
    duration: int,
) -> dict[str, float | int]:
    """Encode a two-minute semantic-keyframe workload through a raw RGB pipe."""

    total_frames = fps * duration
    variants = [_frame(width, height, index) for index in range(4)]
    output.unlink(missing_ok=True)
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-frames:v", str(total_frames),
        str(output),
    ]
    tracemalloc.start()
    started = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    keyframes = 0
    for frame_number in range(total_frames):
        if frame_number % fps == 0:
            keyframes += 1
        process.stdin.write(variants[(frame_number // fps) % len(variants)])
    process.stdin.close()
    stderr = process.stderr.read() if process.stderr else b""
    return_code = process.wait()
    wall_time = time.perf_counter() - started
    _current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if return_code != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))
    return {
        "wall_time_seconds": wall_time,
        "peak_python_memory_bytes": peak_memory,
        "total_frames": total_frames,
        "keyframe_count": keyframes,
        "cache_hit_rate": (total_frames - keyframes) / total_frames,
        "encoded_frames_per_second": total_frames / wall_time,
        "output_bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark/t08-performance"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("FFmpeg is required; install it or add it to PATH.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_once(
            ffmpeg,
            args.output_dir / f"stream_run_{index + 1}.mp4",
            args.width,
            args.height,
            args.fps,
            args.duration,
        )
        for index in range(args.runs)
    ]
    times = [float(item["wall_time_seconds"]) for item in results]
    sorted_times = sorted(times)
    p90_index = max(0, min(len(sorted_times) - 1, int(0.9 * len(sorted_times) + 0.999) - 1))
    report = {
        "schema_version": "1.0",
        "profile": "renderer_raw_stream_two_minute",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "ffmpeg": ffmpeg,
        },
        "config": vars(args) | {"output_dir": args.output_dir.as_posix()},
        "runs": results,
        "median_seconds": statistics.median(times),
        "p90_seconds": sorted_times[p90_index],
        "targets": {"median_seconds": 180, "p90_seconds": 240},
        "targets_met": statistics.median(times) <= 180 and sorted_times[p90_index] <= 240,
        "scope": "Renderer/encoder replacement baseline; excludes provider, TTS, and asset latency.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
