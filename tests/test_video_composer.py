"""Tests for validated FFmpeg video composition."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core import video_composer
from app.core.video_composer import (
    FFmpegNotFoundError,
    VideoComposer,
    VideoCompositionError,
)
from app.models.audio import AudioMetadata
from app.models.video_manifest import SceneManifest, VideoManifest


def sample_manifest() -> VideoManifest:
    """Create a two-frame, 30 FPS video manifest."""

    return VideoManifest(
        fps=30,
        total_frames=2,
        duration=2 / 30,
        scenes=[
            SceneManifest(
                scene_number=1,
                frame_start=1,
                frame_end=2,
                start_time=0.0,
                end_time=2 / 30,
                duration=2 / 30,
            )
        ],
    )


def composition_inputs(tmp_path: Path) -> tuple[Path, AudioMetadata, Path]:
    """Create complete frame/audio paths for composer tests."""

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for frame_number in (1, 2):
        (frames_dir / f"frame_{frame_number:06d}.png").write_bytes(b"png")
    audio_path = tmp_path / "narration.mp3"
    audio_path.write_bytes(b"mp3")
    audio = AudioMetadata(
        file_path=str(audio_path),
        duration=2 / 30,
        sample_rate=24000,
        voice="en-US-AriaNeural",
    )
    return frames_dir, audio, tmp_path / "final.mp4"


def test_correct_ffmpeg_command_is_constructed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composition invokes FFmpeg with H.264, AAC, and 30 FPS inputs."""

    frames_dir, audio, output_path = composition_inputs(tmp_path)
    run_mock = MagicMock(
        side_effect=lambda command, **_kwargs: Path(command[-1]).write_bytes(
            b"mp4"
        )
    )
    monkeypatch.setattr(video_composer.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(video_composer.subprocess, "run", run_mock)
    stream_validation = MagicMock()
    monkeypatch.setattr(
        VideoComposer,
        "_validate_output_streams",
        stream_validation,
    )

    VideoComposer().compose(
        sample_manifest(),
        audio,
        str(frames_dir),
        str(output_path),
    )

    command = run_mock.call_args.args[0]
    assert command[0] == "ffmpeg"
    assert command[command.index("-framerate") + 1] == "30"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-preset") + 1] == "veryfast"
    assert command[command.index("-crf") + 1] == "18"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-map") + 1] == "0:v:0"
    second_map = command.index("-map", command.index("-map") + 1)
    assert command[second_map + 1] == "1:a:0"
    assert "-shortest" not in command
    assert "frame_%06d.png" in command[command.index("-i") + 1]
    assert output_path.is_file()
    stream_validation.assert_called_once()


def test_missing_frames_are_rejected(tmp_path: Path) -> None:
    """A missing frame sequence raises before FFmpeg execution."""

    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    audio = AudioMetadata(
        file_path=str(audio_path),
        duration=1,
        sample_rate=24000,
        voice="voice",
    )

    with pytest.raises(FileNotFoundError, match="Frames folder"):
        VideoComposer().compose(
            sample_manifest(),
            audio,
            str(tmp_path / "missing-frames"),
            str(tmp_path / "video.mp4"),
        )


def test_missing_audio_is_rejected(tmp_path: Path) -> None:
    """A missing narration file raises before checking FFmpeg."""

    frames_dir, _audio, output_path = composition_inputs(tmp_path)
    missing_audio = AudioMetadata(
        file_path=str(tmp_path / "missing.mp3"),
        duration=1,
        sample_rate=24000,
        voice="voice",
    )

    with pytest.raises(FileNotFoundError, match="Narration audio"):
        VideoComposer().compose(
            sample_manifest(),
            missing_audio,
            str(frames_dir),
            str(output_path),
        )


def test_incomplete_frame_sequence_is_rejected(tmp_path: Path) -> None:
    """A numbering gap is rejected before FFmpeg availability is checked."""

    frames_dir, audio, output_path = composition_inputs(tmp_path)
    (frames_dir / "frame_000002.png").unlink()

    with pytest.raises(ValueError, match="complete sequence"):
        VideoComposer().compose(
            sample_manifest(),
            audio,
            str(frames_dir),
            str(output_path),
        )


def test_missing_ffmpeg_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable FFmpeg executable raises a specific exception."""

    frames_dir, audio, output_path = composition_inputs(tmp_path)
    monkeypatch.setattr(video_composer.shutil, "which", lambda _name: None)
    monkeypatch.setattr(VideoComposer, "COMMON_FFMPEG_PATHS", ())

    with pytest.raises(FFmpegNotFoundError, match="PATH"):
        VideoComposer().compose(
            sample_manifest(),
            audio,
            str(frames_dir),
            str(output_path),
        )


def test_ffmpeg_execution_failure_is_translated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed FFmpeg process becomes VideoCompositionError."""

    frames_dir, audio, output_path = composition_inputs(tmp_path)
    monkeypatch.setattr(video_composer.shutil, "which", lambda _name: "ffmpeg")
    failure = subprocess.CalledProcessError(
        returncode=1,
        cmd=["ffmpeg"],
        stderr="encoding failed",
    )
    monkeypatch.setattr(
        video_composer.subprocess,
        "run",
        MagicMock(side_effect=failure),
    )

    with pytest.raises(VideoCompositionError, match="encoding failed"):
        VideoComposer().compose(
            sample_manifest(),
            audio,
            str(frames_dir),
            str(output_path),
        )


def test_audio_and_manifest_duration_must_match(
    tmp_path: Path,
) -> None:
    """Composition rejects synchronization drift above 100 milliseconds."""

    frames_dir, audio, output_path = composition_inputs(tmp_path)
    audio.duration = 1.0

    with pytest.raises(ValueError, match="100 ms"):
        VideoComposer().compose(
            sample_manifest(),
            audio,
            str(frames_dir),
            str(output_path),
        )
