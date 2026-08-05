"""FFmpeg composition of PNG frames and narration into an MP4 file."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - clean install behavior
    cv2 = None  # type: ignore[assignment]
from loguru import logger
try:
    from mutagen.mp4 import MP4
except ModuleNotFoundError:  # pragma: no cover - clean install behavior
    MP4 = None  # type: ignore[assignment,misc]

from app.config.settings import PROJECT_ROOT, settings
from app.models.audio import AudioMetadata
from app.models.video_manifest import VideoManifest
from app.utils.debug_recorder import DebugRecorder
from app.optional import missing_extra


class FFmpegNotFoundError(RuntimeError):
    """Raised when the FFmpeg executable is unavailable."""


class VideoCompositionError(RuntimeError):
    """Raised when FFmpeg cannot produce the requested MP4 file."""


class VideoComposer:
    """Compose synchronized PNG frames and narration through FFmpeg."""

    REQUIRED_FPS = 30
    DURATION_TOLERANCE_SECONDS = 0.1
    COMMON_FFMPEG_PATHS = (
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
    )

    def __init__(self, debug_recorder: DebugRecorder | None = None) -> None:
        """Store the optional recorder used for composition diagnostics."""

        if cv2 is None or MP4 is None:
            raise missing_extra(
                "Verified video composition",
                "requirements-audio.txt",
                "opencv-python and mutagen",
            )
        self._debug_recorder = debug_recorder

    def compose(
        self,
        manifest: VideoManifest,
        audio: AudioMetadata,
        frames_folder: str,
        output_file: str,
    ) -> None:
        """Validate inputs and compose an H.264/AAC MP4 with FFmpeg."""

        if manifest.fps != self.REQUIRED_FPS:
            raise ValueError(
                f"video manifest FPS must be {self.REQUIRED_FPS}"
            )

        configured_frames = Path(frames_folder)
        working_frames = self._working_path(configured_frames)
        self._validate_frames(working_frames, manifest.total_frames)

        configured_audio = Path(audio.file_path)
        working_audio = self._working_path(configured_audio)
        if not working_audio.is_file():
            raise FileNotFoundError(
                f"Narration audio was not found at {str(working_audio)!r}."
            )
        duration_difference = abs(manifest.duration - audio.duration)
        if duration_difference > self.DURATION_TOLERANCE_SECONDS:
            raise ValueError(
                "Video manifest and narration durations differ by more "
                "than 100 ms."
            )

        ffmpeg_executable = self._find_ffmpeg()
        if ffmpeg_executable is None:
            raise FFmpegNotFoundError(
                "FFmpeg is not installed or is not available on PATH."
            )

        configured_output = Path(output_file)
        working_output = self._working_path(configured_output)
        working_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = working_output.with_name(
            f"{working_output.stem}.tmp{working_output.suffix}"
        )
        self._remove_file(temporary_output)
        frames_pattern = working_frames / "frame_%06d.png"
        command = [
            ffmpeg_executable,
            "-y",
            "-framerate",
            str(self.REQUIRED_FPS),
            "-start_number",
            "1",
            "-i",
            str(frames_pattern),
            "-i",
            str(working_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-threads",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(self.REQUIRED_FPS),
            "-frames:v",
            str(manifest.total_frames),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(temporary_output),
        ]
        logger.info(
            "Composing final video (frames={}, audio_duration={:.2f}s, "
            "manifest_duration={:.2f}s, output={}).",
            manifest.total_frames,
            audio.duration,
            manifest.duration,
            configured_output.as_posix(),
        )
        logger.info("Using FFmpeg executable at {}.", ffmpeg_executable)
        if self._debug_recorder is not None:
            self._debug_recorder.write_json(
                "ffmpeg/command.json",
                {
                    "command": command,
                    "manifest_frames": manifest.total_frames,
                    "manifest_duration": manifest.duration,
                    "audio_duration": audio.duration,
                    "frames_folder": working_frames.as_posix(),
                    "temporary_output": temporary_output.as_posix(),
                    "final_output": working_output.as_posix(),
                },
            )

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            self._remove_file(temporary_output)
            details = exc.stderr.strip() if exc.stderr else str(exc)
            logger.error("FFmpeg composition failed: {}", details)
            raise VideoCompositionError(
                f"FFmpeg failed to compose the video: {details}"
            ) from exc
        except OSError as exc:
            self._remove_file(temporary_output)
            logger.error("FFmpeg could not be executed: {}", exc)
            raise VideoCompositionError(
                f"FFmpeg could not be executed: {exc}"
            ) from exc

        if not temporary_output.is_file():
            raise VideoCompositionError(
                "FFmpeg completed without creating the output MP4."
            )
        try:
            stream_info = self._validate_output_streams(
                temporary_output,
                manifest.total_frames,
                manifest.fps,
                audio.duration,
            )
            if self._debug_recorder is not None:
                self._debug_recorder.write_json(
                    "ffmpeg/verified_streams.json",
                    stream_info,
                )
            temporary_output.replace(working_output)
        except VideoCompositionError:
            self._remove_file(temporary_output)
            raise
        except OSError as exc:
            self._remove_file(temporary_output)
            raise VideoCompositionError(
                "Could not replace the final video file. Close any player "
                f"using it and try again: {exc}"
            ) from exc
        logger.info("Final video created at {}.", configured_output.as_posix())

    def compose_stream(
        self,
        manifest: VideoManifest,
        audio: AudioMetadata,
        video_stream: str,
        output_file: str,
    ) -> None:
        """Mux a renderer-produced H.264 MP4 with narration without PNG I/O."""

        if manifest.fps != self.REQUIRED_FPS:
            raise ValueError(f"video manifest FPS must be {self.REQUIRED_FPS}")
        working_stream = self._working_path(Path(video_stream))
        if not working_stream.is_file():
            raise FileNotFoundError(
                f"Rendered video stream was not found at {working_stream!s}."
            )
        working_audio = self._working_path(Path(audio.file_path))
        if not working_audio.is_file():
            raise FileNotFoundError(
                f"Narration audio was not found at {working_audio!s}."
            )
        if abs(manifest.duration - audio.duration) > self.DURATION_TOLERANCE_SECONDS:
            raise ValueError(
                "Video manifest and narration durations differ by more than 100 ms."
            )
        ffmpeg_executable = self._find_ffmpeg()
        if ffmpeg_executable is None:
            raise FFmpegNotFoundError(
                "FFmpeg is not installed or is not available on PATH."
            )
        working_output = self._working_path(Path(output_file))
        working_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = working_output.with_name(
            f"{working_output.stem}.tmp{working_output.suffix}"
        )
        self._remove_file(temporary_output)
        command = [
            ffmpeg_executable,
            "-y",
            "-i",
            str(working_stream),
            "-i",
            str(working_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-frames:v",
            str(manifest.total_frames),
            str(temporary_output),
        ]
        if self._debug_recorder is not None:
            self._debug_recorder.write_json(
                "ffmpeg/command.json",
                {
                    "command": command,
                    "input_mode": "streamed_mp4",
                    "video_stream": working_stream.as_posix(),
                    "audio": working_audio.as_posix(),
                },
            )
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            self._remove_file(temporary_output)
            details = exc.stderr.strip() if exc.stderr else str(exc)
            raise VideoCompositionError(
                f"FFmpeg failed to mux the streamed video: {details}"
            ) from exc
        except OSError as exc:
            self._remove_file(temporary_output)
            raise VideoCompositionError(
                f"FFmpeg could not be executed: {exc}"
            ) from exc
        if not temporary_output.is_file():
            raise VideoCompositionError(
                "FFmpeg completed without creating the output MP4."
            )
        try:
            stream_info = self._validate_output_streams(
                temporary_output,
                manifest.total_frames,
                manifest.fps,
                audio.duration,
            )
            if self._debug_recorder is not None:
                self._debug_recorder.write_json(
                    "ffmpeg/verified_streams.json", stream_info
                )
            temporary_output.replace(working_output)
        except (VideoCompositionError, OSError):
            self._remove_file(temporary_output)
            raise

    @classmethod
    def _find_ffmpeg(cls) -> str | None:
        """Find FFmpeg from configuration, PATH, or common local installs."""

        if settings.FFMPEG_PATH is not None:
            configured_path = cls._working_path(settings.FFMPEG_PATH)
            if not configured_path.is_file():
                raise FFmpegNotFoundError(
                    f"Configured FFmpeg was not found at {configured_path!s}."
                )
            return str(configured_path)

        path_executable = shutil.which("ffmpeg")
        if path_executable is not None:
            return path_executable
        for candidate in cls.COMMON_FFMPEG_PATHS:
            if candidate.is_file():
                return str(candidate)
        return None

    @classmethod
    def _validate_output_streams(
        cls,
        output_path: Path,
        expected_frames: int,
        expected_fps: int,
        expected_audio_duration: float,
    ) -> dict[str, float | int]:
        """Verify that the composed MP4 contains complete video and AAC audio."""

        try:
            audio_info = MP4(output_path).info
            audio_duration = float(audio_info.length)
            channels = int(audio_info.channels)
            sample_rate = int(audio_info.sample_rate)
        except Exception as exc:
            raise VideoCompositionError(
                f"Composed MP4 does not contain readable AAC audio: {exc}"
            ) from exc
        if channels < 1 or sample_rate <= 0:
            raise VideoCompositionError(
                "Composed MP4 contains invalid audio stream metadata."
            )
        if (
            abs(audio_duration - expected_audio_duration)
            > cls.DURATION_TOLERANCE_SECONDS
        ):
            raise VideoCompositionError(
                "Composed MP4 audio duration differs by more than 100 ms."
            )

        capture = cv2.VideoCapture(str(output_path))
        try:
            if not capture.isOpened():
                raise VideoCompositionError(
                    "Composed MP4 does not contain a readable video stream."
                )
            frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()

        if frame_count != expected_frames:
            raise VideoCompositionError(
                "Composed MP4 video frame count does not match the manifest."
            )
        if abs(video_fps - expected_fps) > 0.01:
            raise VideoCompositionError(
                "Composed MP4 video FPS does not match the manifest."
            )
        logger.info(
            "Verified MP4 streams (frames={}, fps={:.2f}, "
            "audio_duration={:.2f}s, channels={}, sample_rate={}).",
            frame_count,
            video_fps,
            audio_duration,
            channels,
            sample_rate,
        )
        return {
            "frames": frame_count,
            "fps": video_fps,
            "audio_duration_seconds": audio_duration,
            "audio_channels": channels,
            "audio_sample_rate": sample_rate,
        }

    @staticmethod
    def _validate_frames(frames_directory: Path, total_frames: int) -> None:
        """Require a complete, continuous frame sequence for the manifest."""

        if not frames_directory.is_dir():
            raise FileNotFoundError(
                f"Frames folder was not found at {str(frames_directory)!r}."
            )
        actual_names = sorted(
            path.name
            for path in frames_directory.glob("frame_*.png")
            if path.is_file()
        )
        expected_names = [
            f"frame_{frame_number:06d}.png"
            for frame_number in range(1, total_frames + 1)
        ]
        if actual_names != expected_names:
            raise ValueError(
                "Frames must be a complete sequence from frame_000001.png "
                f"through frame_{total_frames:06d}.png."
            )

    @staticmethod
    def _working_path(configured_path: Path) -> Path:
        """Resolve a configured relative input or output from project root."""

        if configured_path.is_absolute():
            return configured_path
        return PROJECT_ROOT / configured_path

    @staticmethod
    def _remove_file(path: Path) -> None:
        """Remove one known temporary composition output if it exists."""

        if path.is_file():
            path.unlink()
