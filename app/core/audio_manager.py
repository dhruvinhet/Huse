"""Narration generation through the Edge-TTS service."""

from pathlib import Path
from typing import BinaryIO

import edge_tts
from loguru import logger
from mutagen.mp3 import MP3

from app.config.settings import PROJECT_ROOT, settings
from app.models.audio import AudioMetadata, AudioWordTiming, SceneAudio
from app.models.script import Script
from app.utils.debug_recorder import DebugRecorder


class AudioGenerationError(RuntimeError):
    """Raised when narration audio cannot be generated or inspected."""


class AudioManager:
    """Generate one narration MP3 with measured per-scene timing."""

    OUTPUT_FILENAME = "narration.mp3"
    MANIFEST_FILENAME = "audio_manifest.json"

    def __init__(self, debug_recorder: DebugRecorder | None = None) -> None:
        """Store the optional recorder used for TTS diagnostics."""

        self._debug_recorder = debug_recorder

    def generate(
        self,
        script: Script,
        voice: str = "en-US-AriaNeural",
    ) -> AudioMetadata:
        """Generate narration audio and return validated MP3 metadata."""

        normalized_voice = voice.strip()
        if not normalized_voice:
            raise ValueError("voice must not be empty")

        narrated_scenes = [
            scene
            for scene in script.scenes
            if scene.narration.strip()
        ]
        if not narrated_scenes:
            raise ValueError("script must contain narration text")
        if self._debug_recorder is not None:
            self._debug_recorder.write_json(
                "tts/input.json",
                {
                    "voice": normalized_voice,
                    "scenes": [
                        {
                            "scene_number": scene.scene_number,
                            "text": scene.narration.strip(),
                        }
                        for scene in narrated_scenes
                    ],
                },
            )

        configured_audio_dir = Path(settings.OUTPUT_DIR) / "audio"
        configured_path = configured_audio_dir / self.OUTPUT_FILENAME
        configured_manifest = configured_audio_dir / self.MANIFEST_FILENAME
        working_path = self._working_path(configured_path)
        working_manifest = self._working_path(configured_manifest)
        segments_directory = self._working_path(
            Path(settings.TEMP_DIR) / "audio_segments"
        )
        working_path.parent.mkdir(parents=True, exist_ok=True)
        segments_directory.mkdir(parents=True, exist_ok=True)
        temporary_audio = segments_directory / "narration.tmp.mp3"
        self._remove_file(temporary_audio)
        logger.info(
            "Generating narration audio (scenes={}, voice={}).",
            len(narrated_scenes),
            normalized_voice,
        )

        scene_audio: list[SceneAudio] = []
        elapsed = 0.0
        sample_rate: int | None = None
        segment_paths: list[Path] = []
        word_timings: list[AudioWordTiming] = []

        try:
            with temporary_audio.open("wb") as combined_audio:
                for scene in narrated_scenes:
                    segment_path = (
                        segments_directory
                        / f"scene_{scene.scene_number:04d}.mp3"
                    )
                    segment_paths.append(segment_path)
                    self._remove_file(segment_path)
                    communicator = edge_tts.Communicate(
                        scene.narration.strip(),
                        normalized_voice,
                    )
                    boundaries = self._save_with_word_boundaries(
                        communicator,
                        segment_path,
                    )
                    duration, scene_sample_rate = self._read_metadata(
                        segment_path
                    )
                    if sample_rate is None:
                        sample_rate = scene_sample_rate
                    elif sample_rate != scene_sample_rate:
                        raise AudioGenerationError(
                            "Narration segments have inconsistent sample rates."
                        )

                    start_time = elapsed
                    end_time = start_time + duration
                    scene_audio.append(
                        SceneAudio(
                            scene_number=scene.scene_number,
                            duration=duration,
                            text=scene.narration.strip(),
                            start_time=start_time,
                            end_time=end_time,
                        )
                    )
                    for word, relative_start, relative_end in boundaries:
                        word_start = start_time + min(relative_start, duration)
                        word_end = start_time + min(relative_end, duration)
                        if word_end <= word_start:
                            word_end = min(end_time, word_start + 0.03)
                        if word.strip() and word_end > word_start:
                            word_timings.append(
                                AudioWordTiming(
                                    scene_number=scene.scene_number,
                                    text=word.strip(),
                                    start_time=word_start,
                                    end_time=word_end,
                                )
                            )
                    self._append_file(segment_path, combined_audio)
                    elapsed = end_time
                    logger.info(
                        "Scene {} audio: {:.2f}-{:.2f}s (duration={:.2f}s).",
                        scene.scene_number,
                        start_time,
                        end_time,
                        duration,
                    )
        except Exception as exc:
            self._remove_file(temporary_audio)
            logger.error("Narration audio generation failed: {}", exc)
            if isinstance(exc, AudioGenerationError):
                raise
            raise AudioGenerationError(
                f"Edge-TTS narration generation failed: {exc}"
            ) from exc
        finally:
            for segment_path in segment_paths:
                self._remove_file(segment_path)

        if not temporary_audio.is_file() or temporary_audio.stat().st_size == 0:
            raise AudioGenerationError(
                "Edge-TTS completed without creating the narration MP3."
            )
        temporary_audio.replace(working_path)
        if sample_rate is None:
            raise AudioGenerationError("Narration sample rate is unavailable.")

        metadata = AudioMetadata(
            file_path=configured_path.as_posix(),
            duration=elapsed,
            sample_rate=sample_rate,
            voice=normalized_voice,
            scenes=scene_audio,
            words=word_timings,
        )
        working_manifest.write_text(
            metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if self._debug_recorder is not None:
            self._debug_recorder.write_json(
                "tts/output_metadata.json",
                metadata.model_dump(mode="json"),
            )
        logger.info(
            "Narration audio generated (duration={:.2f}s, sample_rate={}, "
            "path={}, manifest={}).",
            metadata.duration,
            metadata.sample_rate,
            metadata.file_path,
            configured_manifest.as_posix(),
        )
        return metadata

    @staticmethod
    def _save_with_word_boundaries(
        communicator: edge_tts.Communicate,
        segment_path: Path,
    ) -> list[tuple[str, float, float]]:
        """Stream audio while retaining Edge-TTS word-boundary events."""

        boundaries: list[tuple[str, float, float]] = []
        wrote_audio = False
        try:
            with segment_path.open("wb") as destination:
                for chunk in communicator.stream_sync():
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("type") == "audio":
                        data = chunk.get("data")
                        if isinstance(data, bytes) and data:
                            destination.write(data)
                            wrote_audio = True
                    elif chunk.get("type") == "WordBoundary":
                        text = str(chunk.get("text", "")).strip()
                        offset = float(chunk.get("offset", 0)) / 10_000_000
                        duration = float(chunk.get("duration", 0)) / 10_000_000
                        if text and duration > 0:
                            boundaries.append(
                                (text, offset, offset + duration)
                            )
        except (TypeError, AttributeError):
            wrote_audio = False
        if not wrote_audio:
            if segment_path.is_file():
                segment_path.unlink()
            communicator.save_sync(str(segment_path))
            boundaries.clear()
        return boundaries

    @staticmethod
    def _read_metadata(segment_path: Path) -> tuple[float, int]:
        """Read and validate duration and sample rate for one MP3 segment."""

        if not segment_path.is_file():
            raise AudioGenerationError(
                f"Edge-TTS did not create segment {segment_path.name!r}."
            )
        try:
            mp3_info = MP3(segment_path).info
            duration = float(mp3_info.length)
            sample_rate = int(mp3_info.sample_rate)
        except Exception as exc:
            raise AudioGenerationError(
                f"Could not read narration segment metadata: {exc}"
            ) from exc
        if duration <= 0 or sample_rate <= 0:
            raise AudioGenerationError(
                "Narration segment metadata must be positive."
            )
        return duration, sample_rate

    @staticmethod
    def _append_file(source_path: Path, destination: BinaryIO) -> None:
        """Append one MP3 segment without loading it fully into memory."""

        with source_path.open("rb") as source:
            while chunk := source.read(64 * 1024):
                destination.write(chunk)

    @staticmethod
    def _remove_file(path: Path) -> None:
        """Remove one known temporary audio file if it exists."""

        if path.is_file():
            path.unlink()

    @staticmethod
    def _working_path(configured_path: Path) -> Path:
        """Resolve a configured relative audio path from the project root."""

        if configured_path.is_absolute():
            return configured_path
        return PROJECT_ROOT / configured_path
