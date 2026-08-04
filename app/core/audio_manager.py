"""Narration generation through the Edge-TTS service."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
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
    MAX_PARALLEL_SCENES = 4
    SCENE_TIMEOUT_SECONDS = 45.0

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
        cache_directory = self._working_path(
            configured_audio_dir / "cache"
        )
        working_path.parent.mkdir(parents=True, exist_ok=True)
        segments_directory.mkdir(parents=True, exist_ok=True)
        cache_directory.mkdir(parents=True, exist_ok=True)
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
        word_timings: list[AudioWordTiming] = []
        timing_sources: list[str] = []

        try:
            # TTS requests are independent. Keep the worker count bounded so a
            # long lesson does not flood Edge-TTS, then concatenate in scene
            # order so measured timings remain deterministic.
            executor = ThreadPoolExecutor(
                max_workers=min(self.MAX_PARALLEL_SCENES, len(narrated_scenes)),
                thread_name_prefix="edge-tts",
            )
            futures = []
            try:
                futures = [
                    executor.submit(
                        self._synthesize_scene,
                        scene.narration.strip(),
                        normalized_voice,
                        scene.scene_number,
                        segments_directory,
                        cache_directory,
                    )
                    for scene in narrated_scenes
                ]
                synthesized = [
                    future.result(timeout=self.SCENE_TIMEOUT_SECONDS)
                    for future in futures
                ]
            except FuturesTimeoutError as exc:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise AudioGenerationError(
                    "Edge-TTS scene synthesis timed out; retry the narration "
                    "after checking the TTS service."
                ) from exc
            except Exception:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

            ordered_segments: list[Path] = []
            for scene, result in zip(
                narrated_scenes,
                synthesized,
                strict=True,
            ):
                (
                    segment_path,
                    boundaries,
                    duration,
                    scene_sample_rate,
                    timing_source,
                ) = result
                ordered_segments.append(segment_path)
                timing_sources.append(timing_source)
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
                elapsed = end_time
                logger.info(
                    "Scene {} audio: {:.2f}-{:.2f}s (duration={:.2f}s, "
                    "word_boundaries={}).",
                    scene.scene_number,
                    start_time,
                    end_time,
                    duration,
                    len(boundaries),
                )
            self._combine_segments(ordered_segments, temporary_audio)
        except Exception as exc:
            self._remove_file(temporary_audio)
            logger.error("Narration audio generation failed: {}", exc)
            if isinstance(exc, AudioGenerationError):
                raise
            raise AudioGenerationError(
                f"Edge-TTS narration generation failed: {exc}"
            ) from exc
        finally:
            for segment_path in segments_directory.glob("scene_*.mp3"):
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
            word_timing_source=(
                "provider"
                if timing_sources and all(source == "provider" for source in timing_sources)
                else "estimated"
                if timing_sources and all(source == "estimated" for source in timing_sources)
                else "mixed"
            ),
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
                    elif _is_word_boundary_event(chunk.get("type")):
                        event = chunk.get("data")
                        event = event if isinstance(event, dict) else chunk
                        text = str(
                            event.get("text")
                            or event.get("word")
                            or event.get("token")
                            or ""
                        ).strip()
                        offset = _edge_ticks_to_seconds(
                            event.get("offset", event.get("start", 0))
                        )
                        duration = _edge_ticks_to_seconds(
                            event.get("duration", event.get("length", 0))
                        )
                        if text and offset >= 0:
                            boundaries.append((text, offset, duration))
        except (TypeError, AttributeError):
            wrote_audio = False
        if not wrote_audio:
            if segment_path.is_file():
                segment_path.unlink()
            communicator.save_sync(str(segment_path))
            boundaries.clear()
        return _normalize_boundaries(boundaries)

    def _synthesize_scene(
        self,
        text: str,
        voice: str,
        scene_number: int,
        segments_directory: Path,
        cache_directory: Path,
    ) -> tuple[Path, list[tuple[str, float, float]], float, int, str]:
        """Synthesize or load one scene without changing scene ordering."""

        cache_key = self._cache_key(text, voice)
        cached_path = cache_directory / f"{cache_key}.mp3"
        cached_metadata = cache_directory / f"{cache_key}.json"
        cached = self._load_cached_segment(cached_path, cached_metadata)
        if cached is not None:
            logger.info("Reusing cached narration segment for scene {}.", scene_number)
            return cached

        segment_path = segments_directory / f"scene_{scene_number:04d}.mp3"
        self._remove_file(segment_path)
        communicator = edge_tts.Communicate(text, voice)
        boundaries = self._save_with_word_boundaries(communicator, segment_path)
        timing_source = "provider" if boundaries else "estimated"
        duration, sample_rate = self._read_metadata(segment_path)
        if not boundaries:
            logger.warning(
                "Edge-TTS returned no word boundaries for scene {}; using "
                "deterministic local word timing fallback.",
                scene_number,
            )
            boundaries = self._estimate_word_boundaries(text, duration)

        shutil.copyfile(segment_path, cached_path)
        cached_metadata.write_text(
            json.dumps(
                {
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "boundaries": boundaries,
                    "timing_source": timing_source,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return segment_path, boundaries, duration, sample_rate, timing_source

    @staticmethod
    def _cache_key(text: str, voice: str) -> str:
        """Build a stable cache key from all TTS-affecting inputs."""

        source = f"{voice}\0{text}".encode("utf-8")
        return hashlib.sha256(source).hexdigest()

    @staticmethod
    def _load_cached_segment(
        audio_path: Path,
        metadata_path: Path,
    ) -> tuple[Path, list[tuple[str, float, float]], float, int, str] | None:
        """Load a complete cache entry, ignoring partial/corrupt entries."""

        if not audio_path.is_file() or not metadata_path.is_file():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            duration = float(payload["duration"])
            sample_rate = int(payload["sample_rate"])
            boundaries = _normalize_boundaries(payload["boundaries"])
            if duration <= 0 or sample_rate <= 0 or not boundaries:
                return None
            source = str(payload.get("timing_source", "estimated"))
            if source not in {"provider", "estimated"}:
                source = "estimated"
            return audio_path, boundaries, duration, sample_rate, source
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None

    @staticmethod
    def _estimate_word_boundaries(
        text: str,
        duration: float,
    ) -> list[tuple[str, float, float]]:
        """Create low-confidence local timings when TTS emits no boundaries."""

        words = re.findall(r"[\w'-]+", text, flags=re.UNICODE)
        if not words or duration <= 0:
            return []
        weights = [max(1, len(word)) for word in words]
        total_weight = sum(weights)
        cursor = 0.0
        boundaries: list[tuple[str, float, float]] = []
        for word, weight in zip(words, weights, strict=True):
            end = duration if word == words[-1] else cursor + duration * weight / total_weight
            boundaries.append((word, cursor, end))
            cursor = end
        return boundaries

    @staticmethod
    def _combine_segments(
        segment_paths: list[Path],
        output_path: Path,
    ) -> None:
        """Remux real MP3 segments with FFmpeg, retaining a portable fallback."""

        if not segment_paths:
            return
        if not all(AudioManager._looks_like_mp3(path) for path in segment_paths):
            with output_path.open("wb") as destination:
                for path in segment_paths:
                    AudioManager._append_file(path, destination)
            return

        executable = getattr(settings, "FFMPEG_PATH", None)
        executable = str(executable) if executable else shutil.which("ffmpeg")
        if not executable or not Path(executable).is_file():
            with output_path.open("wb") as destination:
                for path in segment_paths:
                    AudioManager._append_file(path, destination)
            return

        concat_list = output_path.with_suffix(".concat.txt")
        try:
            concat_list.write_text(
                "".join(
                    f"file '{path.resolve().as_posix().replace("'", "'\\''")}'\n"
                    for path in segment_paths
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    executable,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    # Normalize once after concatenation so scene boundaries
                    # do not introduce loudness or prosody jumps.
                    "-af",
                    "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=async=1:first_pts=0",
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    "-c:a",
                    "libmp3lame",
                    "-q:a",
                    "3",
                    str(output_path),
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode == 0 and output_path.is_file() and output_path.stat().st_size:
                return
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            AudioManager._remove_file(concat_list)
            if output_path.is_file() and not output_path.stat().st_size:
                AudioManager._remove_file(output_path)

        logger.warning("FFmpeg audio remux failed; using deterministic byte concatenation.")
        with output_path.open("wb") as destination:
            for path in segment_paths:
                AudioManager._append_file(path, destination)

    @staticmethod
    def _looks_like_mp3(path: Path) -> bool:
        """Avoid invoking FFmpeg on test doubles or obviously invalid files."""

        try:
            header = path.read_bytes()[:4]
        except OSError:
            return False
        return header[:3] == b"ID3" or (
            len(header) >= 2
            and header[0] == 0xFF
            and (header[1] & 0xE0) == 0xE0
        )

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


def _is_word_boundary_event(event_type: object) -> bool:
    """Accept Edge-TTS boundary spellings across package versions."""

    normalized = re.sub(r"[^a-z]", "", str(event_type).casefold())
    return normalized in {"wordboundary", "wordboundaries", "word"}


def _edge_ticks_to_seconds(value: object) -> float:
    """Convert Edge-TTS 100-nanosecond ticks defensively."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric <= 0:
        return 0.0
    return numeric / 10_000_000


def _normalize_boundaries(
    boundaries: object,
) -> list[tuple[str, float, float]]:
    """Sort, clamp, and close boundary intervals using the next onset."""

    if not isinstance(boundaries, list):
        return []
    parsed: list[tuple[str, float, float]] = []
    for item in boundaries:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        text = str(item[0]).strip()
        try:
            start = float(item[1])
            end_or_duration = float(item[2])
        except (TypeError, ValueError):
            continue
        if text and start >= 0:
            parsed.append((text, start, end_or_duration))
    parsed.sort(key=lambda item: item[1])
    normalized: list[tuple[str, float, float]] = []
    for index, (text, start, end_or_duration) in enumerate(parsed):
        next_start = (
            parsed[index + 1][1]
            if index + 1 < len(parsed)
            else start + max(end_or_duration, 0.03)
        )
        # Stream events carry duration; cached fallback entries carry an end.
        end = (
            start + end_or_duration
            if end_or_duration <= max(start, 0.0)
            else end_or_duration
        )
        if end <= start:
            end = next_start
        if end > start:
            normalized.append((text, start, end))
    return normalized
