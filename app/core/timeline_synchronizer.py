"""Synchronize animation scenes to narration-driven video timing."""

from math import isclose

from loguru import logger

from app.models.animation import AnimationTimeline
from app.models.audio import AudioMetadata
from app.models.video_manifest import SceneManifest, VideoManifest


class TimelineSynchronizer:
    """Convert local scene animation timings to continuous video timings."""

    def synchronize(
        self,
        timeline: AnimationTimeline,
        audio: AudioMetadata,
        fps: int = 30,
    ) -> VideoManifest:
        """Build a continuous manifest using measured narration intervals."""

        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        if not audio.scenes:
            raise ValueError("audio metadata must include per-scene timing")

        timeline_scene_numbers = {
            scene.scene_number
            for scene in timeline.scenes
        }
        audio_scene_numbers = {
            scene.scene_number
            for scene in audio.scenes
        }
        if timeline_scene_numbers != audio_scene_numbers:
            raise ValueError(
                "animation timeline and audio scenes must match"
            )

        manifests: list[SceneManifest] = []
        next_frame = 1

        for scene_audio in audio.scenes:
            frame_boundary = round(scene_audio.end_time * fps)
            if frame_boundary < next_frame:
                raise ValueError(
                    f"scene {scene_audio.scene_number} is too short for {fps} FPS"
                )

            frame_start = next_frame
            frame_end = frame_boundary
            manifest = SceneManifest(
                scene_number=scene_audio.scene_number,
                frame_start=frame_start,
                frame_end=frame_end,
                audio_start=scene_audio.start_time,
                audio_end=scene_audio.end_time,
                duration=scene_audio.duration,
            )
            manifests.append(manifest)

            logger.info(
                "Scene {} synchronized (audio={:.2f}-{:.2f}s, "
                "frames={}-{}).",
                manifest.scene_number,
                manifest.audio_start,
                manifest.audio_end,
                frame_start,
                frame_end,
            )
            next_frame = frame_end + 1

        total_frames = next_frame - 1
        video_duration = total_frames / fps
        if not isclose(video_duration, audio.duration, abs_tol=0.1):
            raise ValueError(
                "frame-aligned video duration differs from audio by more "
                "than 100 ms"
            )
        video_manifest = VideoManifest(
            fps=fps,
            total_frames=total_frames,
            duration=audio.duration,
            scenes=manifests,
        )
        logger.info(
            "Timeline synchronized (audio_duration={:.2f}s, "
            "video_duration={:.2f}s, total_frames={}, scenes={}).",
            video_manifest.duration,
            video_duration,
            video_manifest.total_frames,
            len(video_manifest.scenes),
        )
        return video_manifest
