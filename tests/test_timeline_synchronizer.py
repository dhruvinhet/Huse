"""Tests for synchronizing animation timelines into video manifests."""

from app.core.timeline_synchronizer import TimelineSynchronizer
from app.models.animation import (
    AnimationInstruction,
    AnimationTimeline,
    AnimationType,
    SceneTimeline,
)
from app.models.audio import AudioMetadata, SceneAudio
from app.models.video_manifest import VideoManifest


def instruction(
    object_id: str,
    start_time: float,
    duration: float,
) -> AnimationInstruction:
    """Create one animation instruction for synchronization tests."""

    return AnimationInstruction(
        object_id=object_id,
        animation=AnimationType.FADE,
        start_time=start_time,
        duration=duration,
    )


def sample_timeline(reverse_scenes: bool = False) -> AnimationTimeline:
    """Create two scenes totaling 3.5 seconds at frame-aligned durations."""

    first_scene = SceneTimeline(
        scene_number=1,
        animations=[
            instruction("scene-1-a", 0.0, 1.0),
            instruction("scene-1-b", 1.0, 1.0),
        ],
    )
    second_scene = SceneTimeline(
        scene_number=2,
        animations=[instruction("scene-2-a", 0.0, 1.5)],
    )
    scenes = [second_scene, first_scene] if reverse_scenes else [
        first_scene,
        second_scene,
    ]
    return AnimationTimeline(scenes=scenes)


def sample_audio() -> AudioMetadata:
    """Create authoritative narration timing for two scenes."""

    return AudioMetadata(
        file_path="outputs/audio/narration.mp3",
        duration=3.5,
        sample_rate=24000,
        voice="en-US-AriaNeural",
        scenes=[
            SceneAudio(
                scene_number=1,
                duration=2.0,
                text="First scene",
                start_time=0.0,
                end_time=2.0,
            ),
            SceneAudio(
                scene_number=2,
                duration=1.5,
                text="Second scene",
                start_time=2.0,
                end_time=3.5,
            ),
        ],
    )


def test_frame_numbering_is_continuous() -> None:
    """Scene frame ranges are one-based, inclusive, and continuous."""

    manifest = TimelineSynchronizer().synchronize(
        sample_timeline(),
        sample_audio(),
        fps=10,
    )

    assert isinstance(manifest, VideoManifest)
    assert manifest.total_frames == 35
    assert (manifest.scenes[0].frame_start, manifest.scenes[0].frame_end) == (
        1,
        20,
    )
    assert (manifest.scenes[1].frame_start, manifest.scenes[1].frame_end) == (
        21,
        35,
    )


def test_duration_is_calculated_from_frame_aligned_scenes() -> None:
    """Manifest and scene durations reflect their synchronized frame counts."""

    manifest = TimelineSynchronizer().synchronize(
        sample_timeline(),
        sample_audio(),
        fps=10,
    )

    assert manifest.duration == 3.5
    assert (manifest.scenes[0].start_time, manifest.scenes[0].end_time) == (
        0.0,
        2.0,
    )
    assert (manifest.scenes[1].start_time, manifest.scenes[1].end_time) == (
        2.0,
        3.5,
    )
    assert manifest.scenes[0].duration == 2.0
    assert manifest.scenes[1].audio_start == 2.0


def test_scenes_are_ordered_by_scene_number() -> None:
    """Out-of-order input timelines become an ordered video manifest."""

    manifest = TimelineSynchronizer().synchronize(
        sample_timeline(reverse_scenes=True),
        sample_audio(),
        fps=10,
    )

    assert [scene.scene_number for scene in manifest.scenes] == [1, 2]


def test_scene_ranges_do_not_overlap() -> None:
    """Each scene starts immediately after its predecessor ends."""

    manifest = TimelineSynchronizer().synchronize(
        sample_timeline(),
        sample_audio(),
        fps=10,
    )
    first_scene, second_scene = manifest.scenes

    assert second_scene.frame_start == first_scene.frame_end + 1
    assert second_scene.start_time == first_scene.end_time


def test_audio_timing_overrides_animation_duration() -> None:
    """Manifest timing comes from narration rather than animation ends."""

    timeline = sample_timeline()
    timeline.scenes[0].animations[0].duration = 0.25

    manifest = TimelineSynchronizer().synchronize(
        timeline,
        sample_audio(),
        fps=10,
    )

    assert manifest.duration == 3.5
    assert manifest.total_frames == 35


def test_fractional_audio_boundaries_drive_global_frames() -> None:
    """Cumulative audio boundaries produce the documented frame ranges."""

    timeline = sample_timeline()
    audio = AudioMetadata(
        file_path="outputs/audio/narration.mp3",
        duration=14.94,
        sample_rate=24000,
        voice="en-US-AriaNeural",
        scenes=[
            SceneAudio(
                scene_number=1,
                duration=5.84,
                text="First narration",
                start_time=0,
                end_time=5.84,
            ),
            SceneAudio(
                scene_number=2,
                duration=9.10,
                text="Second narration",
                start_time=5.84,
                end_time=14.94,
            ),
        ],
    )

    manifest = TimelineSynchronizer().synchronize(timeline, audio, fps=30)

    assert (manifest.scenes[0].frame_start, manifest.scenes[0].frame_end) == (
        1,
        175,
    )
    assert (manifest.scenes[1].frame_start, manifest.scenes[1].frame_end) == (
        176,
        448,
    )
    assert manifest.total_frames == 448
    assert manifest.model_dump()["scenes"][0]["audio_end"] == 5.84
