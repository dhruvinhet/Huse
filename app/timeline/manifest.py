"""Build continuous frame ranges from phrase-aligned storyboard beats."""

from collections import defaultdict

from app.domain.narration import AlignedAudio
from app.domain.storyboard import Storyboard
from app.models.video_manifest import SceneManifest, VideoManifest


class PhraseManifestBuilder:
    """Convert beat phrase windows into the existing validated manifest."""

    def build(
        self,
        storyboard: Storyboard,
        alignment: AlignedAudio,
        fps: int = 30,
    ) -> VideoManifest:
        """Create one manifest scene per storyboard beat."""

        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for phrase in alignment.phrases:
            intervals[phrase.beat_id].append(
                (phrase.audio_start, phrase.audio_end)
            )
        scenes: list[SceneManifest] = []
        next_frame = 1
        expected_start = 0.0
        for scene_number, beat in enumerate(storyboard.beats, start=1):
            beat_intervals = intervals.get(beat.beat_id)
            if not beat_intervals:
                raise ValueError(f"beat has no aligned phrase: {beat.beat_id}")
            start = min(item[0] for item in beat_intervals)
            end = max(item[1] for item in beat_intervals)
            if abs(start - expected_start) > 1e-6:
                raise ValueError("storyboard beat phrase windows must be continuous")
            frame_end = round(end * fps)
            if frame_end < next_frame:
                raise ValueError("storyboard beat is too short for configured FPS")
            scenes.append(
                SceneManifest(
                    scene_number=scene_number,
                    frame_start=next_frame,
                    frame_end=frame_end,
                    audio_start=start,
                    audio_end=end,
                    duration=end - start,
                )
            )
            next_frame = frame_end + 1
            expected_start = end
        if abs(expected_start - alignment.duration) > 1e-6:
            raise ValueError("storyboard beats must cover the complete narration")
        return VideoManifest(
            fps=fps,
            total_frames=next_frame - 1,
            duration=alignment.duration,
            scenes=scenes,
        )
