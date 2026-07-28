"""Deterministic phrase alignment from measured narration segments."""

from app.domain.narration import (
    AlignedAudio,
    NarrationPlan,
    PhraseTiming,
    WordTiming,
)
from app.models.audio import AudioMetadata


class ScenePhraseAligner:
    """Map one generated audio scene to each storyboard narration phrase."""

    def align(
        self,
        narration: NarrationPlan,
        audio: AudioMetadata,
    ) -> AlignedAudio:
        """Return exact timing when TTS generated one segment per phrase."""

        if len(narration.phrases) != len(audio.scenes):
            raise ValueError(
                "phrase alignment requires one measured audio scene per phrase"
            )
        timings = [
            PhraseTiming(
                phrase_id=phrase.phrase_id,
                beat_id=phrase.beat_id,
                audio_start=scene.start_time,
                audio_end=scene.end_time,
                confidence=1.0,
            )
            for phrase, scene in zip(
                narration.phrases,
                audio.scenes,
                strict=True,
            )
        ]
        phrase_by_scene = {
            scene.scene_number: phrase
            for phrase, scene in zip(
                narration.phrases,
                audio.scenes,
                strict=True,
            )
        }
        words = [
            WordTiming(
                phrase_id=phrase_by_scene[word.scene_number].phrase_id,
                beat_id=phrase_by_scene[word.scene_number].beat_id,
                text=word.text,
                audio_start=word.start_time,
                audio_end=word.end_time,
                confidence=1.0,
            )
            for word in audio.words
            if word.scene_number in phrase_by_scene
        ]
        return AlignedAudio(
            audio_path=audio.file_path,
            duration=audio.duration,
            sample_rate=audio.sample_rate,
            phrases=timings,
            words=words,
        )
