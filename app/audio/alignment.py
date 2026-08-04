"""Deterministic phrase alignment from measured narration segments."""

import re

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
                confidence=(
                    1.0 if audio.word_timing_source in {"provider", "mixed"}
                    else 0.35
                ),
            )
            for word in audio.words
            if word.scene_number in phrase_by_scene
        ]
        if not words:
            words = [
                WordTiming(
                    phrase_id=phrase.phrase_id,
                    beat_id=phrase.beat_id,
                    text=text,
                    audio_start=scene.start_time + relative_start,
                    audio_end=scene.start_time + relative_end,
                    confidence=0.35,
                )
                for phrase, scene in zip(
                    narration.phrases,
                    audio.scenes,
                    strict=True,
                )
                for text, relative_start, relative_end in _fallback_word_timings(
                    phrase.text,
                    scene.duration,
                )
            ]
        return AlignedAudio(
            audio_path=audio.file_path,
            duration=audio.duration,
            sample_rate=audio.sample_rate,
            phrases=timings,
            words=words,
        )


def _fallback_word_timings(
    text: str,
    duration: float,
) -> list[tuple[str, float, float]]:
    """Create monotonic low-confidence word anchors without a speech model."""

    tokens = re.findall(r"[\w'-]+", text, flags=re.UNICODE)
    if not tokens or duration <= 0:
        return []
    weights = [max(1, len(token)) for token in tokens]
    total = sum(weights)
    cursor = 0.0
    result: list[tuple[str, float, float]] = []
    for index, (token, weight) in enumerate(zip(tokens, weights, strict=True)):
        end = (
            duration
            if index == len(tokens) - 1
            else cursor + duration * weight / total
        )
        result.append((token, cursor, end))
        cursor = end
    return result
