"""Adapt phrase-addressable V2 narration to the existing AudioManager."""

from math import ceil

from app.core.audio_manager import AudioManager
from app.domain.narration import NarrationPlan
from app.models.audio import AudioMetadata
from app.models.scene import Scene
from app.models.script import Script


class StoryboardSpeechSynthesizer:
    """Synthesize one measured segment per narration phrase."""

    def __init__(self, audio_manager: AudioManager) -> None:
        """Store the working V1 audio implementation."""

        self._audio_manager = audio_manager

    def synthesize(self, narration: NarrationPlan, voice: str) -> AudioMetadata:
        """Convert phrases to V1 scenes and reuse Edge-TTS generation."""

        scenes = [
            Scene(
                scene_number=index,
                title=f"Phrase {index}",
                narration=phrase.text,
                estimated_duration=max(0.5, len(phrase.text.split()) / 2.5),
                visuals=[],
            )
            for index, phrase in enumerate(narration.phrases, start=1)
        ]
        total_duration = max(
            1,
            ceil(sum(scene.estimated_duration for scene in scenes)),
        )
        script = Script(
            title=narration.title,
            topic=narration.title,
            total_duration=total_duration,
            scenes=scenes,
        )
        return self._audio_manager.generate(script, voice=voice)
