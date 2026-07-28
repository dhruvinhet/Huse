"""Ports for narration synthesis and phrase alignment."""

from typing import Protocol

from app.domain.narration import AlignedAudio, NarrationPlan
from app.models.audio import AudioMetadata


class SpeechSynthesizer(Protocol):
    """Generate narration audio without changing narration content."""

    def synthesize(self, narration: NarrationPlan, voice: str) -> AudioMetadata:
        """Return measured audio metadata."""


class PhraseAligner(Protocol):
    """Align narration phrase IDs to measured audio intervals."""

    def align(
        self,
        narration: NarrationPlan,
        audio: AudioMetadata,
    ) -> AlignedAudio:
        """Return continuous phrase timing."""
