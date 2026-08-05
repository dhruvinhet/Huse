"""Phrase-addressable narration and measured audio alignment."""

from math import isclose
from typing import Literal, Self

from pydantic import Field, model_validator

from app.models.base import BaseModel, NonEmptyString


class NarrationPhrase(BaseModel):
    """Represent one spoken phrase tied to one storyboard beat."""

    phrase_id: NonEmptyString
    beat_id: NonEmptyString
    text: NonEmptyString
    delivery: Literal["normal", "emphasis", "pause_before", "pause_after"] = "normal"


class NarrationPlan(BaseModel):
    """Represent ordered narration written from a storyboard."""

    schema_version: Literal["2.0"] = "2.0"
    title: NonEmptyString = "Narration"
    phrases: list[NarrationPhrase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_phrases(self) -> Self:
        """Require stable unique phrase IDs."""

        phrase_ids = [phrase.phrase_id for phrase in self.phrases]
        if len(phrase_ids) != len(set(phrase_ids)):
            raise ValueError("narration phrase IDs must be unique")
        return self


class PhraseTiming(BaseModel):
    """Represent one measured phrase interval in the narration audio."""

    phrase_id: NonEmptyString
    beat_id: NonEmptyString
    audio_start: float = Field(ge=0)
    audio_end: float = Field(gt=0)
    confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Require forward-moving phrase timing."""

        if self.audio_end <= self.audio_start:
            raise ValueError("phrase audio_end must be greater than audio_start")
        return self


class WordTiming(BaseModel):
    """Represent one spoken word aligned to its narration phrase."""

    phrase_id: NonEmptyString
    beat_id: NonEmptyString
    text: NonEmptyString
    audio_start: float = Field(ge=0)
    audio_end: float = Field(gt=0)
    confidence: float = Field(default=1.0, ge=0, le=1)
    timing_source: Literal["provider", "estimated", "aligned"] = "provider"

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Require a forward-moving word interval."""

        if self.audio_end <= self.audio_start:
            raise ValueError("word audio_end must be greater than audio_start")
        return self


class AlignedAudio(BaseModel):
    """Describe continuous phrase-level timing for one audio track."""

    schema_version: Literal["2.0"] = "2.0"
    audio_path: NonEmptyString
    duration: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    phrases: list[PhraseTiming] = Field(min_length=1)
    words: list[WordTiming] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        """Require ordered, continuous, unique phrase intervals."""

        expected_start = 0.0
        phrase_ids: set[str] = set()
        for phrase in self.phrases:
            if phrase.phrase_id in phrase_ids:
                raise ValueError("aligned phrase IDs must be unique")
            if not isclose(phrase.audio_start, expected_start, abs_tol=1e-6):
                raise ValueError("phrase intervals must be continuous")
            phrase_ids.add(phrase.phrase_id)
            expected_start = phrase.audio_end
        if not isclose(expected_start, self.duration, abs_tol=1e-6):
            raise ValueError("aligned phrases must cover the audio duration")
        for word in self.words:
            if word.phrase_id not in phrase_ids:
                raise ValueError("aligned words must reference known phrases")
            if word.audio_end > self.duration + 1e-6:
                raise ValueError("aligned words must fit within the audio duration")
        return self
