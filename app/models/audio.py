"""Metadata models for generated narration and scene-level timing."""

from math import isclose
from typing import Self

from pydantic import Field, model_validator

from app.models.base import BaseModel, NonEmptyString


class SceneAudio(BaseModel):
    """Describe one scene's narration text and exact audio interval."""

    scene_number: int = Field(ge=1)
    duration: float = Field(gt=0)
    text: NonEmptyString
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Require the duration to match a forward-moving time interval."""

        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if not isclose(
            self.duration,
            self.end_time - self.start_time,
            abs_tol=1e-6,
        ):
            raise ValueError("duration must match the scene audio interval")
        return self


class AudioWordTiming(BaseModel):
    """Represent one Edge-TTS word boundary on the combined audio track."""

    scene_number: int = Field(ge=1)
    text: NonEmptyString
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Require a forward-moving word interval."""

        if self.end_time <= self.start_time:
            raise ValueError("word end_time must be greater than start_time")
        return self


class AudioMetadata(BaseModel):
    """Describe narration audio and its continuous per-scene intervals."""

    file_path: NonEmptyString
    duration: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    voice: NonEmptyString
    scenes: list[SceneAudio] = Field(default_factory=list)
    words: list[AudioWordTiming] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scene_timing(self) -> Self:
        """Require ordered scene audio to cover the complete narration."""

        if not self.scenes:
            return self

        expected_start = 0.0
        previous_scene_number = 0
        for scene in self.scenes:
            if scene.scene_number <= previous_scene_number:
                raise ValueError("audio scenes must be ordered and unique")
            if not isclose(scene.start_time, expected_start, abs_tol=1e-6):
                raise ValueError("audio scene intervals must be continuous")
            expected_start = scene.end_time
            previous_scene_number = scene.scene_number

        if not isclose(self.duration, expected_start, abs_tol=1e-6):
            raise ValueError("audio duration must match the final scene end")
        previous_word_start = -1.0
        for word in self.words:
            if word.start_time < previous_word_start:
                raise ValueError("audio words must be ordered")
            if word.end_time > self.duration + 1e-6:
                raise ValueError("audio words must fit within the audio duration")
            previous_word_start = word.start_time
        return self
