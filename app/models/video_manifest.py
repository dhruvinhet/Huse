"""Validated models for synchronized video scene and frame metadata."""

from math import isclose
from typing import Self

from pydantic import AliasChoices, Field, model_validator

from app.models.base import BaseModel


class SceneManifest(BaseModel):
    """Describe one scene's inclusive frames and narration interval."""

    scene_number: int = Field(ge=1)
    frame_start: int = Field(ge=1)
    frame_end: int = Field(ge=1)
    audio_start: float = Field(
        ge=0,
        validation_alias=AliasChoices("audio_start", "start_time"),
    )
    audio_end: float = Field(
        gt=0,
        validation_alias=AliasChoices("audio_end", "end_time"),
    )
    duration: float = Field(gt=0)

    @property
    def start_time(self) -> float:
        """Provide the former public name for audio_start."""

        return self.audio_start

    @property
    def end_time(self) -> float:
        """Provide the former public name for audio_end."""

        return self.audio_end

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """Require positive, forward-moving frame and time ranges."""

        if self.frame_end < self.frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start")
        if self.audio_end <= self.audio_start:
            raise ValueError("audio_end must be greater than audio_start")
        if not isclose(
            self.duration,
            self.audio_end - self.audio_start,
            abs_tol=1e-6,
        ):
            raise ValueError("duration must match the scene audio interval")
        return self


class VideoManifest(BaseModel):
    """Describe synchronized timing and frames for the complete video."""

    fps: int = Field(gt=0)
    total_frames: int = Field(ge=0)
    duration: float = Field(ge=0)
    scenes: list[SceneManifest]

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        """Require ordered, continuous, non-overlapping scene ranges."""

        if not self.scenes:
            if self.total_frames != 0 or self.duration != 0:
                raise ValueError(
                    "an empty manifest must have zero frames and duration"
                )
            return self

        expected_frame = 1
        expected_time = 0.0
        previous_scene_number = 0

        for scene in self.scenes:
            if scene.scene_number <= previous_scene_number:
                raise ValueError("scenes must be ordered by unique scene number")
            if scene.frame_start != expected_frame:
                raise ValueError("scene frame ranges must be continuous")
            if not isclose(
                scene.audio_start,
                expected_time,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "scene time ranges must be continuous and non-overlapping"
                )

            previous_scene_number = scene.scene_number
            expected_frame = scene.frame_end + 1
            expected_time = scene.audio_end

        if self.total_frames != expected_frame - 1:
            raise ValueError("total_frames does not match scene frame ranges")
        if not isclose(self.duration, expected_time, abs_tol=1e-9):
            raise ValueError("duration does not match scene time ranges")
        return self
