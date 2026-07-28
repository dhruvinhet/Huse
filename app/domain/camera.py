"""Virtual camera contracts."""

from enum import Enum
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from app.models.base import BaseModel, NonEmptyString


class CameraOperation(str, Enum):
    """Supported virtual camera operations."""

    FIT = "fit"
    PAN = "pan"
    ZOOM = "zoom"
    FOCUS = "focus"
    TRACK = "track"
    HOLD = "hold"


class CameraCue(BaseModel):
    """Describe one timed camera operation."""

    cue_id: NonEmptyString
    beat_id: NonEmptyString
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0)
    operation: CameraOperation
    target_ids: list[NonEmptyString] = Field(default_factory=list)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class CameraPlan(BaseModel):
    """Represent the complete virtual-camera timeline."""

    schema_version: Literal["2.0"] = "2.0"
    duration: float = Field(gt=0)
    cues: list[CameraCue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cues(self) -> Self:
        """Require unique camera cues within the video duration."""

        cue_ids = [cue.cue_id for cue in self.cues]
        if len(cue_ids) != len(set(cue_ids)):
            raise ValueError("camera cue IDs must be unique")
        for cue in self.cues:
            if cue.start_time + cue.duration > self.duration + 1e-6:
                raise ValueError("camera cues must fit within the plan duration")
        return self
