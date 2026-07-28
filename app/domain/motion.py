"""Semantic motion-plan contracts."""

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from app.models.base import BaseModel, NonEmptyString


class MotionEvent(BaseModel):
    """Describe one renderer-independent animation event."""

    event_id: NonEmptyString
    beat_id: NonEmptyString
    operation_id: NonEmptyString
    object_ids: list[NonEmptyString] = Field(min_length=1)
    strategy: NonEmptyString
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0)
    easing: NonEmptyString = "ease_in_out"
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class MotionPlan(BaseModel):
    """Represent ordered semantic motion events for the full video."""

    schema_version: Literal["2.0"] = "2.0"
    duration: float = Field(gt=0)
    events: list[MotionEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_events(self) -> Self:
        """Require unique events contained within the video duration."""

        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("motion event IDs must be unique")
        for event in self.events:
            if event.start_time + event.duration > self.duration + 1e-6:
                raise ValueError("motion events must fit within the plan duration")
        return self
