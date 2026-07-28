"""Models describing planned narration scenes and visual instructions."""

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString


class VisualInstruction(BaseModel):
    """Describe one visual element requested for a planned scene."""

    type: NonEmptyString
    content: NonEmptyString
    position: NonEmptyString
    animation: NonEmptyString


class Scene(BaseModel):
    """Represent one narrated scene in a generated script."""

    scene_number: int = Field(ge=1)
    title: NonEmptyString
    narration: NonEmptyString
    estimated_duration: float = Field(gt=0)
    visuals: list[VisualInstruction]
