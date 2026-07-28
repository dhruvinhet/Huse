"""Models describing declarative object animation timelines."""

from enum import Enum

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString


class AnimationType(str, Enum):
    """Supported animation categories for renderable objects."""

    WRITE = "write"
    DRAW = "draw"
    FADE = "fade"
    NONE = "none"


class AnimationInstruction(BaseModel):
    """Describe when and how one renderable object should animate."""

    object_id: NonEmptyString
    animation: AnimationType
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0)


class SceneTimeline(BaseModel):
    """Group ordered animation instructions for one scene."""

    scene_number: int = Field(ge=1)
    animations: list[AnimationInstruction]


class AnimationTimeline(BaseModel):
    """Represent the complete declarative animation timeline."""

    scenes: list[SceneTimeline]
