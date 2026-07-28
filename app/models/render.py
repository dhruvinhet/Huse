"""Models describing renderer-ready scenes and objects."""

from typing import Self

from pydantic import Field, model_validator

from app.models.base import BaseModel, NonEmptyString


class RenderableObject(BaseModel):
    """Represent one positioned object on a rendered scene canvas."""

    object_id: NonEmptyString
    type: NonEmptyString
    content: NonEmptyString
    x: int
    y: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    animation: NonEmptyString
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        """Require each object to end after it starts."""

        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        return self


class RenderScene(BaseModel):
    """Represent renderer-ready objects belonging to one scene."""

    scene_number: int = Field(ge=1)
    objects: list[RenderableObject]
