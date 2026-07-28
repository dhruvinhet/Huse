"""Model for a complete whiteboard video script."""

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString
from app.models.scene import Scene


class Script(BaseModel):
    """Represent the planned content and scenes for an entire video."""

    title: NonEmptyString
    topic: NonEmptyString
    total_duration: int = Field(gt=0)
    scenes: list[Scene]
