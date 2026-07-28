"""Model describing the result returned by the processing pipeline."""

from app.models.base import BaseModel, NonEmptyString
from app.models.script import Script


class PipelineResult(BaseModel):
    """Represent a successful or failed pipeline execution result."""

    success: bool
    message: NonEmptyString
    script: Script | None = None
