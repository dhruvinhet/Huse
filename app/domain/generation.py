"""Top-level request, result, and artifact contracts for pipeline V2."""

from datetime import datetime
from enum import Enum
from typing import Generic, Literal, TypeVar

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString
from app.domain.lesson import SourceReference


class PipelineVersion(str, Enum):
    """Supported pipeline implementations."""

    V1 = "v1"
    V2 = "v2"


class AudienceLevel(str, Enum):
    """Supported instructional difficulty levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class AudienceProfile(BaseModel):
    """Describe assumed knowledge and desired visual complexity."""

    level: AudienceLevel = AudienceLevel.INTERMEDIATE
    assumed_knowledge: list[NonEmptyString] = Field(default_factory=list)
    learning_goal: NonEmptyString
    max_visual_density: int | None = Field(default=None, ge=1)


class OutputProfile(BaseModel):
    """Describe deterministic media output requirements."""

    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    fps: int = Field(default=30, gt=0)
    format: Literal["mp4"] = "mp4"
    keep_frames: bool = False


class GenerationRequest(BaseModel):
    """Represent one fully specified V2 generation request."""

    schema_version: Literal["2.0"] = "2.0"
    run_id: NonEmptyString
    topic: NonEmptyString
    target_duration: float = Field(default=60.0, gt=0)
    audience: AudienceProfile
    style_id: NonEmptyString = "whiteboard.default"
    language: NonEmptyString = "en-US"
    voice: NonEmptyString = "en-US-AriaNeural"
    output: OutputProfile = Field(default_factory=OutputProfile)
    sources: list[SourceReference] = Field(default_factory=list, max_length=32)


class GenerationResult(BaseModel):
    """Describe the terminal output and quality of a V2 run."""

    schema_version: Literal["2.0"] = "2.0"
    run_id: NonEmptyString
    output_file: NonEmptyString
    duration: float = Field(gt=0)
    total_frames: int = Field(gt=0)
    quality_score: float = Field(ge=0, le=1)
    pipeline_version: PipelineVersion = PipelineVersion.V2


PayloadT = TypeVar("PayloadT")


class ArtifactEnvelope(BaseModel, Generic[PayloadT]):
    """Persist a versioned artifact with lineage and producer metadata."""

    artifact_id: NonEmptyString
    artifact_type: NonEmptyString
    schema_version: NonEmptyString
    producer: NonEmptyString
    producer_version: NonEmptyString
    created_at: datetime
    parent_artifact_ids: list[NonEmptyString] = Field(default_factory=list)
    content_hash: NonEmptyString
    payload: PayloadT
