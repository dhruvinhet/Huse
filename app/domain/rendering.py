"""Renderer and composition boundary contracts."""

from typing import Literal

from pydantic import Field

from app.domain.camera import CameraPlan
from app.domain.assets import ResolvedAssetSet
from app.domain.layout import LayoutPlan
from app.domain.motion import MotionPlan
from app.domain.visual_document import VisualDocument
from app.models.base import BaseModel, NonEmptyString
from app.models.audio import AudioMetadata
from app.models.video_manifest import VideoManifest


class RenderJob(BaseModel):
    """Collect validated inputs required by a V2 renderer."""

    schema_version: Literal["2.0"] = "2.0"
    run_id: NonEmptyString
    document: VisualDocument
    assets: ResolvedAssetSet
    layout: LayoutPlan
    motion: MotionPlan
    camera: CameraPlan
    manifest: VideoManifest
    output_folder: NonEmptyString


class FrameSequence(BaseModel):
    """Describe a continuous rendered image sequence."""

    folder: NonEmptyString
    pattern: NonEmptyString = "frame_%06d.png"
    total_frames: int = Field(gt=0)
    fps: int = Field(gt=0)
    sample_paths: list[NonEmptyString] = Field(default_factory=list)


class CompositionJob(BaseModel):
    """Describe inputs required to compose final media."""

    frames: FrameSequence
    manifest: VideoManifest
    audio: AudioMetadata
    output_file: NonEmptyString


class VideoArtifact(BaseModel):
    """Describe one verified final video."""

    path: NonEmptyString
    duration: float = Field(gt=0)
    total_frames: int = Field(gt=0)
    fps: float = Field(gt=0)
    video_codec: NonEmptyString
    audio_codec: NonEmptyString
