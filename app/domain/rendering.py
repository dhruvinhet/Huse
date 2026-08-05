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
    keep_frames: bool = True


class FrameSequence(BaseModel):
    """Describe a continuous rendered image sequence."""

    folder: NonEmptyString
    pattern: NonEmptyString = "frame_%06d.png"
    total_frames: int = Field(gt=0)
    fps: int = Field(gt=0)
    sample_paths: list[NonEmptyString] = Field(default_factory=list)
    diagnostics: list[NonEmptyString] = Field(default_factory=list)
    video_stream_path: str | None = None
    retained_frame_count: int = Field(default=0, ge=0)
    keyframe_count: int = Field(default=0, ge=0)
    cache_hit_count: int = Field(default=0, ge=0)
    layer_cache_hit_count: int = Field(default=0, ge=0)
    layer_cache_miss_count: int = Field(default=0, ge=0)
    peak_memory_bytes: int = Field(default=0, ge=0)
    encoded_frames_per_second: float | None = Field(default=None, ge=0)


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
