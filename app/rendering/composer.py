"""Adapt V2 composition contracts to the verified existing composer."""

from pathlib import Path

from app.core.video_composer import VideoComposer
from app.domain.rendering import CompositionJob, VideoArtifact


class ExistingVideoComposerAdapter:
    """Reuse current FFmpeg command construction and stream verification."""

    def __init__(self, composer: VideoComposer) -> None:
        """Store the existing verified composer."""

        self._composer = composer

    def compose(self, job: CompositionJob) -> VideoArtifact:
        """Compose V2 frames and measured audio through the V1 implementation."""

        self._composer.compose(
            job.manifest,
            job.audio,
            job.frames.folder,
            job.output_file,
        )
        output_path = Path(job.output_file)
        if not output_path.is_file():
            raise RuntimeError("video composer did not create the requested output")
        return VideoArtifact(
            path=output_path.as_posix(),
            duration=job.manifest.duration,
            total_frames=job.manifest.total_frames,
            fps=float(job.manifest.fps),
            video_codec="h264",
            audio_codec="aac",
        )
