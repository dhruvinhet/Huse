"""Ports for deterministic frames and final composition."""

from typing import Protocol

from app.domain.rendering import (
    CompositionJob,
    FrameSequence,
    RenderJob,
    VideoArtifact,
)


class RenderEngine(Protocol):
    """Render a validated V2 job without inferring semantics."""

    def render(self, job: RenderJob) -> FrameSequence:
        """Return a continuous frame sequence."""


class VideoCompositionEngine(Protocol):
    """Compose verified frames and audio into final media."""

    def compose(self, job: CompositionJob) -> VideoArtifact:
        """Return a verified video artifact."""
