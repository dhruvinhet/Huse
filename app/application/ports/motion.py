"""Ports for semantic animation and virtual-camera planning."""

from typing import Protocol

from app.domain.camera import CameraPlan
from app.domain.layout import LayoutPlan
from app.domain.motion import MotionPlan
from app.domain.narration import AlignedAudio
from app.domain.storyboard import Storyboard


class AnimationPlanner(Protocol):
    """Schedule semantic animation strategies against speech timing."""

    def plan(
        self,
        storyboard: Storyboard,
        layout: LayoutPlan,
        alignment: AlignedAudio,
    ) -> MotionPlan:
        """Return renderer-independent motion events."""


class VirtualCameraPlanner(Protocol):
    """Plan deterministic fit, focus, pan, zoom, and tracking cues."""

    def plan(
        self,
        storyboard: Storyboard,
        layout: LayoutPlan,
        alignment: AlignedAudio,
    ) -> CameraPlan:
        """Return a virtual-camera timeline."""
