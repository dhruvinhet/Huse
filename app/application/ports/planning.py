"""Ports for specialized AI planning responsibilities."""

from typing import Protocol

from app.domain.generation import AudienceProfile, GenerationRequest
from app.domain.lesson import ConceptGraph, LessonPlan
from app.domain.narration import NarrationPlan
from app.domain.storyboard import Storyboard
from app.domain.strategy import TemplateMatch, VisualStrategy


class LessonPlanner(Protocol):
    """Build a lesson and concept graph from a generation request."""

    def plan(self, request: GenerationRequest) -> LessonPlan:
        """Return a validated lesson plan."""


class StoryboardPlanner(Protocol):
    """Build semantic visual beats without emitting coordinates."""

    def plan(
        self,
        lesson: LessonPlan,
        strategies: list[VisualStrategy],
        templates: list[TemplateMatch],
    ) -> Storyboard:
        """Return a validated persistent storyboard."""


class NarrationWriter(Protocol):
    """Write phrase-addressable narration from an accepted storyboard."""

    def write(
        self,
        storyboard: Storyboard,
        audience: AudienceProfile,
    ) -> NarrationPlan:
        """Return narration tied to storyboard beat IDs."""
