"""Ports for visual knowledge and educational templates."""

from typing import Protocol

from app.domain.generation import AudienceProfile
from app.domain.lesson import ConceptGraph
from app.domain.storyboard import VisualObjectSpec
from app.domain.strategy import TemplateMatch, VisualStrategy
from app.domain.pedagogy import PedagogyPlan


class VisualKnowledgeBase(Protocol):
    """Look up teaching strategies for a concept graph."""

    def strategies_for(
        self,
        graph: ConceptGraph,
        audience: AudienceProfile,
    ) -> list[VisualStrategy]:
        """Return ranked strategies applicable to the audience."""


class TemplateLibrary(Protocol):
    """Match and instantiate reusable educational diagrams."""

    def match(
        self,
        graph: ConceptGraph,
        audience: AudienceProfile | None = None,
    ) -> list[TemplateMatch]:
        """Return ranked templates for graph concepts."""

    def instantiate(
        self,
        template_id: str,
        parameters: dict[str, object],
    ) -> VisualObjectSpec:
        """Instantiate a semantic hierarchy without coordinates."""

    def match_shots(
        self,
        graph: ConceptGraph,
        pedagogy: PedagogyPlan,
        concept_groups: list[list[str]],
        audience: AudienceProfile | None = None,
    ) -> list[TemplateMatch]:
        """Return matches evaluated against shot-local query graphs."""
