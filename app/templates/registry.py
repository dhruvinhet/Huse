"""Deterministic registry and matcher for educational templates."""

from typing import Protocol

from app.domain.lesson import ConceptGraph
from app.domain.storyboard import VisualObjectSpec
from app.domain.strategy import TemplateMatch


class SemanticTemplate(Protocol):
    """Contract implemented by one parameterized visual template."""

    template_id: str
    keywords: frozenset[str]

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build a semantic object hierarchy without coordinates."""


class TemplateRegistry:
    """Register, match, and instantiate approved semantic templates."""

    def __init__(self, templates: list[SemanticTemplate] | None = None) -> None:
        """Register an optional initial template set."""

        self._templates: dict[str, SemanticTemplate] = {}
        for template in templates or []:
            self.register(template)

    def register(self, template: SemanticTemplate) -> None:
        """Register one uniquely named template."""

        if not template.template_id.strip():
            raise ValueError("template_id cannot be empty")
        if template.template_id in self._templates:
            raise ValueError(f"template already registered: {template.template_id}")
        if not template.keywords:
            raise ValueError("templates must declare matching keywords")
        self._templates[template.template_id] = template

    def match(self, graph: ConceptGraph) -> list[TemplateMatch]:
        """Rank templates using concept labels and visual affordances."""

        matches: list[TemplateMatch] = []
        for node in graph.nodes:
            terms = self._terms(node.label, node.visual_affordances)
            for template in self._templates.values():
                overlap = terms.intersection(template.keywords)
                if not overlap:
                    continue
                score = min(1.0, 0.55 + 0.15 * len(overlap))
                parameters = {
                    "object_id": f"{node.concept_id}_visual",
                    "label": node.label,
                }
                matches.append(
                    TemplateMatch(
                        template_id=template.template_id,
                        concept_ids=[node.concept_id],
                        parameters=parameters,
                        prototype=template.instantiate(parameters),
                        score=score,
                        reason=(
                            "Matched semantic terms: "
                            + ", ".join(sorted(overlap))
                        ),
                    )
                )
        return sorted(matches, key=lambda item: (-item.score, item.template_id))

    def instantiate(
        self,
        template_id: str,
        parameters: dict[str, object],
    ) -> VisualObjectSpec:
        """Instantiate a registered template."""

        try:
            template = self._templates[template_id]
        except KeyError as exc:
            raise KeyError(f"unknown template: {template_id}") from exc
        return template.instantiate(parameters)

    def template_ids(self) -> list[str]:
        """Return deterministic registered template IDs."""

        return sorted(self._templates)

    @staticmethod
    def _terms(label: str, affordances: list[str]) -> set[str]:
        """Normalize labels and affordances for conservative matching."""

        source = " ".join([label, *affordances]).lower()
        normalized = source.replace("-", " ").replace("_", " ")
        return set(normalized.split())
