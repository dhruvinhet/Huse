"""Versioned in-memory teaching strategies for the first V2 release."""

from app.domain.generation import AudienceProfile
from app.domain.lesson import ConceptGraph
from app.domain.strategy import VisualStrategy


class InMemoryVisualKnowledgeBase:
    """Return conservative strategies for recognized concept families."""

    _RULES: tuple[tuple[frozenset[str], str, str, tuple[str, ...]], ...] = (
        (
            frozenset({"array", "binary", "search", "sort"}),
            "array.v1",
            "Show indexed values and move a focus marker as the interval changes.",
            ("reveal_cell_by_cell", "move_pointer", "dim_discarded"),
        ),
        (
            frozenset({"transformer", "attention", "llm"}),
            "transformer_block.v1",
            "Build the model as nested processing components and highlight data flow.",
            ("reveal_components", "grow_connectors", "pulse_active_path"),
        ),
        (
            frozenset({"probability", "softmax", "distribution"}),
            "probability_distribution.v1",
            "Map scores to an explicitly labeled distribution and emphasize the selected outcome.",
            ("draw_axes", "grow_bars", "highlight_maximum"),
        ),
        (
            frozenset({"tree", "dfs", "bfs"}),
            "tree.v1",
            "Expand the hierarchy by depth and emphasize the traversal frontier.",
            ("expand_by_depth", "grow_edges", "highlight_frontier"),
        ),
        (
            frozenset({"api", "request", "pipeline", "flow"}),
            "pipeline.v1",
            "Reveal labeled stages and animate the request or data between them.",
            ("reveal_stages", "grow_connectors", "animate_flow"),
        ),
    )

    def strategies_for(
        self,
        graph: ConceptGraph,
        audience: AudienceProfile,
    ) -> list[VisualStrategy]:
        """Match known strategies and scale complexity to the audience."""

        complexity = {"beginner": 2, "intermediate": 4, "advanced": 6}[
            audience.level.value
        ]
        strategies: list[VisualStrategy] = []
        for node in graph.nodes:
            source = " ".join([node.label, *node.visual_affordances]).lower()
            terms = set(source.replace("-", " ").replace("_", " ").split())
            for keywords, template, teaching, animations in self._RULES:
                overlap = terms.intersection(keywords)
                if not overlap:
                    continue
                strategies.append(
                    VisualStrategy(
                        strategy_id=f"strategy_{node.concept_id}_{template}",
                        concept_ids=[node.concept_id],
                        preferred_template=template,
                        teaching_strategy=teaching,
                        animation_hints=list(animations),
                        visual_complexity=complexity,
                        audience_levels=[audience.level.value],
                        evidence_score=min(1.0, 0.6 + 0.1 * len(overlap)),
                    )
                )
        return sorted(
            strategies,
            key=lambda item: (-item.evidence_score, item.strategy_id),
        )
