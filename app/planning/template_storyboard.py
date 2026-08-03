"""Compatibility planner that compiles template metadata through visual intent."""

from math import ceil

from app.domain.lesson import LessonPlan
from app.domain.storyboard import Storyboard
from app.domain.strategy import TemplateMatch, VisualStrategy
from app.domain.visual_intent import RendererOperator, ShotSpec, VisualIntent
from app.planning.visual_intent_compiler import VisualIntentCompiler


class TemplateStoryboardPlanner:
    """Convert matched template families to intent, never scene objects."""

    def plan(
        self,
        lesson: LessonPlan,
        strategies: list[VisualStrategy],
        templates: list[TemplateMatch],
    ) -> Storyboard:
        """Compile one operator-level shot per concept plus a summary."""

        del strategies
        operator = self._operator(templates)
        by_id = {
            node.concept_id: node
            for node in lesson.concept_graph.nodes
        }
        sequence = list(lesson.concept_graph.teaching_sequence)
        chunk = max(1, ceil(len(sequence) / min(5, len(sequence))))
        groups = [
            sequence[index:index + chunk]
            for index in range(0, len(sequence), chunk)
        ]
        shots = []
        for index, concept_ids in enumerate(groups):
            shots.append(
                ShotSpec(
                    shot_id=f"concept_{index + 1:03d}",
                    concept_ids=concept_ids,
                    relation=next(
                        (
                            edge.relation
                            for edge in lesson.concept_graph.edges
                            if {edge.source_id, edge.target_id}.intersection(
                                concept_ids
                            )
                        ),
                        None,
                    ),
                    focal_object=self._bounded_join(
                        [by_id[item].label for item in concept_ids],
                        " / ",
                        120,
                    ),
                    evidence=self._bounded_join(
                        [by_id[item].definition for item in concept_ids],
                        " ",
                        280,
                    ),
                    transformation=(
                        "Reveal the reviewed operator state for these concepts."
                    ),
                    renderer_operator=operator,
                )
            )
        shots.append(
            ShotSpec(
                shot_id="summary",
                concept_ids=list(lesson.concept_graph.teaching_sequence),
                relation=None,
                focal_object=lesson.title,
                evidence=self._bounded_join(
                    list(lesson.concept_graph.objectives),
                    "; ",
                    280,
                ),
                transformation="Show the complete reviewed operator state.",
                renderer_operator=operator,
            )
        )
        return VisualIntentCompiler().compile(
            VisualIntent(lesson_focus=lesson.title, shots=shots),
            lesson,
        )

    @staticmethod
    def _bounded_join(
        parts: list[str],
        separator: str,
        limit: int,
    ) -> str:
        """Join complete factual clauses within the intent size contract."""

        result: list[str] = []
        for part in parts:
            candidate = separator.join([*result, part])
            if len(candidate) > limit:
                break
            result.append(part)
        if result:
            return separator.join(result)
        return parts[0][:limit].rstrip() if parts else "Lesson concept"

    @staticmethod
    def _operator(templates: list[TemplateMatch]) -> RendererOperator:
        template_id = templates[0].template_id if templates else ""
        mappings = {
            "binary_search": RendererOperator.BINARY_SEARCH,
            "quick_sort": RendererOperator.SORTING,
            "merge_sort": RendererOperator.SORTING,
            "dfs": RendererOperator.GRAPH_TRAVERSAL,
            "bfs": RendererOperator.GRAPH_TRAVERSAL,
            "cnn": RendererOperator.NEURAL_NETWORK,
            "rnn": RendererOperator.NEURAL_NETWORK,
            "tcp": RendererOperator.PROTOCOL,
            "rest_api": RendererOperator.PROTOCOL,
            "microservices": RendererOperator.SYSTEM,
            "system_design": RendererOperator.SYSTEM,
            "database_index": RendererOperator.TREE_INDEX,
            "scheduling": RendererOperator.SCHEDULING,
            "memory": RendererOperator.MEMORY_MAP,
            "hash_map": RendererOperator.HASH_MAP,
            "blockchain": RendererOperator.BLOCKCHAIN,
            "comparison": RendererOperator.COMPARISON,
            "timeline": RendererOperator.TIMELINE,
            "cycle": RendererOperator.CYCLE,
            "cause_effect": RendererOperator.CAUSE_EFFECT,
            "flowchart": RendererOperator.FLOWCHART,
            "funnel": RendererOperator.FUNNEL,
            "venn": RendererOperator.VENN,
            "bar_chart": RendererOperator.BAR_CHART,
            "line_chart": RendererOperator.LINE_CHART,
            "equation": RendererOperator.EQUATION,
            "code_trace": RendererOperator.CODE_TRACE,
        }
        return next(
            (
                operator
                for prefix, operator in mappings.items()
                if prefix in template_id
            ),
            RendererOperator.PROCESS,
        )
