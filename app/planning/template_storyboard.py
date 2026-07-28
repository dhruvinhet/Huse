"""Deterministic template-guided storyboard planner and fallback."""

from app.domain.lesson import LessonPlan
from app.domain.operations import OperationType, VisualOperation
from app.domain.storyboard import (
    AttentionCue,
    Storyboard,
    VisualBeat,
    VisualObjectSpec,
)
from app.domain.strategy import TemplateMatch, VisualStrategy


class TemplateStoryboardPlanner:
    """Create an evolving storyboard from matched reviewed templates."""

    def plan(
        self,
        lesson: LessonPlan,
        strategies: list[VisualStrategy],
        templates: list[TemplateMatch],
    ) -> Storyboard:
        """Build one progressively connected beat per concept."""

        del strategies
        matches_by_concept: dict[str, TemplateMatch] = {}
        for match in templates:
            for concept_id in match.concept_ids:
                current = matches_by_concept.get(concept_id)
                if current is None or match.score > current.score:
                    matches_by_concept[concept_id] = match
        nodes = {
            node.concept_id: node
            for node in lesson.concept_graph.nodes
        }
        beats: list[VisualBeat] = []
        previous_root_id: str | None = None
        duration = max(2.0, 60.0 / len(lesson.concept_graph.teaching_sequence))
        for index, concept_id in enumerate(
            lesson.concept_graph.teaching_sequence,
            start=1,
        ):
            node = nodes[concept_id]
            match = matches_by_concept.get(concept_id)
            root = (
                match.prototype.model_copy(deep=True)
                if match is not None and match.prototype is not None
                else self._fallback_object(node.concept_id, node.label, node.definition)
            )
            root.concept_ids = sorted(set([*root.concept_ids, concept_id]))
            operations = [
                VisualOperation(
                    operation_id=f"create_{concept_id}",
                    operation=OperationType.CREATE,
                    target_ids=[root.object_id],
                    arguments={"objects": [root.model_dump(mode="json")]},
                )
            ]
            if previous_root_id is not None:
                connector = VisualObjectSpec(
                    object_id=f"connector_{index - 1}_{index}",
                    kind="connector",
                    semantic_role="teaching_progression",
                    concept_ids=[concept_id],
                    content={
                        "source_id": previous_root_id,
                        "target_id": root.object_id,
                        "label": "next",
                    },
                    style_token="process.active",
                    accessibility_label=(
                        f"Teaching progression from the prior concept to {node.label}"
                    ),
                )
                operations.append(
                    VisualOperation(
                        operation_id=f"connect_{index - 1}_{index}",
                        operation=OperationType.CREATE,
                        target_ids=[connector.object_id],
                        arguments={
                            "objects": [connector.model_dump(mode="json")]
                        },
                    )
                )
                operations.append(
                    VisualOperation(
                        operation_id=f"dim_{index - 1}",
                        operation=OperationType.DIM,
                        target_ids=[previous_root_id],
                    )
                )
            beats.append(
                VisualBeat(
                    beat_id=f"beat_{index:03d}_{concept_id}",
                    section_id=f"section_{index:03d}",
                    concept_ids=[concept_id],
                    teaching_intent=f"Teach {node.label}",
                    phrase_intent=node.definition,
                    estimated_duration=duration,
                    operations=operations,
                    attention=[
                        AttentionCue(
                            cue="focus",
                            target_ids=[root.object_id],
                            intensity=min(1.0, 0.5 + node.importance / 2),
                        )
                    ],
                )
            )
            previous_root_id = root.object_id
        return Storyboard(
            document_id="template_storyboard",
            title=lesson.title,
            beats=beats,
            final_learning_summary=lesson.concept_graph.objectives,
        )

    @staticmethod
    def _fallback_object(
        concept_id: str,
        label: str,
        definition: str,
    ) -> VisualObjectSpec:
        """Create a labeled semantic component when no template matches."""

        return VisualObjectSpec(
            object_id=f"{concept_id}_visual",
            kind="component",
            semantic_role="concept_explanation",
            concept_ids=[concept_id],
            content={"label": label, "definition": definition},
            accessibility_label=f"{label}: {definition}",
        )
