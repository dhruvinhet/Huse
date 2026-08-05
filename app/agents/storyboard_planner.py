"""Specialized AI visual-storyboard planner."""

from app.agents.base import StructuredGeminiAgent
from app.domain.lesson import LessonPlan
from app.domain.pedagogy import PedagogyPlan
from app.domain.storyboard import Storyboard
from app.domain.strategy import TemplateMatch, VisualStrategy
from app.domain.quality import QualityReport
from app.domain.visual_intent import VisualIntent, VisualIntentPatch
from app.planning.visual_intent_compiler import VisualIntentCompiler
from app.services.gemini_client import GeminiClient


class GeminiStoryboardPlanner:
    """Plan high-level visual intent and compile it deterministically."""

    def __init__(self, client: GeminiClient, max_attempts: int = 3) -> None:
        """Initialize the structured storyboard-planning agent."""

        effective_attempts = 1 if client.PROVIDER == "nvidia" else max_attempts
        self._agent = StructuredGeminiAgent(
            client,
            VisualIntent,
            "Visual Intent Planner",
            effective_attempts,
        )
        self._repair_agent = StructuredGeminiAgent(
            client,
            VisualIntentPatch,
            "Visual Intent Beat Repair",
            effective_attempts,
        )
        self._compiler = VisualIntentCompiler()
        self._last_intent: VisualIntent | None = None
        self._last_pedagogy: PedagogyPlan | None = None

    @property
    def last_intent(self) -> VisualIntent | None:
        """Expose the last validated model artifact for debug recording."""

        return self._last_intent

    def plan(
        self,
        lesson: LessonPlan,
        strategies: list[VisualStrategy],
        templates: list[TemplateMatch],
    ) -> Storyboard:
        """Create progressive visual operations without coordinates."""

        return self.plan_with_pedagogy(
            lesson,
            strategies,
            templates,
            None,
        )

    def plan_with_pedagogy(
        self,
        lesson: LessonPlan,
        strategies: list[VisualStrategy],
        templates: list[TemplateMatch],
        pedagogy: PedagogyPlan | None,
    ) -> Storyboard:
        """Create a storyboard that obeys the routed teaching grammar."""

        self._last_intent = None
        self._last_pedagogy = pedagogy
        instructions = (
            "Describe only high-level visual teaching intent. For each shot choose "
            "the lesson concept IDs, their semantic relation when relevant, the "
            "single focal object, factual evidence that must be visible, the "
            "meaningful transformation, and one renderer operator. Do not invent "
            "object IDs, scene objects, connectors, hierarchy, coordinates, "
            "layout constraints, operations, camera instructions, or narration. "
            "Those implementation details are owned by a deterministic compiler. "
            "Use only concept IDs from the lesson and cover every important "
            "concept. Evidence must be a concrete fact, value, state, comparison, "
            "or observation rather than decorative prose."
        )
        if pedagogy is not None:
            instructions += (
                f" The selected teaching mode is {pedagogy.mode.value}. "
                "Use each routed shot_id exactly once in the supplied order and "
                "make its focal object, evidence, and transformation satisfy the "
                "corresponding visual obligation."
            )
        payload = {
            "lesson": lesson.model_dump(mode="json"),
            "strategy_hints": [
                {
                    "concept_ids": item.concept_ids,
                    "teaching_strategy": item.teaching_strategy,
                    "animation_hints": item.animation_hints,
                }
                for item in strategies
            ],
            "pedagogy_plan": (
                pedagogy.model_dump(mode="json")
                if pedagogy is not None
                else None
            ),
        }
        del templates  # Reviewed matches are compiled before this model path.
        intent = self._agent.generate(
            instructions,
            payload,
            validator=lambda item: self._compiler.validate_intent(
                item,
                lesson,
                pedagogy,
            ),
        )
        self._last_intent = intent
        return self._compiler.compile(intent, lesson, pedagogy)

    def repair(
        self,
        lesson: LessonPlan,
        previous: Storyboard,
        report: QualityReport,
        strategies: list[VisualStrategy],
        templates: list[TemplateMatch],
    ) -> Storyboard:
        """Repair only defects identified by a structured quality report."""

        instructions = (
            "Repair the high-level visual intent using every actionable quality "
            "finding. Change only concepts, relation, focal object, evidence, "
            "transformation, or renderer operator. Never emit scene objects, IDs, "
            "coordinates, hierarchy, connectors, or visual operations. Return the "
            "complete compact visual intent, not a patch."
        )
        payload = {
            "lesson": lesson.model_dump(mode="json"),
            "previous_visual_intent": (
                self._last_intent.model_dump(mode="json")
                if self._last_intent is not None
                else self._distill_storyboard(previous)
            ),
            "quality_report": report.model_dump(mode="json"),
            "strategy_hints": [item.teaching_strategy for item in strategies],
        }
        del templates
        intent = self._agent.generate(
            instructions,
            payload,
            validator=lambda item: self._compiler.validate_intent(
                item,
                lesson,
                self._last_pedagogy,
            ),
        )
        self._last_intent = intent
        return self._compiler.compile(intent, lesson, self._last_pedagogy)

    def repair_beats(
        self,
        lesson: LessonPlan,
        previous: Storyboard,
        report: QualityReport,
        strategies: list[VisualStrategy],
        templates: list[TemplateMatch],
        beat_ids: list[str],
    ) -> Storyboard:
        """Request and merge only the intent shots implicated by findings."""

        if self._last_intent is None:
            return self.repair(lesson, previous, report, strategies, templates)
        index_by_beat = {
            beat.beat_id: index for index, beat in enumerate(previous.beats)
        }
        target_indexes = sorted({
            index_by_beat[beat_id]
            for beat_id in beat_ids
            if beat_id in index_by_beat and index_by_beat[beat_id] < len(self._last_intent.shots)
        })
        if not target_indexes:
            return self.repair(lesson, previous, report, strategies, templates)
        target_shots = [self._last_intent.shots[index] for index in target_indexes]
        target_ids = {shot.shot_id for shot in target_shots}
        del templates

        def validate_patch(patch: VisualIntentPatch) -> None:
            actual = {shot.shot_id for shot in patch.shots}
            if actual != target_ids:
                raise ValueError(
                    "visual intent patch must contain exactly the requested shot IDs"
                )
            merged = self._merge_patch(self._last_intent, patch)
            self._compiler.validate_intent(merged, lesson, self._last_pedagogy)

        patch = self._repair_agent.generate(
            (
                "Repair only the supplied high-level shots using the actionable "
                "quality findings. Return exactly the requested shot IDs. Keep "
                "all other lesson shots untouched. Change only concepts, relation, "
                "focal object, evidence, transformation, or renderer operator; "
                "never emit low-level scene objects or operations."
            ),
            {
                "lesson": lesson.model_dump(mode="json"),
                "target_shots": [shot.model_dump(mode="json") for shot in target_shots],
                "quality_findings": [
                    finding.model_dump(mode="json")
                    for finding in report.findings
                    if finding.beat_id in beat_ids
                ],
                "strategy_hints": [item.teaching_strategy for item in strategies],
            },
            validator=validate_patch,
        )
        intent = self._merge_patch(self._last_intent, patch)
        self._last_intent = intent
        return self._compiler.compile(intent, lesson, self._last_pedagogy)

    @staticmethod
    def _merge_patch(
        intent: VisualIntent,
        patch: VisualIntentPatch,
    ) -> VisualIntent:
        replacements = {shot.shot_id: shot for shot in patch.shots}
        return intent.model_copy(update={
            "shots": [
                replacements.get(shot.shot_id, shot) for shot in intent.shots
            ]
        })

    @staticmethod
    def _distill_storyboard(previous: Storyboard) -> dict[str, object]:
        """Remove low-level object trees before asking for intent repair."""

        return {
            "lesson_focus": previous.title,
            "shots": [
                {
                    "shot_id": beat.beat_id,
                    "concept_ids": beat.concept_ids,
                    "teaching_intent": beat.teaching_intent,
                    "phrase_intent": beat.phrase_intent,
                    "purpose": beat.purpose,
                }
                for beat in previous.beats
            ],
        }
