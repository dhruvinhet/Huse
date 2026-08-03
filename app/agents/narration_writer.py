"""Specialized storyboard-driven narration writer."""

from app.agents.base import StructuredGeminiAgent
from app.domain.generation import AudienceProfile
from app.domain.narration import NarrationPlan
from app.domain.pedagogy import PedagogyPlan
from app.domain.storyboard import Storyboard
from app.services.gemini_client import GeminiClient


class GeminiNarrationWriter:
    """Write speech that naturally references evolving storyboard visuals."""

    def __init__(self, client: GeminiClient, max_attempts: int = 3) -> None:
        """Initialize the structured narration-writing agent."""

        effective_attempts = 1 if client.PROVIDER == "nvidia" else max_attempts
        self._agent = StructuredGeminiAgent(
            client,
            NarrationPlan,
            "Narration Writer",
            effective_attempts,
        )

    def write(
        self,
        storyboard: Storyboard,
        audience: AudienceProfile,
    ) -> NarrationPlan:
        """Create concise phrases tied exactly to storyboard beat IDs."""

        return self.write_with_pedagogy(storyboard, audience, None)

    def write_with_pedagogy(
        self,
        storyboard: Storyboard,
        audience: AudienceProfile,
        pedagogy: PedagogyPlan | None,
        target_duration: float | None = None,
    ) -> NarrationPlan:
        """Write phrases that satisfy routed rhetorical obligations."""

        expected_beat_ids = [beat.beat_id for beat in storyboard.beats]
        total_word_target = (
            max(20, round(target_duration * 2.45))
            if target_duration is not None
            else None
        )

        instructions = (
            "Write natural educational narration from the accepted storyboard. "
            "Return exactly one phrase for each required storyboard beat ID listed "
            "below—no missing IDs, extra IDs, or duplicate beat IDs. Preserve the "
            "listed order. Required beat IDs: "
            f"{expected_beat_ids}. Reference what is appearing, "
            "changing, or being emphasized without sounding mechanical. Every "
            "phrase must use an existing beat_id. Phrase IDs must be unique. Keep "
            "vocabulary and explanation depth appropriate for the audience. Do not "
            "add concepts or visual actions that are absent from the storyboard. "
            "Use each beat's teaching_intent and phrase_intent as its specific "
            "rhetorical obligation instead of generating stock language from "
            "the purpose enum alone. Write listener-facing prose, never commands "
            "such as 'show', 'describe', or 'explain'. Keep each phrase to roughly "
            "20-45 words so no single static shot dominates the video."
        )
        if total_word_target is not None:
            instructions += (
                f" Aim for about {total_word_target} spoken words in total, "
                "distributed across beats according to their educational content."
            )
        if pedagogy is not None:
            instructions += (
                f" The teaching mode is {pedagogy.mode.value}. Ordered narration "
                f"obligations: {pedagogy.narration_obligations}."
            )
        payload = {
            "storyboard": storyboard.model_dump(mode="json"),
            "audience": audience.model_dump(mode="json"),
            "required_beat_ids": expected_beat_ids,
            "target_duration_seconds": target_duration,
            "target_spoken_words": total_word_target,
            "pedagogy_plan": (
                pedagogy.model_dump(mode="json")
                if pedagogy is not None
                else None
            ),
        }

        def validate_beat_coverage(plan: NarrationPlan) -> None:
            """Require the model to cover the accepted storyboard exactly."""

            actual = [phrase.beat_id for phrase in plan.phrases]
            if actual != expected_beat_ids:
                raise ValueError(
                    "narration phrases must contain exactly one phrase per beat "
                    f"in this order: {expected_beat_ids}; received: {actual}"
                )

        plan = self._agent.generate(
            instructions,
            payload,
            validator=validate_beat_coverage,
        )
        return plan.model_copy(update={"title": storyboard.title})
