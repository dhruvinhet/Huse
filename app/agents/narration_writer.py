"""Specialized storyboard-driven narration writer."""

from app.agents.base import StructuredGeminiAgent
from app.domain.generation import AudienceProfile
from app.domain.narration import NarrationPlan
from app.domain.storyboard import Storyboard
from app.services.gemini_client import GeminiClient


class GeminiNarrationWriter:
    """Write speech that naturally references evolving storyboard visuals."""

    def __init__(self, client: GeminiClient, max_attempts: int = 3) -> None:
        """Initialize the structured narration-writing agent."""

        self._agent = StructuredGeminiAgent(
            client,
            NarrationPlan,
            "Narration Writer",
            max_attempts,
        )

    def write(
        self,
        storyboard: Storyboard,
        audience: AudienceProfile,
    ) -> NarrationPlan:
        """Create concise phrases tied exactly to storyboard beat IDs."""

        expected_beat_ids = [beat.beat_id for beat in storyboard.beats]

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
            "For demonstrate and transform beats, explain the visible change; "
            "for compare beats, state the meaningful contrast; for summarize "
            "beats, reinforce the learning objective without repeating verbatim."
        )
        payload = {
            "storyboard": storyboard.model_dump(mode="json"),
            "audience": audience.model_dump(mode="json"),
            "required_beat_ids": expected_beat_ids,
        }

        def validate_beat_coverage(plan: NarrationPlan) -> None:
            """Require the model to cover the accepted storyboard exactly."""

            actual = [phrase.beat_id for phrase in plan.phrases]
            if actual != expected_beat_ids:
                raise ValueError(
                    "narration phrases must contain exactly one phrase per beat "
                    f"in this order: {expected_beat_ids}; received: {actual}"
                )

        return self._agent.generate(
            instructions,
            payload,
            validator=validate_beat_coverage,
        )
