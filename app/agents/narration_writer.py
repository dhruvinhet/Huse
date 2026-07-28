"""Specialized storyboard-driven narration writer."""

from app.agents.base import StructuredGeminiAgent
from app.domain.generation import AudienceProfile
from app.domain.narration import NarrationPlan
from app.domain.storyboard import Storyboard
from app.services.gemini_client import GeminiClient


class GeminiNarrationWriter:
    """Write speech that naturally references evolving storyboard visuals."""

    def __init__(self, client: GeminiClient, max_attempts: int = 2) -> None:
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

        instructions = (
            "Write natural educational narration from the accepted storyboard. "
            "Use at least one phrase per beat and reference what is appearing, "
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
        }
        return self._agent.generate(instructions, payload)
