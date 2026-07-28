"""AI-assisted semantic criticism of concept and storyboard artifacts."""

from app.agents.base import StructuredGeminiAgent
from app.domain.quality import QualityReport
from app.services.gemini_client import GeminiClient


class GeminiStoryboardCritic:
    """Evaluate educational coverage and diagram intent before rendering."""

    def __init__(self, client: GeminiClient, max_attempts: int = 2) -> None:
        """Initialize a bounded structured critic."""

        self._agent = StructuredGeminiAgent(
            client,
            QualityReport,
            "Educational Visual Critic",
            max_attempts,
        )

    def evaluate(
        self,
        artifact_id: str,
        artifact: object,
        context: dict[str, object],
    ) -> QualityReport:
        """Score semantic alignment and return targeted repair findings."""

        dump = getattr(artifact, "model_dump", None)
        artifact_value = dump(mode="json") if callable(dump) else artifact
        context_value = {
            key: (
                value.model_dump(mode="json")
                if callable(getattr(value, "model_dump", None))
                else value
            )
            for key, value in context.items()
        }
        return self._agent.generate(
            (
                "Evaluate whether the artifact teaches every important concept "
                "with correct, labeled, progressively disclosed visuals. Detect "
                "empty decorative shapes, missing relationships, misleading "
                "diagrams, excessive cognitive load, weak attention planning, and "
                "unexplained static periods. Scores are between zero and one. Use "
                "decision=repair for actionable planner defects and fail only for "
                "unsafe or fundamentally invalid content."
            ),
            {
                "artifact_id": artifact_id,
                "artifact": artifact_value,
                "context": context_value,
            },
        )
