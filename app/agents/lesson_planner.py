"""Specialized AI lesson planner."""

from app.agents.base import StructuredGeminiAgent
from app.domain.generation import GenerationRequest
from app.domain.lesson import LessonPlan
from app.services.gemini_client import GeminiClient


class GeminiLessonPlanner:
    """Create a concept graph without narration or visual geometry."""

    def __init__(self, client: GeminiClient, max_attempts: int = 2) -> None:
        """Initialize the structured lesson-planning agent."""

        self._agent = StructuredGeminiAgent(
            client,
            LessonPlan,
            "Lesson Planner",
            max_attempts,
        )

    def plan(self, request: GenerationRequest) -> LessonPlan:
        """Plan learning objectives, concepts, dependencies, and order."""

        instructions = (
            "Create an accurate educational lesson plan for the requested "
            "audience and duration. Identify only the concepts needed to meet "
            "the learning goal. Concept IDs and edge IDs must be stable, concise, "
            "and unique. Teaching sequence must contain every concept exactly "
            "once in prerequisite-respecting order. Visual affordances describe "
            "useful diagram families such as pipeline, matrix, graph, timeline, "
            "array, tree, probability distribution, or component hierarchy. Do "
            "not write narration, storyboard operations, animations, or coordinates."
        )
        return self._agent.generate(instructions, request.model_dump(mode="json"))
