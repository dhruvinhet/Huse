"""Specialized AI lesson planner."""

from app.agents.base import StructuredGeminiAgent
from app.domain.generation import GenerationRequest
from app.domain.lesson import LessonPlan
from app.services.gemini_client import GeminiClient


class GeminiLessonPlanner:
    """Create a concept graph without narration or visual geometry."""

    def __init__(self, client: GeminiClient, max_attempts: int = 5) -> None:
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
            "and unique. The concept_graph.teaching_sequence array must contain "
            "each node's concept_id exactly once, in prerequisite-respecting "
            "order; use IDs from nodes[].concept_id, never human-readable labels "
            "from nodes[].label. Before returning, compare the set of sequence "
            "values with the set of node concept_id values. Visual affordances describe "
            "useful diagram families such as pipeline, matrix, graph, timeline, "
            "array, tree, probability distribution, or component hierarchy. Do "
            "not write narration, storyboard operations, animations, or coordinates. "
            "Keep the lesson compact: use only 3–6 concepts and concise strings; "
            "do not add unrelated examples or long assessment text. "
            "Do not copy JSON Schema metadata into the result: never output keys "
            "such as $defs, $schema, definitions, $ref, properties, or required "
            "inside concept_graph."
        )
        return self._agent.generate(instructions, request.model_dump(mode="json"))
