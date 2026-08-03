"""Tests for compact model intent and deterministic scene compilation."""

import json

from app.agents.storyboard_planner import GeminiStoryboardPlanner
from app.domain.generation import AudienceProfile
from app.domain.lesson import LessonPlan
from app.domain.operations import OperationType
from app.planning import PedagogyRouter
from tests.test_domain_v2 import concept_graph


class IntentClient:
    """Return one compact intent while recording the provider prompt."""

    PROVIDER = "nvidia"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        del temperature
        self.prompts.append(prompt)
        return json.dumps(self.payload)


def test_model_emits_visual_intent_and_compiler_owns_scene_graph() -> None:
    """The provider never has to invent object IDs or visual operations."""

    lesson = LessonPlan(
        title="Transformation",
        summary="Input becomes output.",
        concept_graph=concept_graph(),
    )
    audience = AudienceProfile(learning_goal="Understand transformation")
    pedagogy = PedagogyRouter().route(lesson, audience)
    shots = []
    for index, routed in enumerate(pedagogy.shots):
        shots.append(
            {
                "shot_id": routed.shot_id,
                "concept_ids": ["input"] if index < 2 else ["output"],
                "relation": "transforms_to",
                "focal_object": "Incoming data" if index < 2 else "Result",
                "evidence": "The output differs from the input state.",
                "transformation": "Apply the process and reveal the result.",
                "renderer_operator": "process",
            }
        )
    client = IntentClient(
        {
            "lesson_focus": "How input becomes output",
            "shots": shots,
        }
    )
    planner = GeminiStoryboardPlanner(client, max_attempts=1)

    board = planner.plan_with_pedagogy(lesson, [], [], pedagogy)

    prompt = client.prompts[0]
    assert "VisualObjectSpec" not in prompt
    assert "operation_id" not in prompt
    assert "target_ids" not in prompt
    assert planner.last_intent is not None
    assert planner.last_intent.shots[0].focal_object == "Incoming data"
    assert board.document_id.startswith("intent_")
    assert board.beats[0].operations[0].operation is OperationType.CREATE
    assert all(
        operation.operation is not OperationType.CREATE
        for beat in board.beats[1:]
        for operation in beat.operations
    )
    created = board.beats[0].operations[0].arguments["objects"][0]
    assert created["constraints"]
    assert any(child["kind"] == "connector" for child in created["children"])
    assert all(beat.attention[0].target_ids for beat in board.beats)
    assert any(
        "concept_input" in target
        for target in board.beats[0].attention[0].target_ids
    )


def test_visual_intent_schema_is_much_smaller_than_storyboard_schema() -> None:
    """High-level output removes low-level object and lifecycle schema cost."""

    from app.domain.storyboard import Storyboard
    from app.domain.visual_intent import VisualIntent

    intent_size = len(json.dumps(VisualIntent.model_json_schema()))
    storyboard_size = len(json.dumps(Storyboard.model_json_schema()))

    assert intent_size < storyboard_size * 0.35
