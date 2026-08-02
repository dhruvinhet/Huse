"""Tests for provider-neutral structured-output retries."""

from typing import Literal

from pydantic import BaseModel, Field

from app.agents.base import StructuredGeminiAgent


class TinyArtifact(BaseModel):
    """Small artifact used to exercise the retry behavior."""

    name: str


class StrictArtifact(BaseModel):
    """Artifact with the two common model-contract failure types."""

    status: Literal["ok"]
    items: list[str] = Field(min_length=1)


class StubClient:
    """Return queued model responses while recording the prompts."""

    PROVIDER = "nvidia"

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate_text(self, prompt: str, temperature: float = 0.3) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


def test_truncated_json_gets_a_clean_retry() -> None:
    """EOF failures do not cause the broken response to be echoed on retry."""

    client = StubClient([
        '{"name": "unfinished',
        '{"name": "complete"}',
    ])
    agent = StructuredGeminiAgent(
        client, TinyArtifact, "Tiny Planner", max_attempts=2
    )

    result = agent.generate("Return a tiny artifact.", {"input": "value"})

    assert result.name == "complete"
    assert len(client.prompts) == 2
    assert "truncated before the JSON object was complete" in client.prompts[1]
    assert '{"name": "unfinished' not in client.prompts[1]


def test_nvidia_validation_retry_does_not_repeat_invalid_response() -> None:
    """NVIDIA retries receive defects and schema guidance, not stale JSON."""

    client = StubClient([
        '{"status": "visualize", "items": []}',
        '{"status": "ok", "items": ["complete"]}',
    ])
    agent = StructuredGeminiAgent(
        client, StrictArtifact, "Strict Planner", max_attempts=2
    )

    result = agent.generate("Return a strict artifact.", {"input": "value"})

    assert result.status == "ok"
    assert result.items == ["complete"]
    assert "too_short" in client.prompts[1]
    assert "literal_error" in client.prompts[1]
    assert '"visualize"' not in client.prompts[1]


def test_custom_validator_gets_a_repair_attempt() -> None:
    """Semantic contracts can participate in the same bounded retry loop."""

    client = StubClient([
        '{"name": "wrong"}',
        '{"name": "complete"}',
    ])
    agent = StructuredGeminiAgent(
        client, TinyArtifact, "Tiny Planner", max_attempts=2
    )

    def require_complete(item: TinyArtifact) -> None:
        if item.name != "complete":
            raise ValueError("name must be complete")

    agent.generate(
        "Return a tiny artifact.",
        {"input": "value"},
        validator=require_complete,
    )

    assert "name must be complete" in client.prompts[1]


def test_json_artifact_can_follow_model_prose() -> None:
    """A reasoning preamble does not block an otherwise valid artifact."""

    client = StubClient([
        'I will now provide the result.\n```json\n{"name": "complete"}\n```',
    ])
    agent = StructuredGeminiAgent(
        client, TinyArtifact, "Tiny Planner", max_attempts=1
    )

    result = agent.generate("Return a tiny artifact.", {"input": "value"})

    assert result.name == "complete"
