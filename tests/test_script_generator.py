"""Unit tests for structured educational script generation."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.script_generator import (
    ScriptGeminiError,
    ScriptGenerator,
    ScriptJSONError,
    ScriptValidationError,
)
from app.models.script import Script
from app.prompts.script_prompt import build_script_prompt
from app.services.gemini_client import GeminiAPIError, GeminiClient
from app.utils.debug_recorder import DebugRecorder


def valid_script_response() -> str:
    """Return a valid Gemini JSON response for script tests."""

    return json.dumps(
        {
            "title": "Photosynthesis Explained",
            "topic": "Photosynthesis",
            "total_duration": 60,
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "How Plants Make Food",
                    "narration": (
                        "Plants use light to turn water and carbon dioxide "
                        "into food."
                    ),
                    "estimated_duration": 12,
                    "visuals": [
                        {
                            "type": "text",
                            "content": "Light + Water + Carbon Dioxide",
                            "position": "center",
                            "animation": "write",
                        }
                    ],
                }
            ],
        }
    )


def test_generate_returns_valid_script() -> None:
    """A valid Gemini response becomes a validated Script instance."""

    gemini_client = MagicMock(spec=GeminiClient)
    gemini_client.generate_text.return_value = valid_script_response()

    script = ScriptGenerator(gemini_client).generate("Photosynthesis")

    assert isinstance(script, Script)
    assert script.title == "Photosynthesis Explained"
    assert script.scenes[0].scene_number == 1
    gemini_client.generate_text.assert_called_once()


def test_invalid_json_raises_script_json_error() -> None:
    """Malformed Gemini output raises a JSON-specific service error."""

    gemini_client = MagicMock(spec=GeminiClient)
    gemini_client.generate_text.return_value = "not valid JSON"

    with pytest.raises(ScriptJSONError, match="not valid JSON"):
        ScriptGenerator(gemini_client).generate("Photosynthesis")


def test_validation_failure_raises_script_validation_error() -> None:
    """Valid JSON with missing model fields raises a validation error."""

    gemini_client = MagicMock(spec=GeminiClient)
    gemini_client.generate_text.return_value = json.dumps(
        {
            "title": "Incomplete script",
            "topic": "Photosynthesis",
            "total_duration": 60,
        }
    )

    with pytest.raises(ScriptValidationError, match="Script model"):
        ScriptGenerator(gemini_client).generate("Photosynthesis")


def test_empty_topic_raises_value_error() -> None:
    """Blank topics are rejected without calling Gemini."""

    gemini_client = MagicMock(spec=GeminiClient)

    with pytest.raises(ValueError, match="topic must not be empty"):
        ScriptGenerator(gemini_client).generate("   ")

    gemini_client.generate_text.assert_not_called()


def test_gemini_failure_is_translated() -> None:
    """Gemini client errors become script-generation service errors."""

    gemini_client = MagicMock(spec=GeminiClient)
    gemini_client.generate_text.side_effect = GeminiAPIError("API unavailable")

    with pytest.raises(ScriptGeminiError, match="Gemini failed"):
        ScriptGenerator(gemini_client).generate("Photosynthesis")


def test_prompt_requires_json_only_output() -> None:
    """The template explicitly enforces the required output contract."""

    prompt = build_script_prompt("Photosynthesis")

    assert "Return ONLY one valid JSON object" in prompt
    assert "Do not include markdown" in prompt
    assert "code fences" in prompt
    assert '"topic": "Photosynthesis"' in prompt
    assert "title, text, arrow, box, and circle" in prompt
    assert "Never request images" in prompt
    assert "never overlap" in prompt


def test_llm_prompt_and_raw_response_are_debuggable(tmp_path: Path) -> None:
    """Script generation stores exact LLM input and output when enabled."""

    recorder = DebugRecorder(True, tmp_path / "debug")
    run_dir = recorder.start_run("Photosynthesis")
    assert run_dir is not None
    gemini_client = MagicMock(spec=GeminiClient)
    gemini_client.generate_text.return_value = valid_script_response()

    ScriptGenerator(gemini_client, recorder).generate("Photosynthesis")

    assert "Photosynthesis" in (
        run_dir / "llm" / "prompt.txt"
    ).read_text(encoding="utf-8")
    assert json.loads(
        (run_dir / "llm" / "raw_response.txt").read_text(encoding="utf-8")
    )["title"] == "Photosynthesis Explained"
    assert (
        run_dir / "llm" / "validated_script.json"
    ).is_file()
