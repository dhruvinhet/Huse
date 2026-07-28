"""Service for generating and validating structured educational scripts."""

import json
from typing import Any

from loguru import logger
from pydantic import ValidationError

from app.models.script import Script
from app.prompts.script_prompt import build_script_prompt
from app.services.gemini_client import GeminiClient, GeminiClientError
from app.utils.debug_recorder import DebugRecorder


class ScriptGeneratorError(RuntimeError):
    """Base exception for structured script generation failures."""


class ScriptGeminiError(ScriptGeneratorError):
    """Raised when Gemini cannot return a response for script generation."""


class ScriptJSONError(ScriptGeneratorError):
    """Raised when Gemini returns content that is not valid JSON."""


class ScriptValidationError(ScriptGeneratorError):
    """Raised when returned JSON does not match the Script model."""


class ScriptGenerator:
    """Generate a validated Script using an injected Gemini client."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        debug_recorder: DebugRecorder | None = None,
    ) -> None:
        """Store the Gemini client used for text generation."""

        self._gemini_client = gemini_client
        self._debug_recorder = debug_recorder

    def generate(self, topic: str) -> Script:
        """Generate, parse, and validate an educational script for a topic."""

        normalized_topic = topic.strip()
        if not normalized_topic:
            raise ValueError("topic must not be empty")

        logger.info("Script generation topic received: {}", normalized_topic)
        prompt = build_script_prompt(normalized_topic)
        if self._debug_recorder is not None:
            self._debug_recorder.write_text("llm/prompt.txt", prompt)
        logger.info("Sending script-generation prompt to Gemini.")
        logger.debug("Script-generation prompt: {}", prompt)

        try:
            response_text = self._gemini_client.generate_text(prompt)
        except GeminiClientError as exc:
            logger.error("Gemini failed during script generation: {}", exc)
            raise ScriptGeminiError(
                f"Gemini failed during script generation: {exc}"
            ) from exc

        logger.info(
            "Script-generation response received (characters={}).",
            len(response_text),
        )
        logger.debug("Script-generation response: {}", response_text)
        if self._debug_recorder is not None:
            self._debug_recorder.write_text(
                "llm/raw_response.txt",
                response_text,
            )
        payload = self._parse_response(response_text)

        try:
            script = Script.model_validate(payload)
        except ValidationError as exc:
            logger.error("Generated script validation failed: {}", exc)
            raise ScriptValidationError(
                "Gemini response JSON does not match the Script model."
            ) from exc

        logger.info(
            "Generated script validated successfully (scenes={}).",
            len(script.scenes),
        )
        if self._debug_recorder is not None:
            self._debug_recorder.write_json(
                "llm/validated_script.json",
                script.model_dump(mode="json"),
            )
        return script

    @staticmethod
    def _parse_response(response_text: str) -> Any:
        """Parse a Gemini response as JSON with a domain-specific error."""

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as exc:
            logger.error("Generated script contains invalid JSON: {}", exc)
            raise ScriptJSONError(
                "Gemini response is not valid JSON."
            ) from exc
