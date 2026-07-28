"""Reusable structured-output behavior for bounded AI planners."""

import json
from typing import Generic, TypeVar

from loguru import logger
from pydantic import BaseModel as PydanticModel
from pydantic import ValidationError

from app.services.gemini_client import GeminiClient, GeminiClientError


OutputT = TypeVar("OutputT", bound=PydanticModel)


class StructuredAgentError(RuntimeError):
    """Raised when a bounded AI planner cannot produce valid output."""


class StructuredGeminiAgent(Generic[OutputT]):
    """Generate one strict Pydantic artifact with bounded repair attempts."""

    def __init__(
        self,
        client: GeminiClient,
        output_type: type[OutputT],
        agent_name: str,
        max_attempts: int = 2,
    ) -> None:
        """Store dependencies and a strict retry budget."""

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self._client = client
        self._output_type = output_type
        self._agent_name = agent_name
        self._max_attempts = max_attempts

    def generate(self, instructions: str, input_payload: object) -> OutputT:
        """Generate, validate, and optionally repair one JSON artifact."""

        schema = self._output_type.model_json_schema()
        payload_text = json.dumps(input_payload, ensure_ascii=False, indent=2)
        base_prompt = (
            f"You are the {self._agent_name}.\n\n"
            f"{instructions.strip()}\n\n"
            "Return only one JSON object. Do not use markdown or code fences. "
            "The output must validate against this JSON Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}\n\n"
            "Input artifact:\n"
            f"{payload_text}"
        )
        previous_response = ""
        previous_error = ""
        for attempt in range(1, self._max_attempts + 1):
            prompt = base_prompt
            if previous_error:
                prompt += (
                    "\n\nThe previous response failed validation. Repair only the "
                    "specified contract defects.\nValidation error:\n"
                    f"{previous_error}\nPrevious response:\n{previous_response}"
                )
            logger.info(
                "Structured agent request started (agent={}, attempt={}/{}).",
                self._agent_name,
                attempt,
                self._max_attempts,
            )
            try:
                response = self._client.generate_text(prompt, temperature=0.2)
            except GeminiClientError as exc:
                raise StructuredAgentError(
                    f"{self._agent_name} provider request failed"
                ) from exc
            try:
                artifact = self._output_type.model_validate_json(response)
            except ValidationError as exc:
                previous_response = response
                previous_error = str(exc)
                logger.warning(
                    "Structured agent response failed validation "
                    "(agent={}, attempt={}).",
                    self._agent_name,
                    attempt,
                )
                continue
            logger.info(
                "Structured agent response validated (agent={}, attempt={}).",
                self._agent_name,
                attempt,
            )
            return artifact
        raise StructuredAgentError(
            f"{self._agent_name} failed to produce valid output after "
            f"{self._max_attempts} attempts: {previous_error}"
        )
