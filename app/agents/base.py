"""Reusable structured-output behavior for bounded AI planners."""

import json
from collections.abc import Callable
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

    def generate(
        self,
        instructions: str,
        input_payload: object,
        validator: Callable[[OutputT], None] | None = None,
    ) -> OutputT:
        """Generate, validate, and optionally repair one JSON artifact."""

        provider = getattr(self._client, "PROVIDER", "").strip().lower()
        schema = self._output_type.model_json_schema()
        if provider == "nvidia":
            schema = _compact_json_schema(schema)
            schema_text = json.dumps(
                schema,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            payload_text = json.dumps(
                input_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            schema_text = json.dumps(schema, ensure_ascii=False)
            payload_text = json.dumps(
                input_payload,
                ensure_ascii=False,
                indent=2,
            )
        base_prompt = (
            f"You are the {self._agent_name}.\n\n"
            f"{instructions.strip()}\n\n"
            "Return only one JSON object. Do not use markdown or code fences. "
            "Every array constrained by minItems must contain at least one "
            "item. Every enum or literal field must use one of its schema values "
            "exactly; never invent a replacement value. "
            "The output must validate against this JSON Schema:\n"
            f"{schema_text}\n\n"
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
                    "specified contract defects. For `too_short` errors, add at "
                    "least one valid item. For `literal_error` errors, replace "
                    "the value with one of the allowed literal values shown in "
                    "the schema. Return the complete object, never a patch.\n"
                    "Validation error:\n"
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
                artifact = self._output_type.model_validate_json(
                    _prepare_json_response(response)
                )
                if validator is not None:
                    validator(artifact)
            except ValidationError as exc:
                if _is_truncated_json_error(exc):
                    previous_response = ""
                    previous_error = (
                        "The previous response was truncated before the JSON "
                        "object was complete. Regenerate the entire object from "
                        "the input and keep it compact enough to finish."
                    )
                else:
                    previous_response = "" if provider == "nvidia" else response
                    previous_error = str(exc)
                logger.warning(
                    "Structured agent response failed validation "
                    "(agent={}, attempt={}).",
                    self._agent_name,
                    attempt,
                )
                continue
            except ValueError as exc:
                previous_response = "" if provider == "nvidia" else response
                previous_error = str(exc)
                logger.warning(
                    "Structured agent response failed semantic validation "
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


def _is_truncated_json_error(error: ValidationError) -> bool:
    """Return whether Pydantic rejected an incomplete JSON response."""

    for detail in error.errors():
        if detail.get("type") != "json_invalid":
            continue
        message = str(detail.get("msg", "")).lower()
        context = str(detail.get("ctx", "")).lower()
        if "eof" in message or "eof" in context:
            return True
    return False


def _compact_json_schema(value: object) -> object:
    """Remove prose-only schema metadata for smaller NVIDIA prompts."""

    if isinstance(value, list):
        return [_compact_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    prose_keys = {"title", "description", "examples", "default"}
    return {
        key: _compact_json_schema(item)
        for key, item in value.items()
        if key not in prose_keys
    }


def _prepare_json_response(response: str) -> str:
    """Remove harmless prose or markdown wrapped around one JSON artifact."""

    stripped = response.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    starts = [
        position for position in (stripped.find("{"), stripped.find("["))
    ]
    starts = [position for position in starts if position >= 0]
    if not starts:
        return stripped
    candidate = stripped[min(starts):]
    try:
        parsed, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return stripped
    return json.dumps(parsed, ensure_ascii=False)
