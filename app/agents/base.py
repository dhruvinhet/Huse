"""Reusable structured-output behavior for bounded AI planners."""

import json
from collections.abc import Callable
from inspect import signature
from typing import Generic, TypeVar

from loguru import logger
from pydantic import BaseModel as PydanticModel
from pydantic import ValidationError

from app.services.gemini_client import GeminiClient, GeminiClientError


OutputT = TypeVar("OutputT", bound=PydanticModel)

_JSON_PATCH_SCHEMA: dict[str, object] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["add", "replace", "remove"]},
            "path": {"type": "string"},
            "value": {},
        },
        "required": ["op", "path"],
        "additionalProperties": False,
    },
}


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
        provider_schema = provider in {"gemini", "nvidia"} and self._supports_schema()
        schema = self._output_type.model_json_schema()
        if provider == "nvidia" and not provider_schema:
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
            schema_text = ""
            payload_text = json.dumps(
                input_payload,
                ensure_ascii=False,
                indent=2,
            )
        repair_payload_text = json.dumps(
            _compact_repair_value(input_payload),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        schema_instruction = (
            "The provider enforces the response schema; return only the "
            "schema-conforming JSON object."
            if provider_schema
            else (
                "The output must validate against this JSON Schema:\n"
                f"{schema_text}"
            )
        )
        base_prompt = (
            f"You are the {self._agent_name}.\n\n"
            f"{instructions.strip()}\n\n"
            "Return only one JSON object. Do not use markdown or code fences. "
            "Every array constrained by minItems must contain at least one "
            "item. Every enum or literal field must use one of its schema values "
            "exactly; never invent a replacement value. "
            f"{schema_instruction}\n\n"
            "Input artifact:\n"
            f"{payload_text}"
        )
        previous_error = ""
        repair_candidate: dict[str, object] | None = None
        patch_retry = False
        for attempt in range(1, self._max_attempts + 1):
            if previous_error:
                if patch_retry and repair_candidate is not None:
                    patch_context = json.dumps(
                        _compact_repair_value(repair_candidate),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    prompt = (
                        f"You are the {self._agent_name}. Return only an RFC 6902 "
                        "JSON Patch array; do not return the full artifact or "
                        "commentary. Apply the smallest changes needed to the "
                        "listed defects and keep every valid field unchanged. "
                        "Use add, replace, or remove operations with JSON Pointer "
                        "paths.\n\nCurrent candidate:\n"
                        f"{patch_context}\n\n"
                        "Contract defects:\n"
                        f"{previous_error}"
                    )
                else:
                    retry_payload = (
                        repair_payload_text
                        if provider_schema
                        else payload_text
                    )
                    prompt = (
                        f"You are the {self._agent_name}. Return one complete JSON "
                        "object and no commentary. This is a targeted repair; keep "
                        "all valid fields stable and correct only the listed defects. "
                        f"{schema_instruction}\n\n"
                        "Compact repair context (valid fields are preserved; do "
                        "not echo the full original artifact):\n"
                        f"{retry_payload}\n\n"
                        "This is a compact repair attempt. Regenerate the complete "
                        "JSON object from the input artifact, correcting only the "
                        "reported contract defects. Do not repeat a previous response "
                        "or add commentary. For `too_short` errors, add at least one "
                        "valid item; for `literal_error`, use the allowed value.\n"
                        "Contract defects:\n"
                        f"{previous_error}"
                    )
            else:
                prompt = base_prompt
            logger.info(
                "Structured agent request started (agent={}, attempt={}/{}).",
                self._agent_name,
                attempt,
                self._max_attempts,
            )
            try:
                response = self._request(
                    prompt,
                    provider,
                    patch_mode=patch_retry and repair_candidate is not None,
                )
            except GeminiClientError as exc:
                raise StructuredAgentError(
                    f"{self._agent_name} provider request failed"
                ) from exc
            try:
                prepared = _prepare_json_response(response)
                decoded = json.loads(prepared)
                if patch_retry and repair_candidate is not None and isinstance(decoded, list):
                    decoded = _apply_json_patch(repair_candidate, decoded)
                artifact = self._output_type.model_validate(decoded)
                if validator is not None:
                    validator(artifact)
            except ValidationError as exc:
                decoded_candidate = _decode_json_object(response)
                repair_candidate = decoded_candidate or repair_candidate
                if _is_truncated_json_error(exc):
                    previous_error = (
                        "The previous response was truncated before the JSON "
                        "object was complete. Regenerate the entire object from "
                        "the input and keep it compact enough to finish."
                    )
                else:
                    previous_error = str(exc)
                    patch_retry = bool(provider_schema and repair_candidate)
                logger.warning(
                    "Structured agent response failed validation "
                    "(agent={}, attempt={}).",
                    self._agent_name,
                    attempt,
                )
                continue
            except ValueError as exc:
                decoded_candidate = _decode_json_object(response)
                repair_candidate = decoded_candidate or repair_candidate
                if _is_truncated_decode_error(exc):
                    previous_error = (
                        "The previous response was truncated before the JSON "
                        "object was complete. Regenerate the entire object from "
                        "the input and keep it compact enough to finish."
                    )
                    patch_retry = False
                else:
                    previous_error = str(exc)
                    patch_retry = bool(provider_schema and repair_candidate)
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

    def _request(
        self,
        prompt: str,
        provider: str,
        *,
        patch_mode: bool = False,
    ) -> str:
        """Use provider-enforced schema when the injected client supports it."""

        if provider in {"gemini", "nvidia"} and self._supports_schema():
            return self._client.generate_text(
                prompt,
                temperature=0.2,
                response_schema=(
                    _JSON_PATCH_SCHEMA if patch_mode else self._output_type
                ),
            )
        return self._client.generate_text(prompt, temperature=0.2)

    def _supports_schema(self) -> bool:
        """Return whether the injected provider accepts a response schema."""

        parameters = signature(self._client.generate_text).parameters.values()
        return any(
            item.name == "response_schema" or item.kind.name == "VAR_KEYWORD"
            for item in parameters
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


def _is_truncated_decode_error(error: ValueError) -> bool:
    """Recognize the JSON decoder's incomplete-response diagnostics."""

    message = str(error).casefold()
    return any(
        marker in message
        for marker in ("unterminated", "expecting value", "eof", "end of json")
    )


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


def _compact_repair_value(value: object, depth: int = 0) -> object:
    """Bound retry context so repair requests do not resend giant artifacts."""

    if depth >= 4:
        return "..."
    if isinstance(value, dict):
        items = list(value.items())[:32]
        return {
            str(key): _compact_repair_value(item, depth + 1)
            for key, item in items
        }
    if isinstance(value, list):
        return [
            _compact_repair_value(item, depth + 1)
            for item in value[:12]
        ]
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return normalized if len(normalized) <= 280 else normalized[:277] + "..."
    return value


def _decode_json_object(response: str) -> dict[str, object] | None:
    """Keep a parseable failed candidate as the base for a JSON Patch retry."""

    try:
        value = json.loads(_prepare_json_response(response))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _apply_json_patch(
    document: dict[str, object],
    operations: list[object],
) -> dict[str, object]:
    """Apply the bounded patch vocabulary emitted by provider repair calls."""

    result: object = json.loads(json.dumps(document, ensure_ascii=False))
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("JSON Patch operations must be objects")
        op = operation.get("op")
        path = operation.get("path")
        if op not in {"add", "replace", "remove"} or not isinstance(path, str):
            raise ValueError("JSON Patch operation has an invalid op or path")
        tokens = [
            token.replace("~1", "/").replace("~0", "~")
            for token in path.removeprefix("/").split("/")
            if path != "/"
        ]
        if path in {"", "/"}:
            if op == "remove":
                raise ValueError("cannot remove the root artifact")
            result = operation.get("value")
            continue
        parent = result
        for token in tokens[:-1]:
            if isinstance(parent, list):
                parent = parent[int(token)]
            elif isinstance(parent, dict):
                parent = parent[token]
            else:
                raise ValueError("JSON Patch path does not address a container")
        key = tokens[-1]
        value = operation.get("value")
        if isinstance(parent, list):
            index = len(parent) if key == "-" else int(key)
            if op == "add":
                parent.insert(index, value)
            elif op == "replace":
                parent[index] = value
            else:
                parent.pop(index)
        elif isinstance(parent, dict):
            if op == "remove":
                parent.pop(key)
            else:
                parent[key] = value
        else:
            raise ValueError("JSON Patch path does not address a container")
    if not isinstance(result, dict):
        raise ValueError("patched artifact must remain a JSON object")
    return result


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
