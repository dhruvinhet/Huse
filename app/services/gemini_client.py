"""Provider-aware text-generation client for Gemini and NVIDIA APIs."""

from time import sleep

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from loguru import logger

from app.config.settings import settings


class GeminiClientError(RuntimeError):
    """Base exception for Gemini client failures."""


class GeminiConfigurationError(GeminiClientError):
    """Raised when required Gemini configuration is missing or invalid."""


class GeminiTimeoutError(GeminiClientError):
    """Raised when the Gemini transport reports a timeout."""


class GeminiNetworkError(GeminiClientError):
    """Raised when Gemini cannot be reached over the network."""


class GeminiAPIError(GeminiClientError):
    """Raised when the Gemini API rejects or cannot process a request."""


class GeminiEmptyResponseError(GeminiClientError):
    """Raised when Gemini returns no usable response text."""


class GeminiClient:
    """Provide a shared text-generation interface for the configured provider.

    The historical class name is retained because the rest of the application
    injects ``GeminiClient``. The actual provider is selected with
    ``AI_PROVIDER=gemini`` or ``AI_PROVIDER=nvidia`` in ``.env``.
    """

    _DEFAULT_PROVIDER = "gemini"
    _NVIDIA_PROVIDER_NAMES = {"nvidia", "nvidea"}
    # Provider outages are transient; three bounded attempts cover a brief
    # 503/504 window without making a failed run wait indefinitely.
    _NVIDIA_REQUEST_ATTEMPTS = 3
    _NVIDIA_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

    @property
    def PROVIDER(self) -> str:
        """Return the normalized provider name used by this client."""

        configured_provider = getattr(
            settings, "AI_PROVIDER", self._DEFAULT_PROVIDER
        ).strip().lower()
        if configured_provider == "nvidea":
            return "nvidia"
        return configured_provider

    @property
    def MODEL_NAME(self) -> str:
        if self.PROVIDER == "nvidia":
            return getattr(
                settings, "NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"
            )
        return settings.GEMINI_MODEL


    def __init__(self) -> None:
        """Validate configuration and initialize the selected client."""

        if self.PROVIDER not in {"gemini", "nvidia"}:
            raise GeminiConfigurationError(
                "AI_PROVIDER must be 'gemini' or 'nvidia'."
            )

        if self.PROVIDER == "nvidia":
            api_key = getattr(settings, "NVIDIA_API_KEY", "").strip()
            if not api_key:
                raise GeminiConfigurationError(
                    "NVIDIA_API_KEY is missing; set it in the environment or .env file."
                )
            self._client = None
            return

        api_key = settings.GEMINI_API_KEY.strip()
        if not api_key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is missing; set it in the environment or .env file."
            )

        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=None),
        )

    def generate_text(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate and return stripped text for a nonempty prompt."""

        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        logger.info(
            "Sending {} request (model={}, prompt_characters={})",
            self.PROVIDER,
            self.MODEL_NAME,
            len(prompt),
        )
        logger.debug("{} request prompt: {}", self.PROVIDER, prompt)

        if self.PROVIDER == "nvidia":
            return self._generate_nvidia(prompt, temperature)

        return self._generate_gemini(prompt, temperature)

    def _generate_gemini(self, prompt: str, temperature: float) -> str:
        """Generate text through the Google GenAI SDK."""

        try:
            response = self._client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
        except httpx.TimeoutException as exc:
            logger.error("Gemini request timed out: {}", exc)
            raise GeminiTimeoutError("Gemini request timed out.") from exc
        except httpx.NetworkError as exc:
            logger.error("Gemini network request failed: {}", exc)
            raise GeminiNetworkError(
                "Gemini could not be reached due to a network failure."
            ) from exc
        except genai_errors.APIError as exc:
            logger.error("Gemini API request failed: {}", exc)
            raise GeminiAPIError(f"Gemini API request failed: {exc}") from exc

        response_text = response.text
        if not response_text or not response_text.strip():
            logger.error("Gemini returned an empty text response.")
            raise GeminiEmptyResponseError(
                "Gemini returned an empty text response."
            )

        stripped_text = response_text.strip()
        logger.info(
            "Gemini response received (characters={})",
            len(stripped_text),
        )
        logger.debug("Gemini response text: {}", stripped_text)
        return stripped_text

    def _generate_nvidia(self, prompt: str, temperature: float) -> str:
        """Generate text through NVIDIA's OpenAI-compatible endpoint."""

        base_url = getattr(
            settings,
            "NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        ).rstrip("/")
        payload = {
            "model": self.MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": getattr(settings, "NVIDIA_MAX_TOKENS", 16384),
            "stream": False,
        }
        if self.MODEL_NAME == "nvidia/nemotron-3-nano-30b-a3b":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {
            "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        response_payload: object | None = None
        for attempt in range(1, self._NVIDIA_REQUEST_ATTEMPTS + 1):
            try:
                response = httpx.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=None,
                )
                response.raise_for_status()
                response_payload = response.json()
                break
            except httpx.TimeoutException as exc:
                if attempt < self._NVIDIA_REQUEST_ATTEMPTS:
                    self._log_nvidia_retry(attempt, "transport timeout")
                    sleep(float(attempt))
                    continue
                logger.error("NVIDIA request timed out: {}", exc)
                raise GeminiTimeoutError("NVIDIA request timed out.") from exc
            except httpx.NetworkError as exc:
                if attempt < self._NVIDIA_REQUEST_ATTEMPTS:
                    self._log_nvidia_retry(attempt, "network failure")
                    sleep(float(attempt))
                    continue
                logger.error("NVIDIA network request failed: {}", exc)
                raise GeminiNetworkError(
                    "NVIDIA could not be reached due to a network failure."
                ) from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if (
                    status in self._NVIDIA_RETRY_STATUS_CODES
                    and attempt < self._NVIDIA_REQUEST_ATTEMPTS
                ):
                    self._log_nvidia_retry(attempt, f"HTTP {status}")
                    sleep(float(attempt))
                    continue
                logger.error("NVIDIA API request failed: {}", exc)
                raise GeminiAPIError(
                    f"NVIDIA API request failed: {exc}"
                ) from exc
            except httpx.HTTPError as exc:
                logger.error("NVIDIA API request failed: {}", exc)
                raise GeminiAPIError(f"NVIDIA API request failed: {exc}") from exc
            except ValueError as exc:
                logger.error("NVIDIA returned invalid JSON: {}", exc)
                raise GeminiAPIError(
                    "NVIDIA returned an invalid JSON response."
                ) from exc

        if response_payload is None:
            raise GeminiAPIError("NVIDIA returned no response payload.")

        response_text = _extract_nvidia_response_text(response_payload)
        if not response_text:
            logger.error("NVIDIA returned an empty text response.")
            raise GeminiEmptyResponseError("NVIDIA returned an empty text response.")

        stripped_text = response_text.strip()
        logger.info(
            "NVIDIA response received (characters={})",
            len(stripped_text),
        )
        logger.debug("NVIDIA response text: {}", stripped_text)
        return stripped_text

    @staticmethod
    def _log_nvidia_retry(attempt: int, reason: str) -> None:
        """Record one bounded retry without logging secrets or response bodies."""

        logger.warning(
            "NVIDIA request attempt {} failed with {}; retrying.",
            attempt,
            reason,
        )


def _extract_nvidia_response_text(payload: object) -> str:
    """Extract text from a standard NVIDIA/OpenAI chat-completion payload."""

    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "".join(text_parts)
    return ""
