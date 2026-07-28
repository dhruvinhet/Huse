"""Reusable text-generation client for the Gemini API."""

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
    """Provide a minimal, reusable interface to Gemini text generation."""

    MODEL_NAME = "gemini-3.5-flash"

    def __init__(self) -> None:
        """Validate configuration and initialize one SDK client instance."""

        api_key = settings.GEMINI_API_KEY.strip()
        if not api_key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is missing; set it in the environment or .env file."
            )

        timeout_milliseconds = (
            int(settings.GEMINI_TIMEOUT_SECONDS * 1_000)
            if settings.GEMINI_TIMEOUT_SECONDS is not None
            else None
        )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_milliseconds),
        )

    def generate_text(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate and return stripped text for a nonempty prompt."""

        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        logger.info(
            "Sending Gemini request (model={}, prompt_characters={})",
            self.MODEL_NAME,
            len(prompt),
        )
        logger.debug("Gemini request prompt: {}", prompt)

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
