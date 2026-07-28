"""Official Google GenAI-backed multimodal frame evaluation client."""

from pathlib import Path

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from app.config.settings import settings
from app.services.gemini_client import (
    GeminiAPIError,
    GeminiConfigurationError,
    GeminiEmptyResponseError,
    GeminiNetworkError,
    GeminiTimeoutError,
)


class GeminiVisionClient:
    """Evaluate local frame samples with a configured Gemini model."""

    MODEL_NAME = "gemini-3.5-flash"

    def __init__(self) -> None:
        """Initialize one SDK client without logging credentials."""

        api_key = settings.GEMINI_API_KEY.strip()
        if not api_key:
            raise GeminiConfigurationError("GEMINI_API_KEY is required")
        timeout_milliseconds = (
            int(settings.GEMINI_TIMEOUT_SECONDS * 1_000)
            if settings.GEMINI_TIMEOUT_SECONDS is not None
            else None
        )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_milliseconds),
        )

    def evaluate_images(self, prompt: str, image_paths: list[Path]) -> str:
        """Return stripped structured criticism for local images."""

        if not prompt.strip() or not image_paths:
            raise ValueError("prompt and image paths are required")
        images = [Image.open(path).convert("RGB") for path in image_paths]
        try:
            response = self._client.models.generate_content(
                model=self.MODEL_NAME,
                contents=[prompt, *images],
                config=types.GenerateContentConfig(temperature=0.1),
            )
        except httpx.TimeoutException as exc:
            raise GeminiTimeoutError("multimodal request timed out") from exc
        except httpx.NetworkError as exc:
            raise GeminiNetworkError("multimodal network request failed") from exc
        except genai_errors.APIError as exc:
            raise GeminiAPIError(f"multimodal API request failed: {exc}") from exc
        finally:
            for image in images:
                image.close()
        if not response.text or not response.text.strip():
            raise GeminiEmptyResponseError("multimodal response was empty")
        return response.text.strip()
