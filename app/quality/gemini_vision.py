"""Provider-aware multimodal frame evaluation client."""

import base64
import mimetypes
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
    _extract_nvidia_response_text,
)


class GeminiVisionClient:
    """Evaluate local frame samples with the configured AI provider."""

    @property
    def MODEL_NAME(self) -> str:
        if self.PROVIDER == "nvidia":
            return getattr(
                settings,
                "NVIDIA_VISION_MODEL",
                "google/gemma-3n-e4b-it",
            )
        return settings.GEMINI_MODEL

    @property
    def PROVIDER(self) -> str:
        """Return the normalized provider name used by this client."""

        provider = getattr(settings, "AI_PROVIDER", "gemini").strip().lower()
        return "nvidia" if provider == "nvidea" else provider

    def __init__(self) -> None:
        """Initialize the selected provider client without logging credentials."""

        if self.PROVIDER not in {"gemini", "nvidia"}:
            raise GeminiConfigurationError(
                "AI_PROVIDER must be 'gemini' or 'nvidia'."
            )

        if self.PROVIDER == "nvidia":
            if not getattr(settings, "NVIDIA_API_KEY", "").strip():
                raise GeminiConfigurationError("NVIDIA_API_KEY is required")
            self._client = None
            return

        api_key = settings.GEMINI_API_KEY.strip()
        if not api_key:
            raise GeminiConfigurationError("GEMINI_API_KEY is required")
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=None),
        )

    def evaluate_images(self, prompt: str, image_paths: list[Path]) -> str:
        """Return stripped structured criticism for local images."""

        if not prompt.strip() or not image_paths:
            raise ValueError("prompt and image paths are required")

        if self.PROVIDER == "nvidia":
            return self._evaluate_nvidia(prompt, image_paths)

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

    def _evaluate_nvidia(self, prompt: str, image_paths: list[Path]) -> str:
        """Evaluate images through NVIDIA's OpenAI-compatible vision API."""

        image_parts = []
        for path in image_paths:
            mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}"
                    },
                }
            )

        base_url = getattr(
            settings,
            "NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        ).rstrip("/")
        payload = {
            "model": self.MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, *image_parts],
                }
            ],
            "temperature": 0.1,
            "max_tokens": getattr(settings, "NVIDIA_MAX_TOKENS", 16384),
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=None,
            )
            response.raise_for_status()
            response_payload = response.json()
        except httpx.TimeoutException as exc:
            raise GeminiTimeoutError("NVIDIA multimodal request timed out") from exc
        except httpx.NetworkError as exc:
            raise GeminiNetworkError(
                "NVIDIA multimodal network request failed"
            ) from exc
        except httpx.HTTPError as exc:
            raise GeminiAPIError(f"NVIDIA multimodal API request failed: {exc}") from exc
        except ValueError as exc:
            raise GeminiAPIError(
                "NVIDIA multimodal response was not valid JSON"
            ) from exc

        response_text = _extract_nvidia_response_text(response_payload)
        if not response_text.strip():
            raise GeminiEmptyResponseError("NVIDIA multimodal response was empty")
        return response_text.strip()
