"""Small live smoke test for the configured AI provider.

Examples:
    python check_ai_api.py
    python check_ai_api.py "binary search trees"
    python check_ai_api.py "binary search trees" --list-models
"""

import sys

import httpx

from app.config.settings import settings
from app.services.gemini_client import GeminiClient, GeminiClientError


def list_nvidia_models() -> int:
    """Print model IDs returned by NVIDIA for the configured API key."""

    base_url = settings.NVIDIA_BASE_URL.rstrip("/")
    try:
        response = httpx.get(
            f"{base_url}/models",
            headers={
                "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
                "Accept": "application/json",
            },
            timeout=None,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Could not list NVIDIA models: {exc}", file=sys.stderr)
        return 1

    models = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(models, list):
        print("NVIDIA returned an unexpected model-list response.", file=sys.stderr)
        return 1

    print("Models returned by NVIDIA for this API key:")
    model_ids = [
        item.get("id")
        for item in models
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not model_ids:
        print("(The API returned no model IDs.)")
    else:
        for model_id in model_ids:
            print(f"- {model_id}")
    return 0


def main() -> int:
    """Generate readable content and optionally list NVIDIA models."""

    arguments = sys.argv[1:]
    should_list_models = "--list-models" in arguments
    topic_parts = [argument for argument in arguments if argument != "--list-models"]
    topic = " ".join(topic_parts).strip() or "binary search trees"

    try:
        client = GeminiClient()
        print(f"Provider: {client.PROVIDER}")
        print(f"Model: {client.MODEL_NAME}")
        print(f"Generating content about: {topic}")

        response = client.generate_text(
            (
                f"Explain {topic} for a beginner in three concise paragraphs. "
                "Use plain text only. Do not return JSON, markdown code fences, "
                "or analysis. Include one simple example."
            ),
            temperature=0.2,
        )

        if not response.strip():
            print("ERROR: The API returned an empty response.")
            return 1

        print("Content output:")
        print(response)
        print("API call succeeded.")

        if should_list_models:
            if client.PROVIDER != "nvidia":
                print(
                    "Model listing was skipped because AI_PROVIDER is not NVIDIA."
                )
                return 0
            return list_nvidia_models()
        return 0
    except (GeminiClientError, ValueError) as exc:
        print(f"API test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
