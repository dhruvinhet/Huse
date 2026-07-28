"""List Gemini models available to the locally configured API key."""

from google import genai

from app.config.settings import settings

def main() -> None:
    """Print models visible to the locally configured Gemini account."""

    if not settings.GEMINI_API_KEY.strip():
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and configure it."
        )

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    print("=" * 100)
    print("Models available for this API key")
    print("=" * 100)

    for model in client.models.list():
        print(f"Model: {model.name}")
        if hasattr(model, "display_name"):
            print(f"Display Name : {model.display_name}")
        if hasattr(model, "supported_actions"):
            print("Supported Actions:")
            for action in model.supported_actions:
                print(f"   - {action}")
        print("-" * 100)


if __name__ == "__main__":
    main()
