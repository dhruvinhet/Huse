"""Unit tests for the reusable Gemini API client."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from app.services import gemini_client
from app.services.gemini_client import (
    GeminiAPIError,
    GeminiClient,
    GeminiConfigurationError,
    GeminiEmptyResponseError,
    GeminiNetworkError,
    GeminiTimeoutError,
)


@pytest.fixture
def configured_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide nonsecret test configuration to the client module."""

    test_settings = SimpleNamespace(
        GEMINI_API_KEY="test-api-key",
        GEMINI_TIMEOUT_SECONDS=30.0,
    )
    monkeypatch.setattr(gemini_client, "settings", test_settings)


def test_empty_prompt_raises_value_error(
    configured_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only prompts are rejected before an SDK request."""

    sdk_client = MagicMock()
    monkeypatch.setattr(
        gemini_client.genai,
        "Client",
        MagicMock(return_value=sdk_client),
    )
    client = GeminiClient()

    with pytest.raises(ValueError, match="prompt must not be empty"):
        client.generate_text("   ")

    sdk_client.models.generate_content.assert_not_called()


def test_missing_api_key_is_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing API key raises a configuration error before SDK creation."""

    test_settings = SimpleNamespace(
        GEMINI_API_KEY=" ",
        GEMINI_TIMEOUT_SECONDS=30.0,
    )
    sdk_constructor = MagicMock()
    monkeypatch.setattr(gemini_client, "settings", test_settings)
    monkeypatch.setattr(gemini_client.genai, "Client", sdk_constructor)

    with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY"):
        GeminiClient()

    sdk_constructor.assert_not_called()


def test_client_initializes_once(
    configured_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One GeminiClient instance reuses one initialized SDK client."""

    sdk_client = MagicMock()
    sdk_client.models.generate_content.return_value = SimpleNamespace(text="first")
    sdk_constructor = MagicMock(return_value=sdk_client)
    monkeypatch.setattr(gemini_client.genai, "Client", sdk_constructor)

    client = GeminiClient()
    client.generate_text("Request one")
    sdk_client.models.generate_content.return_value = SimpleNamespace(text="second")
    client.generate_text("Request two")

    sdk_constructor.assert_called_once()
    assert sdk_client.models.generate_content.call_count == 2


def test_client_supports_unlimited_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A None timeout disables the application-level request deadline."""

    test_settings = SimpleNamespace(
        GEMINI_API_KEY="test-api-key",
        GEMINI_TIMEOUT_SECONDS=None,
    )
    sdk_constructor = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(gemini_client, "settings", test_settings)
    monkeypatch.setattr(gemini_client.genai, "Client", sdk_constructor)

    GeminiClient()

    http_options = sdk_constructor.call_args.kwargs["http_options"]
    assert http_options.timeout is None


def test_generate_text_returns_stripped_response(
    configured_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client sends SDK generation config and returns only clean text."""

    sdk_client = MagicMock()
    sdk_client.models.generate_content.return_value = SimpleNamespace(
        text="  Generated response.\n"
    )
    monkeypatch.setattr(
        gemini_client.genai,
        "Client",
        MagicMock(return_value=sdk_client),
    )

    result = GeminiClient().generate_text("Test prompt", temperature=0.2)

    assert result == "Generated response."
    request = sdk_client.models.generate_content.call_args.kwargs
    assert request["model"] == "gemini-3.5-flash"
    assert request["contents"] == "Test prompt"
    assert request["config"].temperature == pytest.approx(0.2)


def test_timeout_is_translated(
    configured_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDK transport timeouts become meaningful client exceptions."""

    sdk_client = MagicMock()
    sdk_client.models.generate_content.side_effect = httpx.ReadTimeout(
        "request timed out"
    )
    monkeypatch.setattr(
        gemini_client.genai,
        "Client",
        MagicMock(return_value=sdk_client),
    )

    with pytest.raises(GeminiTimeoutError, match="timed out"):
        GeminiClient().generate_text("Test prompt")


def test_network_failure_is_translated(
    configured_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDK transport failures become meaningful client exceptions."""

    sdk_client = MagicMock()
    sdk_client.models.generate_content.side_effect = httpx.ConnectError(
        "connection failed"
    )
    monkeypatch.setattr(
        gemini_client.genai,
        "Client",
        MagicMock(return_value=sdk_client),
    )

    with pytest.raises(GeminiNetworkError, match="network failure"):
        GeminiClient().generate_text("Test prompt")


def test_api_error_is_translated(
    configured_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Official SDK API errors become meaningful client exceptions."""

    sdk_client = MagicMock()
    sdk_client.models.generate_content.side_effect = (
        gemini_client.genai_errors.ClientError(
            400,
            {
                "error": {
                    "message": "Invalid request",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )
    )
    monkeypatch.setattr(
        gemini_client.genai,
        "Client",
        MagicMock(return_value=sdk_client),
    )

    with pytest.raises(GeminiAPIError, match="Gemini API request failed"):
        GeminiClient().generate_text("Test prompt")


def test_empty_response_is_rejected(
    configured_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An SDK response without usable text raises a clear exception."""

    sdk_client = MagicMock()
    sdk_client.models.generate_content.return_value = SimpleNamespace(text="  ")
    monkeypatch.setattr(
        gemini_client.genai,
        "Client",
        MagicMock(return_value=sdk_client),
    )

    with pytest.raises(GeminiEmptyResponseError, match="empty text response"):
        GeminiClient().generate_text("Test prompt")
