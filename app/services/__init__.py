"""Optional provider clients, imported only when explicitly requested."""

from importlib import import_module

_NAMES = {
    "GeminiAPIError", "GeminiClient", "GeminiClientError",
    "GeminiConfigurationError", "GeminiEmptyResponseError",
    "GeminiNetworkError", "GeminiTimeoutError",
}


def __getattr__(name: str) -> object:
    if name not in _NAMES:
        raise AttributeError(name)
    value = getattr(import_module("app.services.gemini_client"), name)
    globals()[name] = value
    return value


__all__ = sorted(_NAMES)
