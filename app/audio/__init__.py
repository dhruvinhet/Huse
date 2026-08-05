"""V2 alignment plus an optional TTS adapter."""

from importlib import import_module

from app.audio.alignment import ScenePhraseAligner


def __getattr__(name: str) -> object:
    if name != "StoryboardSpeechSynthesizer":
        raise AttributeError(name)
    value = getattr(
        import_module("app.audio.adapters"), "StoryboardSpeechSynthesizer"
    )
    globals()[name] = value
    return value


__all__ = ["ScenePhraseAligner", "StoryboardSpeechSynthesizer"]
