"""Bounded AI planners, loaded only when a provider adapter is requested."""

from importlib import import_module

from app.agents.base import StructuredAgentError

_LAZY = {
    "GeminiLessonPlanner": ("app.agents.lesson_planner", "GeminiLessonPlanner"),
    "GeminiNarrationWriter": ("app.agents.narration_writer", "GeminiNarrationWriter"),
    "GeminiStoryboardPlanner": ("app.agents.storyboard_planner", "GeminiStoryboardPlanner"),
}


def __getattr__(name: str) -> object:
    if name not in _LAZY:
        raise AttributeError(name)
    module_name, attribute = _LAZY[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [*_LAZY, "StructuredAgentError"]
