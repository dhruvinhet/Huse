"""Core adapters with optional media/provider modules loaded on demand."""

from importlib import import_module

_LAZY = {
    "AnimationTimelineBuilder": ("app.core.animation_timeline_builder", "AnimationTimelineBuilder"),
    "AudioGenerationError": ("app.core.audio_manager", "AudioGenerationError"),
    "AudioManager": ("app.core.audio_manager", "AudioManager"),
    "AssetManager": ("app.core.asset_manager", "AssetManager"),
    "AssetPlanner": ("app.core.asset_planner", "AssetPlanner"),
    "FrameRenderer": ("app.core.frame_renderer", "FrameRenderer"),
    "PipelineRunner": ("app.core.pipeline_runner", "PipelineRunner"),
    "SceneGraphBuilder": ("app.core.scene_graph_builder", "SceneGraphBuilder"),
    "SceneRenderer": ("app.core.scene_renderer", "SceneRenderer"),
    "ScriptGeminiError": ("app.core.script_generator", "ScriptGeminiError"),
    "ScriptGenerator": ("app.core.script_generator", "ScriptGenerator"),
    "ScriptGeneratorError": ("app.core.script_generator", "ScriptGeneratorError"),
    "ScriptJSONError": ("app.core.script_generator", "ScriptJSONError"),
    "ScriptValidationError": ("app.core.script_generator", "ScriptValidationError"),
    "TimelineSynchronizer": ("app.core.timeline_synchronizer", "TimelineSynchronizer"),
    "FFmpegNotFoundError": ("app.core.video_composer", "FFmpegNotFoundError"),
    "VideoComposer": ("app.core.video_composer", "VideoComposer"),
    "VideoCompositionError": ("app.core.video_composer", "VideoCompositionError"),
}


def __getattr__(name: str) -> object:
    if name not in _LAZY:
        raise AttributeError(name)
    module_name, attribute = _LAZY[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_LAZY)
