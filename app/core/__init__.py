"""Core application components."""

from app.core.animation_timeline_builder import AnimationTimelineBuilder
from app.core.audio_manager import AudioGenerationError, AudioManager
from app.core.asset_planner import AssetPlanner
from app.core.asset_manager import AssetManager
from app.core.frame_renderer import FrameRenderer
from app.core.pipeline_runner import PipelineRunner
from app.core.scene_graph_builder import SceneGraphBuilder
from app.core.scene_renderer import SceneRenderer
from app.core.script_generator import (
    ScriptGeminiError,
    ScriptGenerator,
    ScriptGeneratorError,
    ScriptJSONError,
    ScriptValidationError,
)
from app.core.timeline_synchronizer import TimelineSynchronizer
from app.core.video_composer import (
    FFmpegNotFoundError,
    VideoComposer,
    VideoCompositionError,
)

__all__ = [
    "AnimationTimelineBuilder",
    "AudioGenerationError",
    "AudioManager",
    "AssetManager",
    "AssetPlanner",
    "FrameRenderer",
    "PipelineRunner",
    "SceneGraphBuilder",
    "SceneRenderer",
    "ScriptGeminiError",
    "ScriptGenerator",
    "ScriptGeneratorError",
    "ScriptJSONError",
    "ScriptValidationError",
    "TimelineSynchronizer",
    "FFmpegNotFoundError",
    "VideoComposer",
    "VideoCompositionError",
]
