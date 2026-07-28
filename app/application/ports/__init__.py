"""Replaceable subsystem interfaces used by pipeline V2."""

from app.application.ports.assets import SemanticAssetResolver
from app.application.ports.audio import PhraseAligner, SpeechSynthesizer
from app.application.ports.knowledge import TemplateLibrary, VisualKnowledgeBase
from app.application.ports.layout import ConstraintLayoutEngine, VisualStateEngine
from app.application.ports.motion import AnimationPlanner, VirtualCameraPlanner
from app.application.ports.planning import (
    LessonPlanner,
    NarrationWriter,
    StoryboardPlanner,
)
from app.application.ports.quality import QualityEvaluator
from app.application.ports.rendering import RenderEngine, VideoCompositionEngine

__all__ = [
    "AnimationPlanner",
    "ConstraintLayoutEngine",
    "LessonPlanner",
    "NarrationWriter",
    "PhraseAligner",
    "QualityEvaluator",
    "RenderEngine",
    "SemanticAssetResolver",
    "SpeechSynthesizer",
    "StoryboardPlanner",
    "TemplateLibrary",
    "VideoCompositionEngine",
    "VirtualCameraPlanner",
    "VisualKnowledgeBase",
    "VisualStateEngine",
]
