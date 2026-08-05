"""Deterministic and multimodal V2 quality evaluation."""

from app.quality.deterministic import DeterministicQualityEvaluator
from app.quality.composite import CompositeQualityEvaluator
from app.quality.multimodal import (
    MultimodalFrameEvaluator,
    VisionLanguageClient,
)
from app.quality.policy import QualityPolicy, QualityReviewPolicy
from app.quality.educational import EducationalQualityEvaluator
from app.quality.visual import VisualQualityEvaluator
from app.quality.rendered import RenderedFrameQualityEvaluator
from app.quality.repair import QualityRepairPlanner

__all__ = [
    "DeterministicQualityEvaluator",
    "CompositeQualityEvaluator",
    "GeminiStoryboardCritic",
    "GeminiVisionClient",
    "MultimodalFrameEvaluator",
    "QualityPolicy",
    "QualityReviewPolicy",
    "VisionLanguageClient",
    "EducationalQualityEvaluator",
    "VisualQualityEvaluator",
    "RenderedFrameQualityEvaluator",
    "QualityRepairPlanner",
]


def __getattr__(name: str) -> object:
    """Load provider-backed critics only when explicitly enabled."""

    if name == "GeminiVisionClient":
        from app.quality.gemini_vision import GeminiVisionClient

        return GeminiVisionClient
    if name == "GeminiStoryboardCritic":
        from app.quality.storyboard_critic import GeminiStoryboardCritic

        return GeminiStoryboardCritic
    raise AttributeError(name)
