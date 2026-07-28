"""Versioned domain contracts for the semantic V2 video pipeline."""

from app.domain.assets import (
    AssetKind,
    AssetQuery,
    AssetSource,
    ResolvedAssetSet,
    ResolvedSemanticAsset,
)
from app.domain.camera import CameraCue, CameraOperation, CameraPlan
from app.domain.generation import (
    ArtifactEnvelope,
    AudienceLevel,
    AudienceProfile,
    GenerationRequest,
    GenerationResult,
    OutputProfile,
    PipelineVersion,
)
from app.domain.layout import (
    ConstraintStrength,
    ConstraintType,
    LaidOutNode,
    LayoutBox,
    LayoutConstraint,
    LayoutPlan,
    Viewport,
)
from app.domain.lesson import (
    ConceptEdge,
    ConceptGraph,
    ConceptNode,
    ConceptRelation,
    LessonPlan,
)
from app.domain.motion import MotionEvent, MotionPlan
from app.domain.narration import (
    AlignedAudio,
    NarrationPhrase,
    NarrationPlan,
    PhraseTiming,
)
from app.domain.operations import OperationType, VisualOperation
from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityFinding,
    QualityReport,
)
from app.domain.storyboard import (
    AttentionCue,
    CameraIntent,
    Storyboard,
    VisualBeat,
    VisualObjectSpec,
)
from app.domain.visual_document import (
    ObjectLifecycle,
    ObjectState,
    VisualDocument,
    VisualState,
)

__all__ = [
    "AlignedAudio",
    "ArtifactEnvelope",
    "AssetKind",
    "AssetQuery",
    "AssetSource",
    "AttentionCue",
    "AudienceLevel",
    "AudienceProfile",
    "CameraCue",
    "CameraIntent",
    "CameraOperation",
    "CameraPlan",
    "ConceptEdge",
    "ConceptGraph",
    "ConceptNode",
    "ConceptRelation",
    "ConstraintStrength",
    "ConstraintType",
    "EvaluationDecision",
    "FindingSeverity",
    "GenerationRequest",
    "GenerationResult",
    "LaidOutNode",
    "LayoutBox",
    "LayoutConstraint",
    "LayoutPlan",
    "LessonPlan",
    "MotionEvent",
    "MotionPlan",
    "NarrationPhrase",
    "NarrationPlan",
    "ObjectLifecycle",
    "ObjectState",
    "OperationType",
    "OutputProfile",
    "PhraseTiming",
    "PipelineVersion",
    "QualityFinding",
    "QualityReport",
    "ResolvedAssetSet",
    "ResolvedSemanticAsset",
    "Storyboard",
    "Viewport",
    "VisualBeat",
    "VisualDocument",
    "VisualObjectSpec",
    "VisualOperation",
    "VisualState",
]
