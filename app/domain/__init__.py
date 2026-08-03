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
from app.domain.pedagogy import PedagogyMode, PedagogyPlan, PedagogyShot
from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityFinding,
    QualityReport,
)
from app.domain.storyboard import (
    AttentionCue,
    CameraIntent,
    ShotPlan,
    Storyboard,
    VisualBeat,
    VisualObjectSpec,
)
from app.domain.strategy import CompiledTemplateProgram, TemplateMatch, VisualStrategy
from app.domain.visual_document import (
    ObjectLifecycle,
    ObjectState,
    VisualDocument,
    VisualState,
)
from app.domain.visual_intent import RendererOperator, ShotSpec, VisualIntent

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
    "ShotPlan",
    "ConceptEdge",
    "ConceptGraph",
    "ConceptNode",
    "ConceptRelation",
    "CompiledTemplateProgram",
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
    "PedagogyMode",
    "PedagogyPlan",
    "PedagogyShot",
    "PhraseTiming",
    "PipelineVersion",
    "QualityFinding",
    "QualityReport",
    "ResolvedAssetSet",
    "ResolvedSemanticAsset",
    "RendererOperator",
    "ShotSpec",
    "Storyboard",
    "TemplateMatch",
    "Viewport",
    "VisualBeat",
    "VisualStrategy",
    "VisualDocument",
    "VisualIntent",
    "VisualObjectSpec",
    "VisualOperation",
    "VisualState",
]
