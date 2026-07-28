"""Versioned JSON Schema registry for persisted V2 artifacts."""

import json
from pathlib import Path

from pydantic import BaseModel as PydanticModel

from app.domain.assets import ResolvedAssetSet
from app.domain.camera import CameraPlan
from app.domain.generation import GenerationRequest, GenerationResult
from app.domain.layout import LayoutPlan
from app.domain.lesson import ConceptGraph, LessonPlan
from app.domain.motion import MotionPlan
from app.domain.narration import AlignedAudio, NarrationPlan
from app.domain.quality import QualityReport
from app.domain.rendering import FrameSequence, RenderJob, VideoArtifact
from app.domain.storyboard import Storyboard
from app.domain.visual_document import VisualDocument


SCHEMA_MODELS: dict[str, type[PydanticModel]] = {
    "aligned-audio-2.0": AlignedAudio,
    "camera-plan-2.0": CameraPlan,
    "concept-graph-2.0": ConceptGraph,
    "frame-sequence-2.0": FrameSequence,
    "generation-request-2.0": GenerationRequest,
    "generation-result-2.0": GenerationResult,
    "layout-plan-2.0": LayoutPlan,
    "lesson-plan-2.0": LessonPlan,
    "motion-plan-2.0": MotionPlan,
    "narration-plan-2.0": NarrationPlan,
    "quality-report-2.0": QualityReport,
    "render-job-2.0": RenderJob,
    "resolved-assets-2.0": ResolvedAssetSet,
    "storyboard-2.0": Storyboard,
    "video-artifact-2.0": VideoArtifact,
    "visual-document-2.0": VisualDocument,
}


def schema(name: str) -> dict[str, object]:
    """Return a registered JSON Schema by stable name."""

    try:
        model = SCHEMA_MODELS[name]
    except KeyError as exc:
        raise KeyError(f"unknown V2 schema: {name}") from exc
    value = model.model_json_schema()
    value["$id"] = f"whiteboard/{name}"
    return value


def export_schemas(output_dir: Path) -> list[Path]:
    """Write every registered schema deterministically for documentation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in sorted(SCHEMA_MODELS):
        path = output_dir / f"{name}.json"
        path.write_text(
            json.dumps(schema(name), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paths.append(path)
    return paths
