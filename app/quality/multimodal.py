"""Injectable multimodal evaluation of sampled rendered frames."""

import json
from pathlib import Path
from typing import Protocol

from app.domain.quality import QualityReport
from app.domain.narration import NarrationPlan
from app.domain.storyboard import Storyboard


class VisionLanguageClient(Protocol):
    """Provider-neutral client capable of evaluating local images."""

    def evaluate_images(self, prompt: str, image_paths: list[Path]) -> str:
        """Return only a QualityReport JSON object."""


class MultimodalFrameEvaluator:
    """Compare frame samples with narration and storyboard semantics."""

    def __init__(self, client: VisionLanguageClient) -> None:
        """Store an explicitly configured multimodal client."""

        self._client = client

    def evaluate_frames(
        self,
        artifact_id: str,
        sample_paths: list[Path],
        storyboard: Storyboard,
        narration: NarrationPlan,
    ) -> QualityReport:
        """Evaluate semantic alignment, correctness, focus, and readability."""

        if not sample_paths:
            raise ValueError("multimodal evaluation requires sampled frames")
        missing = [path for path in sample_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"sampled frames are missing: {missing}")
        schema = QualityReport.model_json_schema()
        prompt = (
            "Evaluate these educational whiteboard frames against the accepted "
            "storyboard and narration. Score semantic_coverage, alignment, "
            "readability, layout, animation_density, and diagram_correctness. "
            "Identify missing concepts, misleading relationships, missing labels, "
            "poor emphasis, or excessive cognitive load. Return only JSON matching "
            f"this schema: {json.dumps(schema)}\n"
            f"Artifact ID: {artifact_id}\n"
            f"Storyboard: {storyboard.model_dump_json()}\n"
            f"Narration: {narration.model_dump_json()}"
        )
        response = self._client.evaluate_images(prompt, sample_paths)
        return QualityReport.model_validate_json(response)
