"""Backward-compatible facade selecting pipeline V1 or V2."""

from uuid import uuid4

from app.application.orchestrators.pipeline_v2 import V2PipelineRunner
from app.core.pipeline_runner import PipelineRunner
from app.config.settings import settings
from app.domain.generation import (
    AudienceLevel,
    AudienceProfile,
    GenerationRequest,
    GenerationResult,
    PipelineVersion,
)


class VideoGenerationService:
    """Expose a stable topic-to-video use case across pipeline versions."""

    def __init__(
        self,
        v1: PipelineRunner | None = None,
        v2: V2PipelineRunner | None = None,
    ) -> None:
        """Allow either version to be replaced for tests or deployment."""

        self._v1 = v1
        self._v2 = v2

    def run(
        self,
        topic: str,
        output_dir: str = "outputs",
        pipeline_version: PipelineVersion | None = None,
        audience_level: AudienceLevel = AudienceLevel.INTERMEDIATE,
        learning_goal: str | None = None,
        target_duration: float = 60.0,
    ) -> str | GenerationResult:
        """Run the requested version while preserving the V1 return contract."""

        normalized_topic = topic.strip()
        if not normalized_topic:
            raise ValueError("topic cannot be empty")
        selected_version = pipeline_version or PipelineVersion(
            settings.PIPELINE_VERSION
        )
        if selected_version is PipelineVersion.V1:
            if self._v1 is None:
                self._v1 = PipelineRunner()
            return self._v1.run(normalized_topic, output_dir)
        if self._v2 is None:
            self._v2 = V2PipelineRunner()
        request = GenerationRequest(
            run_id=f"v2_{uuid4().hex}",
            topic=normalized_topic,
            target_duration=target_duration,
            audience=AudienceProfile(
                level=audience_level,
                learning_goal=(
                    learning_goal.strip()
                    if learning_goal and learning_goal.strip()
                    else f"Understand {normalized_topic}"
                ),
            ),
        )
        return self._v2.run(request, output_dir)
