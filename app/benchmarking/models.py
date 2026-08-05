"""Versioned models for repeatable Huse benchmark runs."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from app.domain.generation import AudienceLevel
from app.domain.lesson import ConceptRelation
from app.domain.strategy import ParameterProvenance
from app.models.base import BaseModel, NonEmptyString


class BenchmarkMode(str, Enum):
    """Supported benchmark execution tiers."""

    OFFLINE = "offline"
    PROVIDER = "provider"


class BenchmarkCase(BaseModel):
    """One versioned educational prompt and its structural expectations."""

    case_id: NonEmptyString
    category: NonEmptyString
    topic: NonEmptyString
    learning_goal: NonEmptyString
    expected_concepts: list[NonEmptyString] = Field(min_length=2)
    expected_actions: list[NonEmptyString] = Field(default_factory=list)
    relation: ConceptRelation = ConceptRelation.FLOWS_TO
    audience_level: AudienceLevel = AudienceLevel.INTERMEDIATE
    target_duration: float = Field(default=120.0, gt=0)
    seed: int = Field(default=0, ge=0)


class BenchmarkSuite(BaseModel):
    """A stable collection of benchmark cases."""

    schema_version: Literal["1.0"] = "1.0"
    suite_id: NonEmptyString
    description: NonEmptyString
    cases: list[BenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self) -> "BenchmarkSuite":
        """Reject ambiguous comparison keys."""

        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark case IDs must be unique")
        return self


class BenchmarkConfig(BaseModel):
    """Record every setting required to interpret benchmark output."""

    schema_version: Literal["1.0"] = "1.0"
    mode: BenchmarkMode = BenchmarkMode.OFFLINE
    suite_version: NonEmptyString = "1.0"
    provider_id: str | None = None
    model_id: str | None = None
    seed: int = Field(default=0, ge=0)
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    fps: int = Field(default=30, gt=0)

    @model_validator(mode="after")
    def provider_metadata(self) -> "BenchmarkConfig":
        """Require explicit mutable-provider identity for live runs."""

        if self.mode is BenchmarkMode.PROVIDER and (
            not self.provider_id or not self.model_id
        ):
            raise ValueError(
                "provider benchmark mode requires provider_id and model_id"
            )
        return self


class BenchmarkMetrics(BaseModel):
    """Comparable measurements collected from one completed case."""

    topic_specificity: float | None = Field(default=None, ge=0, le=1)
    action_coverage: float | None = Field(default=None, ge=0, le=1)
    state_delta_coverage: float | None = Field(default=None, ge=0, le=1)
    composition_changes_per_minute: float | None = Field(default=None, ge=0)
    camera_change_coverage: float | None = Field(default=None, ge=0, le=1)
    clipping_violations: int = Field(default=0, ge=0)
    readability_violations: int = Field(default=0, ge=0)
    audio_timing_coverage: float | None = Field(default=None, ge=0, le=1)
    provider_timing_coverage: float | None = Field(default=None, ge=0, le=1)
    estimated_timing_coverage: float | None = Field(default=None, ge=0, le=1)
    mean_word_timing_confidence: float | None = Field(default=None, ge=0, le=1)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    total_beats: int = Field(default=0, ge=0)
    total_objects: int = Field(default=0, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    frame_cache_hit_rate: float | None = Field(default=None, ge=0, le=1)
    layer_cache_hit_rate: float | None = Field(default=None, ge=0, le=1)
    keyframe_count: int | None = Field(default=None, ge=0)
    encoded_frames_per_second: float | None = Field(default=None, ge=0)
    retained_frame_count: int | None = Field(default=None, ge=0)


class BenchmarkCaseResult(BaseModel):
    """Auditable result for one benchmark case."""

    schema_version: Literal["1.0"] = "1.0"
    case_id: NonEmptyString
    status: Literal["completed", "failed"]
    wall_time_seconds: float = Field(ge=0)
    stage_timings_seconds: dict[NonEmptyString, float] = Field(default_factory=dict)
    metrics: BenchmarkMetrics = Field(default_factory=BenchmarkMetrics)
    selected_templates: list[NonEmptyString] = Field(default_factory=list)
    selected_operators: list[NonEmptyString] = Field(default_factory=list)
    template_match_confidence: float | None = Field(default=None, ge=0, le=1)
    template_capability_evidence: list[NonEmptyString] = Field(default_factory=list)
    template_parameter_provenance: dict[
        NonEmptyString, ParameterProvenance
    ] = Field(default_factory=dict)
    template_default_usage: list[NonEmptyString] = Field(default_factory=list)
    repair_attempts: int = Field(default=0, ge=0)
    quality_findings: list[NonEmptyString] = Field(default_factory=list)
    artifact_paths: dict[NonEmptyString, NonEmptyString] = Field(default_factory=dict)
    error: str | None = None


class BenchmarkReport(BaseModel):
    """Complete machine-readable output for a benchmark invocation."""

    schema_version: Literal["1.0"] = "1.0"
    suite_id: NonEmptyString
    config: BenchmarkConfig
    started_at: datetime
    finished_at: datetime
    environment: dict[NonEmptyString, NonEmptyString]
    cases: list[BenchmarkCaseResult]

    @property
    def reliability(self) -> float:
        """Return the fraction of cases that completed without an exception."""

        if not self.cases:
            return 0.0
        return sum(case.status == "completed" for case in self.cases) / len(
            self.cases
        )


class MetricDelta(BaseModel):
    """One comparable metric change between reports."""

    case_id: NonEmptyString
    metric: NonEmptyString
    baseline: float
    current: float
    delta: float


class BenchmarkComparison(BaseModel):
    """Difference summary for two compatible reports."""

    schema_version: Literal["1.0"] = "1.0"
    suite_id: NonEmptyString
    reliability_delta: float
    metric_deltas: list[MetricDelta] = Field(default_factory=list)
    added_cases: list[NonEmptyString] = Field(default_factory=list)
    missing_cases: list[NonEmptyString] = Field(default_factory=list)
