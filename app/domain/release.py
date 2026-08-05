"""Auditable release-gate and waiver contracts."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.benchmarking.models import BenchmarkComparison
from app.models.base import BaseModel, NonEmptyString


class ReleaseWaiver(BaseModel):
    """Temporarily authorize one failed deterministic gate."""

    gate_id: NonEmptyString
    owner: NonEmptyString
    reason: NonEmptyString
    expires_at: datetime


class ReleaseGateResult(BaseModel):
    """One measured release requirement and its disposition."""

    gate_id: NonEmptyString
    status: Literal["passed", "failed", "waived", "reported"]
    measured_value: float | None = None
    required_value: float | None = None
    message: NonEmptyString
    waiver: ReleaseWaiver | None = None


class ReleaseReport(BaseModel):
    """Machine-readable release decision with benchmark evidence."""

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    suite_id: NonEmptyString
    passed: bool
    gates: list[ReleaseGateResult] = Field(min_length=1)
    comparison: BenchmarkComparison | None = None
    artifact_paths: dict[NonEmptyString, NonEmptyString] = Field(
        default_factory=dict
    )
