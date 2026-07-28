"""Quality findings, scores, and repair decisions."""

from enum import Enum
from typing import Literal

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString


class FindingSeverity(str, Enum):
    """Severity assigned to one quality finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class EvaluationDecision(str, Enum):
    """Terminal or repair decision returned by an evaluator."""

    PASS = "pass"
    REPAIR = "repair"
    FAIL = "fail"


class QualityFinding(BaseModel):
    """Describe one actionable quality defect."""

    code: NonEmptyString
    severity: FindingSeverity
    artifact_id: NonEmptyString
    message: NonEmptyString
    repair_target: str | None = None


class QualityReport(BaseModel):
    """Represent normalized quality scores and the gate decision."""

    schema_version: Literal["2.0"] = "2.0"
    overall_score: float = Field(ge=0, le=1)
    scores: dict[NonEmptyString, float]
    findings: list[QualityFinding] = Field(default_factory=list)
    decision: EvaluationDecision

