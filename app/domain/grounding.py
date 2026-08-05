"""Factual-grounding reports shared by validators and the pipeline."""

from enum import Enum
from typing import Literal

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString


class GroundingDecision(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


class GroundingIssue(BaseModel):
    code: NonEmptyString
    severity: Literal["review", "error"]
    message: NonEmptyString
    claim_ids: list[NonEmptyString] = Field(default_factory=list)
    edge_ids: list[NonEmptyString] = Field(default_factory=list)
    source_ids: list[NonEmptyString] = Field(default_factory=list)


class GroundingReport(BaseModel):
    decision: GroundingDecision
    issues: list[GroundingIssue] = Field(default_factory=list)
    validated_claims: int = Field(default=0, ge=0)
    validated_relations: int = Field(default=0, ge=0)


class FactualGroundingError(ValueError):
    """Raised when supplied evidence contradicts or cannot support a lesson."""

