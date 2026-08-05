"""Stage-local quality repair contracts."""

from enum import Enum
from hashlib import sha256
import json
from typing import Literal

from pydantic import Field, model_validator

from app.domain.quality import QualityFinding
from app.models.base import BaseModel, NonEmptyString


class RepairStage(str, Enum):
    """Pipeline stage that owns a repair and its downstream invalidation."""

    STORYBOARD = "storyboard"
    NARRATION = "narration"
    ASSETS = "assets"
    AUDIO = "audio"
    STATE = "state"
    LAYOUT = "layout"
    MOTION = "motion"
    CAMERA = "camera"
    RENDERER = "renderer"
    COMPOSER = "composer"


class RepairPlan(BaseModel):
    """Bounded repair request derived from one quality report."""

    schema_version: Literal["1.0"] = "1.0"
    owner_stage: RepairStage
    invalidated_stages: list[RepairStage] = Field(min_length=1)
    finding_codes: list[NonEmptyString] = Field(min_length=1)
    findings: list[QualityFinding] = Field(min_length=1)
    beat_ids: list[NonEmptyString] = Field(default_factory=list)
    object_ids: list[NonEmptyString] = Field(default_factory=list)
    frame_numbers: list[int] = Field(default_factory=list)
    allowed_patch_paths: list[NonEmptyString] = Field(default_factory=list)
    fingerprint: NonEmptyString

    @model_validator(mode="after")
    def owner_is_invalidated(self) -> "RepairPlan":
        if self.owner_stage not in self.invalidated_stages:
            raise ValueError("repair owner stage must be invalidated")
        return self

    @classmethod
    def fingerprint_for(
        cls,
        owner_stage: RepairStage,
        findings: list[QualityFinding],
    ) -> str:
        """Hash only stable evidence so repeated no-progress repairs match."""

        payload = {
            "owner_stage": owner_stage.value,
            "findings": [
                {
                    "code": item.code,
                    "beat_id": item.beat_id,
                    "object_ids": sorted(item.object_ids),
                    "frame_numbers": sorted(item.frame_numbers),
                    "measured_value": item.measured_value,
                    "required_value": item.required_value,
                }
                for item in sorted(findings, key=lambda value: value.code)
            ],
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
