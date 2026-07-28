"""Models for scene asset requirements and execution plans."""

from enum import Enum

from pydantic import Field

from app.models.base import BaseModel, NonEmptyString


class AssetType(str, Enum):
    """Supported categories of visual assets."""

    TEXT = "text"
    ICON = "icon"
    SVG = "svg"
    IMAGE = "image"


class AssetRequirement(BaseModel):
    """Describe one asset required to realize a visual instruction."""

    asset_id: NonEmptyString
    asset_type: AssetType
    content: NonEmptyString
    exists_locally: bool
    local_path: str | None = None
    needs_generation: bool


class SceneAssets(BaseModel):
    """Group the asset requirements belonging to one script scene."""

    scene_number: int = Field(ge=1)
    assets: list[AssetRequirement]


class AssetPlan(BaseModel):
    """Represent the complete asset execution plan for a script."""

    scenes: list[SceneAssets]
