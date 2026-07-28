"""Models for assets resolved and prepared for a future renderer."""

from pydantic import Field

from app.models.assets import AssetType
from app.models.base import BaseModel, NonEmptyString


class ResolvedAsset(BaseModel):
    """Describe the preparation status and path of one planned asset."""

    asset_id: NonEmptyString
    asset_type: AssetType
    resolved_path: str
    ready: bool
    generated: bool


class SceneResolvedAssets(BaseModel):
    """Group resolved assets belonging to one scene."""

    scene_number: int = Field(ge=1)
    assets: list[ResolvedAsset]


class ResolvedAssetPlan(BaseModel):
    """Represent all assets prepared from an asset execution plan."""

    scenes: list[SceneResolvedAssets]
