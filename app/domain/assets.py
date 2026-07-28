"""Semantic asset query and resolution contracts."""

from enum import Enum
from typing import Literal, Self

from pydantic import Field, model_validator

from app.models.base import BaseModel, NonEmptyString


class AssetKind(str, Enum):
    """Kinds of semantic assets available to the V2 resolver."""

    ICON = "icon"
    SVG = "svg"
    LINE_ART = "line_art"
    TEMPLATE = "template"
    IMAGE = "image"


class AssetSource(str, Enum):
    """Provenance of a resolved semantic asset."""

    TEMPLATE = "template"
    CATALOG = "catalog"
    GENERATED = "generated"
    LEGACY = "legacy"


class AssetQuery(BaseModel):
    """Describe an asset semantically without naming a file."""

    concept: NonEmptyString
    asset_kind: AssetKind
    style_id: NonEmptyString
    required_semantics: list[NonEmptyString] = Field(default_factory=list)


class ResolvedSemanticAsset(BaseModel):
    """Describe a ready asset with cache and license provenance."""

    asset_id: NonEmptyString
    query_digest: NonEmptyString
    source: AssetSource
    path: NonEmptyString
    mime_type: NonEmptyString
    license_id: str | None = None
    content_hash: NonEmptyString
    editable: bool
    ready: bool


class ResolvedAssetSet(BaseModel):
    """Group unique resolved assets for a storyboard."""

    schema_version: Literal["2.0"] = "2.0"
    assets: list[ResolvedSemanticAsset] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_assets(self) -> Self:
        """Require unique IDs and hashes within the asset set."""

        asset_ids = [asset.asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("resolved asset IDs must be unique")
        return self
