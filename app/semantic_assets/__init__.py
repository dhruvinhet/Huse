"""Semantic asset catalog, resolution, and line-art fallback."""

from app.semantic_assets.catalog import AssetCatalog, CatalogAsset
from app.semantic_assets.resolver import CatalogSemanticAssetResolver

__all__ = ["AssetCatalog", "CatalogAsset", "CatalogSemanticAssetResolver"]
