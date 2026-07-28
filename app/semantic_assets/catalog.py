"""In-memory semantic catalog with explicit licensing metadata."""

from dataclasses import dataclass
from pathlib import Path

from app.domain.assets import AssetKind, AssetQuery


@dataclass(frozen=True, slots=True)
class CatalogAsset:
    """Describe one approved local semantic asset."""

    asset_id: str
    concepts: frozenset[str]
    kind: AssetKind
    style_id: str
    path: Path
    mime_type: str
    license_id: str
    editable: bool = True


class AssetCatalog:
    """Register and rank allowlisted local assets."""

    def __init__(self, assets: list[CatalogAsset] | None = None) -> None:
        """Create a catalog from approved metadata records."""

        self._assets: dict[str, CatalogAsset] = {}
        for asset in assets or []:
            self.register(asset)

    def register(self, asset: CatalogAsset) -> None:
        """Register a valid, licensed local asset."""

        if not asset.asset_id.strip() or not asset.concepts:
            raise ValueError("catalog assets require an ID and concepts")
        if not asset.license_id.strip():
            raise ValueError("catalog assets require explicit license metadata")
        if asset.asset_id in self._assets:
            raise ValueError(f"catalog asset already registered: {asset.asset_id}")
        self._assets[asset.asset_id] = asset

    def find(self, query: AssetQuery) -> CatalogAsset | None:
        """Return the best exact semantic/style match with stable tie-breaking."""

        terms = set(query.concept.lower().replace("-", " ").split())
        terms.update(item.lower() for item in query.required_semantics)
        candidates: list[tuple[int, CatalogAsset]] = []
        for asset in self._assets.values():
            if asset.kind is not query.asset_kind:
                continue
            if asset.style_id not in {query.style_id, "*"}:
                continue
            score = len(terms.intersection(asset.concepts))
            if score:
                candidates.append((score, asset))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (-item[0], item[1].asset_id))[0][1]

    def records(self) -> list[CatalogAsset]:
        """Return deterministic catalog records."""

        return [self._assets[key] for key in sorted(self._assets)]
