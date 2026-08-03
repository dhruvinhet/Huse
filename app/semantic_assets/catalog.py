"""Offline semantic asset catalog with normalized retrieval metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

from app.config.settings import PROJECT_ROOT
from app.domain.assets import AssetKind, AssetQuery


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_INDEX = Path("app/semantic_assets/offline/index.json")
FULL_INDEX = Path("app/semantic_assets/offline/full_index.json")


def normalize_tokens(value: str) -> frozenset[str]:
    """Return stable lowercase search tokens with conservative morphology."""

    tokens: set[str] = set()
    for raw in _TOKEN_PATTERN.findall(value.lower().replace("_", " ")):
        tokens.add(raw)
        if len(raw) > 4 and raw.endswith("ies"):
            tokens.add(f"{raw[:-3]}y")
        elif len(raw) > 4 and raw.endswith("ing"):
            stem = raw[:-3]
            tokens.add(stem)
            tokens.add(f"{stem}e")
        elif len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
            tokens.add(raw[:-1])
    return frozenset(tokens)


@dataclass(frozen=True, slots=True)
class CatalogAsset:
    """Describe one approved local semantic asset and its search vocabulary."""

    asset_id: str
    concepts: frozenset[str]
    kind: AssetKind
    style_id: str
    path: Path
    mime_type: str
    license_id: str
    editable: bool = True
    aliases: frozenset[str] = field(default_factory=frozenset)
    categories: frozenset[str] = field(default_factory=frozenset)
    tags: frozenset[str] = field(default_factory=frozenset)
    relations: frozenset[str] = field(default_factory=frozenset)
    composition_roles: frozenset[str] = field(default_factory=frozenset)
    source_collection: str = "internal"

    def exact_tokens(self) -> frozenset[str]:
        """Return high-confidence concept and alias tokens."""

        return normalize_tokens(" ".join((*self.concepts, *self.aliases)))

    def search_tokens(self) -> frozenset[str]:
        """Return all normalized retrieval terms including compositional tags."""

        return normalize_tokens(
            " ".join(
                (
                    *self.concepts,
                    *self.aliases,
                    *self.categories,
                    *self.tags,
                )
            )
        )


class AssetCatalog:
    """Load, validate, and rank the shipped local SVG vocabulary."""

    def __init__(
        self,
        assets: list[CatalogAsset] | None = None,
        index_path: Path | None = None,
    ) -> None:
        """Load the default offline pack unless explicit records are supplied."""

        self._assets: dict[str, CatalogAsset] = {}
        self._concept_tokens: dict[str, frozenset[str]] = {}
        self._alias_tokens: dict[str, frozenset[str]] = {}
        self._search_tokens: dict[str, frozenset[str]] = {}
        self._concept_phrases: dict[str, frozenset[str]] = {}
        self._alias_phrases: dict[str, frozenset[str]] = {}
        if assets is None:
            records = self._load_index(index_path or DEFAULT_INDEX)
            if index_path is None and (PROJECT_ROOT / FULL_INDEX).is_file():
                records.extend(self._load_index(FULL_INDEX))
        else:
            records = assets
        for asset in records:
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
        self._concept_tokens[asset.asset_id] = normalize_tokens(
            " ".join(asset.concepts)
        )
        self._alias_tokens[asset.asset_id] = normalize_tokens(
            " ".join(asset.aliases)
        )
        self._search_tokens[asset.asset_id] = asset.search_tokens()
        self._concept_phrases[asset.asset_id] = frozenset(
            item.lower().replace("-", " ").replace("_", " ")
            for item in asset.concepts
        )
        self._alias_phrases[asset.asset_id] = frozenset(
            item.lower().replace("-", " ").replace("_", " ")
            for item in asset.aliases
        )

    def find(self, query: AssetQuery) -> CatalogAsset | None:
        """Return a high-confidence direct match, leaving weak hits to composition."""

        ranked = self.rank(query, limit=1, exact_only=True)
        if not ranked:
            return None
        best = ranked[0]
        phrase = " ".join(_TOKEN_PATTERN.findall(query.concept.lower()))
        if (
            phrase in self._concept_phrases[best.asset_id]
            or phrase in self._alias_phrases[best.asset_id]
        ):
            return best
        query_terms = normalize_tokens(query.concept)
        covered = query_terms.intersection(
            self._concept_tokens[best.asset_id]
            | self._alias_tokens[best.asset_id]
        )
        coverage = len(covered) / max(1, len(query_terms))
        return best if coverage >= 0.6 else None

    def rank(
        self,
        query: AssetQuery,
        limit: int = 3,
        *,
        exact_only: bool = False,
    ) -> list[CatalogAsset]:
        """Rank direct or compositional candidates with deterministic tie-breaking."""

        concept_terms = normalize_tokens(query.concept)
        semantic_terms = normalize_tokens(" ".join(query.required_semantics))
        query_terms = concept_terms | semantic_terms
        normalized_phrase = " ".join(_TOKEN_PATTERN.findall(query.concept.lower()))
        candidates: list[tuple[float, CatalogAsset]] = []
        for asset in self._assets.values():
            if not self._kind_compatible(asset.kind, query.asset_kind):
                continue
            if asset.style_id not in {query.style_id, "*"}:
                continue
            concept_vocabulary = self._concept_tokens[asset.asset_id]
            alias_vocabulary = self._alias_tokens[asset.asset_id]
            exact_terms = concept_vocabulary | alias_vocabulary
            concept_overlap = concept_terms.intersection(concept_vocabulary)
            alias_overlap = concept_terms.intersection(alias_vocabulary)
            direct_overlap = concept_overlap | alias_overlap
            if exact_only and not direct_overlap:
                continue
            all_terms = self._search_tokens[asset.asset_id]
            score = 6.0 * len(concept_overlap)
            score += 4.0 * len(alias_overlap)
            score += 5.0 * len(semantic_terms.intersection(concept_vocabulary))
            score += 3.0 * len(semantic_terms.intersection(alias_vocabulary))
            score += 1.5 * len(query_terms.intersection(all_terms - exact_terms))
            if normalized_phrase in self._concept_phrases[asset.asset_id]:
                score += 20.0
            elif normalized_phrase in self._alias_phrases[asset.asset_id]:
                score += 10.0
            if asset.style_id == query.style_id:
                score += 0.25
            if score > 0.25:
                candidates.append((score, asset))
        return [
            asset
            for _, asset in sorted(
                candidates,
                key=lambda item: (-item[0], item[1].asset_id),
            )[: max(0, limit)]
        ]

    def records(self) -> list[CatalogAsset]:
        """Return deterministic catalog records."""

        return [self._assets[key] for key in sorted(self._assets)]

    @staticmethod
    def _kind_compatible(asset_kind: AssetKind, query_kind: AssetKind) -> bool:
        """Allow normalized SVG line art to serve icon, SVG, and line-art queries."""

        vector_kinds = {AssetKind.ICON, AssetKind.SVG, AssetKind.LINE_ART}
        return asset_kind is query_kind or (
            asset_kind in vector_kinds and query_kind in vector_kinds
        )

    @staticmethod
    def _load_index(index_path: Path) -> list[CatalogAsset]:
        """Load the checked-in metadata index without performing network access."""

        working_path = index_path if index_path.is_absolute() else PROJECT_ROOT / index_path
        if not working_path.is_file():
            raise FileNotFoundError(f"semantic asset index is missing: {working_path}")
        payload = json.loads(working_path.read_text(encoding="utf-8"))
        records = payload.get("assets")
        if not isinstance(records, list) or not records:
            raise ValueError("semantic asset index must contain non-empty assets")
        result: list[CatalogAsset] = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("semantic asset index records must be objects")
            result.append(
                CatalogAsset(
                    asset_id=str(record["asset_id"]),
                    concepts=frozenset(record["concepts"]),
                    kind=AssetKind(record.get("kind", "line_art")),
                    style_id=str(record.get("style_id", "whiteboard.default")),
                    path=Path(record["path"]),
                    mime_type=str(record.get("mime_type", "image/svg+xml")),
                    license_id=str(record["license_id"]),
                    editable=bool(record.get("editable", True)),
                    aliases=frozenset(record.get("aliases", [])),
                    categories=frozenset(record.get("categories", [])),
                    tags=frozenset(record.get("tags", [])),
                    relations=frozenset(record.get("relations", [])),
                    composition_roles=frozenset(record.get("composition_roles", [])),
                    source_collection=str(record.get("source_collection", "internal")),
                )
            )
        return result
