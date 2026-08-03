"""Resolve semantic assets through catalog lookup and line-art fallback."""

from hashlib import sha256
import json
from pathlib import Path

from app.config.settings import PROJECT_ROOT, settings
from app.domain.assets import (
    AssetKind,
    AssetQuery,
    AssetSource,
    ResolvedAssetSet,
    ResolvedSemanticAsset,
)
from app.domain.operations import OperationType
from app.domain.storyboard import Storyboard, VisualObjectSpec
from app.semantic_assets.catalog import AssetCatalog
from app.semantic_assets.line_art import DeterministicLineArtGenerator


class CatalogSemanticAssetResolver:
    """Resolve all semantic queries without producing placeholders."""

    def __init__(
        self,
        catalog: AssetCatalog | None = None,
        generator: DeterministicLineArtGenerator | None = None,
        generated_dir: Path | None = None,
    ) -> None:
        """Store catalog and an editable SVG fallback generator."""

        self._catalog = catalog if catalog is not None else AssetCatalog()
        self._generator = (
            generator
            if generator is not None
            else DeterministicLineArtGenerator(self._catalog)
        )
        self._generated_dir = generated_dir

    def resolve(self, storyboard: Storyboard) -> ResolvedAssetSet:
        """Resolve unique asset queries found in initial and created objects."""

        objects = self._storyboard_objects(storyboard)
        resolved: list[ResolvedSemanticAsset] = []
        seen_objects: set[tuple[str, str]] = set()
        for item in objects:
            query = item.asset_query
            if query is None:
                continue
            digest = self._query_digest(query)
            identity = (item.object_id, digest)
            if identity in seen_objects:
                continue
            seen_objects.add(identity)
            resolved.append(self._resolve_query(item.object_id, query, digest))
        return ResolvedAssetSet(assets=resolved)

    def _resolve_query(
        self,
        object_id: str,
        query: AssetQuery,
        digest: str,
    ) -> ResolvedSemanticAsset:
        """Resolve one catalog entry or generate a deterministic fallback."""

        if query.asset_kind is AssetKind.TEMPLATE:
            return ResolvedSemanticAsset(
                asset_id=f"asset_{object_id}",
                query_digest=digest,
                source=AssetSource.TEMPLATE,
                path=f"template://{query.concept}",
                mime_type="application/x-whiteboard-template",
                license_id="internal-template",
                content_hash=digest,
                editable=True,
                ready=True,
            )
        catalog_asset = self._catalog.find(query)
        if catalog_asset is not None:
            working_path = self._working_path(catalog_asset.path)
            if not working_path.is_file():
                raise FileNotFoundError(
                    f"catalog asset is missing: {working_path}"
                )
            return ResolvedSemanticAsset(
                asset_id=f"asset_{object_id}",
                query_digest=digest,
                source=AssetSource.CATALOG,
                path=catalog_asset.path.as_posix(),
                mime_type=catalog_asset.mime_type,
                license_id=catalog_asset.license_id,
                content_hash=self._file_hash(working_path),
                editable=catalog_asset.editable,
                ready=True,
            )

        configured_path = (
            self._generated_dir / f"{digest}.svg"
            if self._generated_dir is not None
            else Path(settings.TEMP_DIR) / "v2" / "semantic_assets" / f"{digest}.svg"
        )
        working_path = self._working_path(configured_path)
        composition = self._generator.plan(query)
        if not working_path.is_file():
            self._generator.generate(query, working_path)
        license_id = (
            "generated-internal"
            if composition.diagnostic_fallback
            else "composed:" + "+".join(composition.license_ids)
        )
        return ResolvedSemanticAsset(
            asset_id=f"asset_{object_id}",
            query_digest=digest,
            source=AssetSource.GENERATED,
            path=configured_path.as_posix(),
            mime_type="image/svg+xml",
            license_id=license_id,
            content_hash=self._file_hash(working_path),
            editable=True,
            ready=True,
        )

    def _storyboard_objects(self, storyboard: Storyboard) -> list[VisualObjectSpec]:
        """Collect initial and create-operation definitions."""

        objects = [
            item
            for root in storyboard.initial_objects
            for item in root.flatten()
        ]
        for beat in storyboard.beats:
            for operation in beat.operations:
                if operation.operation is not OperationType.CREATE:
                    continue
                raw_objects = operation.arguments.get("objects")
                if not isinstance(raw_objects, list):
                    continue
                definitions = [
                    VisualObjectSpec.model_validate(raw)
                    for raw in raw_objects
                ]
                objects.extend(
                    item for root in definitions for item in root.flatten()
                )
        return objects

    @staticmethod
    def _query_digest(query: AssetQuery) -> str:
        """Return a stable digest for cache and deduplication."""

        payload = json.dumps(
            query.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Hash file contents for provenance and cache validation."""

        digest = sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _working_path(path: Path) -> Path:
        """Resolve configured relative paths against the project root."""

        return path if path.is_absolute() else PROJECT_ROOT / path
