"""Atomic, content-addressed stage checkpoints with lineage metadata."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel as PydanticModel


ModelT = TypeVar("ModelT", bound=PydanticModel)


class ArtifactManifestEntry(PydanticModel):
    """Describe one persisted stage artifact."""

    artifact_id: str
    stage: str
    schema_version: str
    producer: str
    content_hash: str
    path: str
    created_at: datetime
    parent_artifact_ids: list[str]


class ArtifactCheckpointStore:
    """Persist and reload validated artifacts for resumable runs."""

    def __init__(self, root_dir: Path) -> None:
        """Create a run-scoped checkpoint directory."""

        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root_dir / "artifact_manifest.json"

    def write(
        self,
        stage: str,
        artifact: PydanticModel,
        producer: str,
        parent_artifact_ids: list[str] | None = None,
    ) -> ArtifactManifestEntry:
        """Atomically write one artifact and update its manifest entry."""

        normalized_stage = self._safe_stage(stage)
        value = artifact.model_dump(mode="json")
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        content_hash = sha256(serialized.encode("utf-8")).hexdigest()
        artifact_id = f"{normalized_stage}_{content_hash[:16]}"
        destination = self.root_dir / f"{normalized_stage}.json"
        self._atomic_json(destination, value)
        entry = ArtifactManifestEntry(
            artifact_id=artifact_id,
            stage=normalized_stage,
            schema_version=str(value.get("schema_version", "1.0")),
            producer=producer,
            content_hash=content_hash,
            path=destination.as_posix(),
            created_at=datetime.now().astimezone(),
            parent_artifact_ids=parent_artifact_ids or [],
        )
        entries = {
            item["stage"]: item
            for item in self._read_manifest()
        }
        entries[normalized_stage] = entry.model_dump(mode="json")
        self._atomic_json(
            self._manifest_path,
            [entries[key] for key in sorted(entries)],
        )
        return entry

    def load(
        self,
        stage: str,
        model_type: type[ModelT],
        schema_version: str = "2.0",
    ) -> ModelT:
        """Reload and validate a compatible checkpoint."""

        normalized_stage = self._safe_stage(stage)
        entries = {
            item["stage"]: item
            for item in self._read_manifest()
        }
        try:
            entry = entries[normalized_stage]
        except KeyError as exc:
            raise FileNotFoundError(
                f"checkpoint stage is unavailable: {normalized_stage}"
            ) from exc
        if entry["schema_version"] != schema_version:
            raise ValueError(
                f"checkpoint schema {entry['schema_version']} is incompatible "
                f"with {schema_version}"
            )
        path = Path(entry["path"])
        try:
            path.resolve().relative_to(self.root_dir.resolve())
        except ValueError as exc:
            raise ValueError("checkpoint path escapes the run directory") from exc
        payload = path.read_bytes()
        digest = sha256(
            json.dumps(
                json.loads(payload),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if digest != entry["content_hash"]:
            raise ValueError("checkpoint content hash does not match manifest")
        return model_type.model_validate_json(payload)

    def has(self, stage: str, schema_version: str = "2.0") -> bool:
        """Return whether a compatible checkpoint exists."""

        normalized_stage = self._safe_stage(stage)
        return any(
            item["stage"] == normalized_stage
            and item["schema_version"] == schema_version
            and Path(item["path"]).is_file()
            for item in self._read_manifest()
        )

    def _read_manifest(self) -> list[dict[str, Any]]:
        """Read the current manifest or return an empty list."""

        if not self._manifest_path.is_file():
            return []
        value = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("artifact manifest must contain a JSON array")
        return value

    @staticmethod
    def _safe_stage(stage: str) -> str:
        """Normalize a stage name to a traversal-safe filename."""

        normalized = "_".join(stage.strip().lower().replace("-", " ").split())
        if not normalized or not normalized.replace("_", "").isalnum():
            raise ValueError("stage must contain only letters, numbers, spaces, or hyphens")
        return normalized

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        """Write JSON atomically in the destination directory."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
