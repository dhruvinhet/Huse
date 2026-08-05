"""Storyboard contracts connecting concepts to visual operations."""

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from app.domain.assets import AssetQuery
from app.domain.layout import LayoutConstraint
from app.domain.operations import OperationType, SemanticAction, VisualOperation
from app.models.base import BaseModel, NonEmptyString


class AttentionCue(BaseModel):
    """Describe where the viewer should look during a beat."""

    cue: Literal["highlight", "dim", "zoom", "pulse", "underline", "glow", "focus"]
    target_ids: list[NonEmptyString] = Field(min_length=1)
    intensity: float = Field(default=1.0, ge=0, le=1)


class CameraIntent(BaseModel):
    """Describe camera intent before geometric planning."""

    operation: Literal["fit", "pan", "zoom", "focus", "track", "hold"]
    target_ids: list[NonEmptyString] = Field(default_factory=list)
    emphasis: float = Field(default=0.5, ge=0, le=1)


class ShotPlan(BaseModel):
    """Deterministic occupancy and lifecycle policy for one teaching shot."""

    enter: Literal["reveal", "draw", "fade"] = "reveal"
    hold: Literal["focus", "compare", "trace", "hold"] = "focus"
    exit: Literal["hide", "dim", "replace", "retain"] = "hide"
    occupancy_target: tuple[float, float] = (0.35, 0.70)
    max_active_objects: int = Field(default=12, ge=3, le=64)
    focal_region: Literal["center", "left", "right", "top", "bottom"] = "center"
    cleanup_policy: Literal["hide_non_target", "dim_non_target", "replace", "retain"] = "hide_non_target"

    @classmethod
    def for_purpose(cls, purpose: str) -> "ShotPlan":
        """Create a bounded shot policy from the pedagogical purpose."""

        if purpose == "summarize":
            return cls(
                enter="fade",
                hold="hold",
                exit="replace",
                occupancy_target=(0.40, 0.65),
                max_active_objects=8,
            )
        if purpose in {"transform", "demonstrate"}:
            return cls(
                enter="draw",
                hold="trace",
                exit="hide",
                occupancy_target=(0.40, 0.70),
                max_active_objects=10,
            )
        if purpose == "connect":
            return cls(
                enter="reveal",
                hold="compare",
                exit="hide",
                occupancy_target=(0.35, 0.65),
                max_active_objects=14,
            )
        return cls()


class VisualObjectSpec(BaseModel):
    """Describe a semantic object before layout and rendering."""

    object_id: NonEmptyString
    kind: NonEmptyString
    semantic_role: NonEmptyString
    concept_ids: list[NonEmptyString] = Field(default_factory=list)
    content: dict[str, JsonValue] = Field(default_factory=dict)
    style_token: NonEmptyString = "concept.primary"
    asset_query: AssetQuery | None = None
    children: list["VisualObjectSpec"] = Field(default_factory=list)
    constraints: list[LayoutConstraint] = Field(default_factory=list)
    accessibility_label: NonEmptyString

    def flatten(self) -> list["VisualObjectSpec"]:
        """Return this object and descendants in stable pre-order."""

        flattened = [self]
        for child in self.children:
            flattened.extend(child.flatten())
        return flattened


class VisualBeat(BaseModel):
    """Represent one pedagogical visual step and its intended speech."""

    beat_id: NonEmptyString
    section_id: NonEmptyString
    concept_ids: list[NonEmptyString] = Field(min_length=1)
    teaching_intent: NonEmptyString
    phrase_intent: NonEmptyString
    purpose: Literal[
        "introduce",
        "demonstrate",
        "compare",
        "transform",
        "connect",
        "emphasize",
        "summarize",
    ] = "introduce"
    estimated_duration: float = Field(gt=0)
    operations: list[VisualOperation] = Field(min_length=1)
    semantic_actions: list[SemanticAction] = Field(default_factory=list, max_length=8)
    attention: list[AttentionCue] = Field(default_factory=list)
    camera_intent: CameraIntent | None = None
    shot_plan: ShotPlan | None = None


class Storyboard(BaseModel):
    """Describe a complete persistent, storyboard-first visual lesson."""

    schema_version: Literal["2.0"] = "2.0"
    document_id: NonEmptyString
    title: NonEmptyString
    beats: list[VisualBeat] = Field(min_length=1)
    initial_objects: list[VisualObjectSpec] = Field(default_factory=list)
    final_learning_summary: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_model_shape(cls, value: object) -> object:
        """Normalize two harmless placement omissions from smaller models."""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if not normalized.get("document_id"):
            normalized["document_id"] = "storyboard"

        beats = normalized.get("beats")
        root_objects = normalized.get("initial_objects")
        if isinstance(beats, list) and beats and (
            not isinstance(root_objects, list) or not root_objects
        ):
            first = beats[0]
            if isinstance(first, dict) and isinstance(
                first.get("initial_objects"), list
            ):
                normalized["initial_objects"] = first["initial_objects"]
                beats[0] = {
                    key: item
                    for key, item in first.items()
                    if key != "initial_objects"
                }
                normalized["beats"] = beats
        return normalized

    @model_validator(mode="before")
    @classmethod
    def drop_duplicate_create_operations(cls, value: object) -> object:
        """Drop model-generated creates for objects that already exist.

        A repeated create cannot represent a valid state transition. Removing
        only that redundant operation preserves the existing object and lets
        later update/highlight operations continue to target it.
        """

        if not isinstance(value, dict):
            return value
        beats = value.get("beats")
        if not isinstance(beats, list):
            return value

        known: set[str] = set()
        retired: set[str] = set()
        for root in value.get("initial_objects", []):
            if isinstance(root, dict):
                known.update(_raw_object_ids(root))

        changed = False
        cleaned_beats: list[object] = []
        for beat in beats:
            if not isinstance(beat, dict):
                cleaned_beats.append(beat)
                continue
            operations = beat.get("operations")
            if not isinstance(operations, list):
                cleaned_beats.append(beat)
                continue

            cleaned_operations: list[object] = []
            for operation in operations:
                if not isinstance(operation, dict):
                    cleaned_operations.append(operation)
                    continue
                operation_name = str(operation.get("operation", "")).lower()
                targets = {
                    item
                    for item in operation.get("target_ids", [])
                    if isinstance(item, str)
                }
                if operation_name == "create":
                    definitions = operation.get("arguments", {}).get("objects", [])
                    created_ids = {
                        object_id
                        for definition in definitions
                        if isinstance(definition, dict)
                        for object_id in _raw_object_ids(definition)
                    }
                    if created_ids & (known | retired):
                        changed = True
                        continue
                    known.update(created_ids or targets)
                elif operation_name == "duplicate":
                    known.update(targets)
                elif operation_name == "erase":
                    known.difference_update(targets)
                    retired.update(targets)
                cleaned_operations.append(operation)

            if cleaned_operations:
                cleaned_beats.append({**beat, "operations": cleaned_operations})
            else:
                changed = True

        if not changed:
            return value
        return {**value, "beats": cleaned_beats}

    @model_validator(mode="before")
    @classmethod
    def drop_empty_operation_beats(cls, value: object) -> object:
        """Remove model-generated operations that cannot affect rendering.

        Some providers emit prose-only beats with ``operations: []`` or create
        operations without the required ``arguments.objects`` definitions. These
        entries cannot affect the rendered document, so remove them when the
        storyboard also contains valid visual beats. If every beat is empty, the
        normal field validation still rejects the storyboard.
        """

        if not isinstance(value, dict):
            return value
        beats = value.get("beats")
        if not isinstance(beats, list):
            return value
        cleaned_beats: list[object] = []
        for beat in beats:
            if not isinstance(beat, dict):
                cleaned_beats.append(beat)
                continue
            operations = beat.get("operations")
            if not isinstance(operations, list):
                cleaned_beats.append(beat)
                continue
            cleaned_operations = [
                operation
                for operation in operations
                if not (
                    isinstance(operation, dict)
                    and str(operation.get("operation", "")).lower() == "create"
                    and not isinstance(
                        operation.get("arguments", {}).get("objects")
                        if isinstance(operation.get("arguments"), dict)
                        else None,
                        list,
                    )
                )
            ]
            if cleaned_operations:
                cleaned_beats.append({**beat, "operations": cleaned_operations})
        valid_beats = cleaned_beats
        if valid_beats and len(valid_beats) != len(beats):
            return {**value, "beats": valid_beats}
        return value

    @model_validator(mode="after")
    def validate_storyboard(self) -> Self:
        """Require unique IDs and a valid sequential object lifecycle."""

        beat_ids = [beat.beat_id for beat in self.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("storyboard beat IDs must be unique")
        operations = [operation for beat in self.beats for operation in beat.operations]
        operation_ids = [operation.operation_id for operation in operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("visual operation IDs must be unique")

        flattened = [item for root in self.initial_objects for item in root.flatten()]
        initial_ids = [item.object_id for item in flattened]
        if len(initial_ids) != len(set(initial_ids)):
            raise ValueError("visual object IDs must be unique")

        known = set(initial_ids)
        retired: set[str] = set()
        for operation in operations:
            forbidden_geometry = {
                key.lower()
                for key in operation.arguments
                if key.lower() in {"x", "y", "width", "height", "left", "top"}
            }
            if forbidden_geometry:
                raise ValueError(
                    "storyboard operations cannot contain pixel geometry: "
                    f"{sorted(forbidden_geometry)}"
                )
            targets = set(operation.target_ids)
            if operation.operation is OperationType.CREATE:
                if targets & (known | retired):
                    raise ValueError("create targets must use new object IDs")
                raw_objects = operation.arguments.get("objects")
                if not isinstance(raw_objects, list):
                    raise ValueError("create operations require object definitions")
                definitions = [
                    VisualObjectSpec.model_validate(item)
                    for item in raw_objects
                ]
                root_ids = {item.object_id for item in definitions}
                if root_ids != targets:
                    raise ValueError(
                        "create target IDs must match root object definitions"
                    )
                created_ids = {
                    item.object_id
                    for root in definitions
                    for item in root.flatten()
                }
                if created_ids & (known | retired):
                    raise ValueError("created object IDs must be new")
                known.update(created_ids)
                continue
            if operation.operation is OperationType.DUPLICATE:
                source_id = operation.arguments.get("source_id")
                if not isinstance(source_id, str) or source_id not in known:
                    raise ValueError("duplicate operations require a known source_id")
                if targets & (known | retired):
                    raise ValueError("duplicate targets must use new object IDs")
                known.update(targets)
                continue
            if not targets.issubset(known):
                raise ValueError("operations must target currently existing objects")
            if operation.operation is OperationType.ERASE:
                known.difference_update(targets)
                retired.update(targets)

        all_known_ids = set(initial_ids)
        for operation in operations:
            all_known_ids.update(operation.target_ids)
            if operation.operation is OperationType.CREATE:
                raw_objects = operation.arguments.get("objects")
                if isinstance(raw_objects, list):
                    definitions = [
                        VisualObjectSpec.model_validate(item)
                        for item in raw_objects
                    ]
                    all_known_ids.update(
                        item.object_id
                        for root in definitions
                        for item in root.flatten()
                    )
        for beat in self.beats:
            for cue in beat.attention:
                if not set(cue.target_ids).issubset(all_known_ids):
                    raise ValueError("attention cues must target known objects")
            if beat.camera_intent and not set(beat.camera_intent.target_ids).issubset(all_known_ids):
                raise ValueError("camera intents must target known objects")
        return self


def _raw_object_ids(value: dict[str, object]) -> set[str]:
    """Collect object IDs from one raw object tree."""

    object_id = value.get("object_id")
    ids = {object_id} if isinstance(object_id, str) else set()
    children = value.get("children", [])
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                ids.update(_raw_object_ids(child))
    return ids
