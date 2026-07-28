"""Storyboard contracts connecting concepts to visual operations."""

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from app.domain.assets import AssetQuery
from app.domain.layout import LayoutConstraint
from app.domain.operations import OperationType, VisualOperation
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
    attention: list[AttentionCue] = Field(default_factory=list)
    camera_intent: CameraIntent | None = None


class Storyboard(BaseModel):
    """Describe a complete persistent, storyboard-first visual lesson."""

    schema_version: Literal["2.0"] = "2.0"
    document_id: NonEmptyString
    title: NonEmptyString
    beats: list[VisualBeat] = Field(min_length=1)
    initial_objects: list[VisualObjectSpec] = Field(default_factory=list)
    final_learning_summary: list[NonEmptyString] = Field(default_factory=list)

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
