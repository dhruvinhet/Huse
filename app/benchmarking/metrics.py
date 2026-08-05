"""Deterministic benchmark metrics and lightweight perceptual comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from app.domain.camera import CameraOperation, CameraPlan
from app.domain.layout import LaidOutNode, LayoutPlan
from app.domain.narration import AlignedAudio, NarrationPlan
from app.domain.operations import OperationType
from app.domain.quality import QualityReport
from app.domain.storyboard import Storyboard
from app.domain.visual_document import ObjectLifecycle, VisualDocument
from app.benchmarking.models import BenchmarkMetrics


_PRESENTATION_ONLY = {
    OperationType.CREATE,
    OperationType.SHOW,
    OperationType.HIDE,
    OperationType.HIGHLIGHT,
    OperationType.DIM,
}
_CLIPPING_CODES = {
    "object_clipped",
    "frame_content_clipped",
    "safe_area_violation",
}
_READABILITY_CODES = {
    "text_box_unreadable",
    "semantic_text_geometry_unreadable",
    "text_collision",
    "connector_label_collision",
}


def _flatten(root: LaidOutNode) -> list[LaidOutNode]:
    result = [root]
    for child in root.children:
        result.extend(_flatten(child))
    return result


def _state_signature(document: VisualDocument, index: int) -> str:
    state = document.states[index]
    payload = [
        (
            object_id,
            item.lifecycle.value,
            item.content,
            item.parent_id,
            item.child_ids,
        )
        for object_id, item in sorted(state.object_states.items())
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _source_rect(cue: object) -> tuple[float, float, float, float] | None:
    values = [
        cue.parameters.get(name)
        for name in (
            "source_left",
            "source_top",
            "source_right",
            "source_bottom",
        )
    ]
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def collect_metrics(
    *,
    storyboard: Storyboard | None = None,
    document: VisualDocument | None = None,
    layout: LayoutPlan | None = None,
    camera: CameraPlan | None = None,
    narration: NarrationPlan | None = None,
    alignment: AlignedAudio | None = None,
    quality: QualityReport | None = None,
    duration: float | None = None,
) -> BenchmarkMetrics:
    """Calculate stable structural metrics from available pipeline artifacts."""

    topic_specificity: float | None = None
    total_objects = 0
    if document is not None and document.states:
        objects = [
            item
            for item in document.states[-1].object_states.values()
            if item.kind != "connector"
            and item.lifecycle is not ObjectLifecycle.REMOVED
            and item.parent_id is not None
        ]
        total_objects = len(objects)
        linked = [
            item
            for item in objects
            if isinstance(item.metadata.get("concept_ids"), list)
            and bool(item.metadata["concept_ids"])
        ]
        topic_specificity = len(linked) / len(objects) if objects else 1.0

    action_coverage: float | None = None
    state_delta_coverage: float | None = None
    total_beats = len(storyboard.beats) if storyboard is not None else 0
    if storyboard is not None:
        transformation_beats = [
            beat
            for beat in storyboard.beats
            if beat.purpose in {"transform", "demonstrate", "connect", "compare"}
        ]
        if transformation_beats:
            action_coverage = sum(
                any(operation.operation not in _PRESENTATION_ONLY for operation in beat.operations)
                for beat in transformation_beats
            ) / len(transformation_beats)
            if document is not None:
                by_beat = {
                    state.beat_id: index
                    for index, state in enumerate(document.states)
                }
                changed = 0
                comparable = 0
                for beat in transformation_beats:
                    index = by_beat.get(beat.beat_id)
                    if index is None or index == 0:
                        continue
                    comparable += 1
                    changed += _state_signature(document, index) != _state_signature(
                        document, index - 1
                    )
                state_delta_coverage = (
                    changed / comparable if comparable else 0.0
                )
        else:
            action_coverage = 1.0
            state_delta_coverage = 1.0

    composition_changes_per_minute: float | None = None
    if layout is not None and layout.state_roots:
        fingerprints: list[tuple[tuple[str, str], ...]] = []
        for root in layout.state_roots.values():
            fingerprints.append(tuple(
                sorted((node.kind, node.operator) for node in _flatten(root))
            ))
        changes = sum(
            first != second
            for first, second in zip(fingerprints, fingerprints[1:])
        )
        measured_duration = duration or float(len(fingerprints))
        composition_changes_per_minute = changes / max(
            measured_duration / 60.0,
            1 / 60,
        )

    camera_change_coverage: float | None = None
    if camera is not None:
        expected = [
            cue
            for cue in camera.cues
            if cue.operation not in {CameraOperation.FIT, CameraOperation.HOLD}
        ]
        meaningful = 0
        for cue in expected:
            current = _source_rect(cue)
            previous_values = [
                cue.parameters.get(name)
                for name in (
                    "source_start_left",
                    "source_start_top",
                    "source_start_right",
                    "source_start_bottom",
                )
            ]
            previous = (
                tuple(float(value) for value in previous_values)
                if all(isinstance(value, (int, float)) for value in previous_values)
                else None
            )
            if current is not None and previous is not None:
                meaningful += max(
                    abs(first - second)
                    for first, second in zip(current, previous, strict=True)
                ) >= 2.0
        camera_change_coverage = (
            meaningful / len(expected) if expected else 1.0
        )

    audio_timing_coverage: float | None = None
    if narration is not None and alignment is not None:
        spoken_words = sum(
            len(re.findall(r"[A-Za-z0-9']+", phrase.text))
            for phrase in narration.phrases
        )
        aligned_words = sum(word.confidence >= 0.5 for word in alignment.words)
        audio_timing_coverage = (
            min(1.0, aligned_words / spoken_words) if spoken_words else 1.0
        )

    finding_codes = [finding.code for finding in quality.findings] if quality else []
    return BenchmarkMetrics(
        topic_specificity=topic_specificity,
        action_coverage=action_coverage,
        state_delta_coverage=state_delta_coverage,
        composition_changes_per_minute=composition_changes_per_minute,
        camera_change_coverage=camera_change_coverage,
        clipping_violations=sum(code in _CLIPPING_CODES for code in finding_codes),
        readability_violations=sum(code in _READABILITY_CODES for code in finding_codes),
        audio_timing_coverage=audio_timing_coverage,
        quality_score=quality.overall_score if quality is not None else None,
        total_beats=total_beats,
        total_objects=total_objects,
    )


def perceptual_distance(
    first_path: str | Path,
    second_path: str | Path,
    sample_size: tuple[int, int] = (64, 64),
) -> float:
    """Return normalized RMS pixel distance after deterministic resampling."""

    with Image.open(first_path) as first_image, Image.open(second_path) as second_image:
        first = first_image.convert("RGB").resize(sample_size, Image.Resampling.LANCZOS)
        second = second_image.convert("RGB").resize(sample_size, Image.Resampling.LANCZOS)
        difference = ImageChops.difference(first, second)
        rms = ImageStat.Stat(difference).rms
    return sum(rms) / (len(rms) * 255.0)
