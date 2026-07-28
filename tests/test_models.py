"""Tests for the core pipeline data models."""

import pytest
from pydantic import ValidationError

from app.models import (
    PipelineResult,
    RenderableObject,
    RenderScene,
    Scene,
    Script,
    VisualInstruction,
)


def build_script() -> Script:
    """Create a valid nested script for reuse across model tests."""

    visual = VisualInstruction(
        type="text",
        content="Define artificial intelligence",
        position="center",
        animation="draw",
    )
    scene = Scene(
        scene_number=1,
        title="Introduction",
        narration="Artificial intelligence helps computers solve problems.",
        estimated_duration=12.5,
        visuals=[visual],
    )
    return Script(
        title="AI Explained",
        topic="Artificial intelligence",
        total_duration=60,
        scenes=[scene],
    )


def build_render_scene() -> RenderScene:
    """Create a valid renderer-ready scene for reuse in tests."""

    renderable = RenderableObject(
        object_id="title-1",
        type="text",
        content="Artificial Intelligence",
        x=100,
        y=80,
        width=600,
        height=100,
        animation="write",
        start_time=0.0,
        end_time=3.5,
    )
    return RenderScene(scene_number=1, objects=[renderable])


def test_models_can_be_instantiated() -> None:
    """All core models accept valid typed pipeline data."""

    script = build_script()
    render_scene = build_render_scene()
    result = PipelineResult(
        success=True,
        message="Script created successfully.",
        script=script,
    )

    assert script.scenes[0].visuals[0].type == "text"
    assert render_scene.objects[0].width == 600
    assert result.script == script


def test_models_support_dump_validate_and_json_round_trip() -> None:
    """Models support Pydantic dictionary and JSON serialization APIs."""

    result = PipelineResult(
        success=True,
        message="Script created successfully.",
        script=build_script(),
    )
    validated_result = PipelineResult.model_validate(result.model_dump())
    json_result = PipelineResult.model_validate_json(result.model_dump_json())

    render_scene = build_render_scene()
    validated_render_scene = RenderScene.model_validate(render_scene.model_dump())
    json_render_scene = RenderScene.model_validate_json(
        render_scene.model_dump_json()
    )

    assert validated_result == result
    assert json_result == result
    assert validated_render_scene == render_scene
    assert json_render_scene == render_scene


def test_models_reject_invalid_values_and_extra_fields() -> None:
    """Core validation rejects malformed or unexpected pipeline data."""

    with pytest.raises(ValidationError):
        Scene(
            scene_number=0,
            title=" ",
            narration="Narration",
            estimated_duration=0,
            visuals=[],
            unexpected="value",
        )

    with pytest.raises(ValidationError):
        RenderableObject(
            object_id="object-1",
            type="text",
            content="Invalid timing",
            x=0,
            y=0,
            width=100,
            height=100,
            animation="draw",
            start_time=2.0,
            end_time=1.0,
        )


def test_assignment_is_validated() -> None:
    """Assignment validation prevents invalid post-creation state."""

    scene = build_script().scenes[0]

    with pytest.raises(ValidationError):
        scene.estimated_duration = 0
