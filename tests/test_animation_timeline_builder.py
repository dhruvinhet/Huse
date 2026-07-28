"""Tests for declarative, non-overlapping animation timelines."""

from itertools import pairwise

import pytest

from app.core.animation_timeline_builder import AnimationTimelineBuilder
from app.models.animation import AnimationTimeline, AnimationType
from app.models.render import RenderableObject, RenderScene


def renderable(
    object_id: str,
    object_type: str,
    start_time: float = 0.0,
    end_time: float = 5.0,
) -> RenderableObject:
    """Create a minimal valid renderable object for timeline tests."""

    return RenderableObject(
        object_id=object_id,
        type=object_type,
        content="content",
        x=960,
        y=540,
        width=128,
        height=128,
        animation="unused",
        start_time=start_time,
        end_time=end_time,
    )


def timeline_scene() -> RenderScene:
    """Create a scene covering all mapped animation categories."""

    return RenderScene(
        scene_number=1,
        objects=[
            renderable("text-1", "text"),
            renderable("svg-1", "svg"),
            renderable("icon-1", "icon", end_time=0.5),
            renderable("image-1", "image"),
        ],
    )


def test_timeline_is_created() -> None:
    """The builder returns one scene timeline for each render scene."""

    timeline = AnimationTimelineBuilder().build([timeline_scene()])

    assert isinstance(timeline, AnimationTimeline)
    assert len(timeline.scenes) == 1
    assert timeline.scenes[0].scene_number == 1
    assert len(timeline.scenes[0].animations) == 4


def test_animation_types_are_mapped() -> None:
    """Renderable object types map to their required animation categories."""

    animations = AnimationTimelineBuilder().build(
        [timeline_scene()]
    ).scenes[0].animations

    assert [item.animation for item in animations] == [
        AnimationType.WRITE,
        AnimationType.DRAW,
        AnimationType.FADE,
        AnimationType.FADE,
    ]


def test_animations_are_ordered_without_overlap() -> None:
    """Animation start times follow object order without overlap."""

    animations = AnimationTimelineBuilder().build(
        [timeline_scene()]
    ).scenes[0].animations

    assert [item.start_time for item in animations] == [0.0, 1.0, 2.0, 2.5]
    for previous, current in pairwise(animations):
        assert current.start_time >= previous.start_time + previous.duration


def test_animation_durations_are_defaulted_and_clamped() -> None:
    """Long objects use one second and shorter objects are clamped."""

    animations = AnimationTimelineBuilder().build(
        [timeline_scene()]
    ).scenes[0].animations

    assert [item.duration for item in animations] == pytest.approx(
        [1.0, 1.0, 0.5, 1.0]
    )
