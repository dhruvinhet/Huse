"""Tests for rendering animation timelines into sequential PNG frames."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.core import frame_renderer
from app.core.frame_renderer import FrameRenderer
from app.models.animation import (
    AnimationInstruction,
    AnimationType,
    SceneTimeline,
)
from app.models.render import RenderableObject, RenderScene
from app.utils.debug_recorder import DebugRecorder


def frame_inputs() -> tuple[RenderScene, SceneTimeline]:
    """Create a one-second text scene and matching write animation."""

    obj = RenderableObject(
        object_id="text-1",
        type="text",
        content="Hello",
        x=960,
        y=540,
        width=320,
        height=64,
        animation="write",
        start_time=0.0,
        end_time=1.0,
    )
    scene = RenderScene(scene_number=1, objects=[obj])
    timeline = SceneTimeline(
        scene_number=1,
        animations=[
            AnimationInstruction(
                object_id="text-1",
                animation=AnimationType.WRITE,
                start_time=0.0,
                duration=1.0,
            )
        ],
    )
    return scene, timeline


def configure_temp_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point frame output at an isolated temporary directory."""

    temp_dir = tmp_path / "temp"
    monkeypatch.setattr(
        frame_renderer,
        "settings",
        SimpleNamespace(TEMP_DIR=temp_dir),
    )
    return temp_dir / "frames"


def test_correct_frame_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scene duration multiplied by FPS determines the PNG frame count."""

    frames_directory = configure_temp_frames(tmp_path, monkeypatch)
    scene, timeline = frame_inputs()

    FrameRenderer().render(scene, timeline, fps=4)

    assert len(list(frames_directory.glob("frame_*.png"))) == 4


def test_frame_filenames_are_sequential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame files use one-based, six-digit sequential names."""

    frames_directory = configure_temp_frames(tmp_path, monkeypatch)
    scene, timeline = frame_inputs()

    FrameRenderer().render(scene, timeline, fps=3)

    assert [path.name for path in sorted(frames_directory.glob("*.png"))] == [
        "frame_000001.png",
        "frame_000002.png",
        "frame_000003.png",
    ]


def test_png_frames_are_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every generated frame is a valid full-HD PNG image."""

    frames_directory = configure_temp_frames(tmp_path, monkeypatch)
    scene, timeline = frame_inputs()

    FrameRenderer().render(scene, timeline, fps=2)

    frame_paths = sorted(frames_directory.glob("*.png"))
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in frame_paths)
    with Image.open(frame_paths[-1]) as final_frame:
        assert final_frame.size == (1920, 1080)


def test_global_frame_numbering_continues_between_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit frame start continues the shared output sequence."""

    frames_directory = configure_temp_frames(tmp_path, monkeypatch)
    scene, timeline = frame_inputs()
    renderer = FrameRenderer()

    renderer.render(scene, timeline, fps=2, frame_count=2)
    renderer.render(
        scene,
        timeline,
        fps=2,
        frame_start=3,
        frame_count=2,
        clear_output=False,
    )

    assert [path.name for path in sorted(frames_directory.glob("*.png"))] == [
        "frame_000001.png",
        "frame_000002.png",
        "frame_000003.png",
        "frame_000004.png",
    ]


def test_existing_global_frame_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendering refuses a range containing an existing frame file."""

    configure_temp_frames(tmp_path, monkeypatch)
    scene, timeline = frame_inputs()
    renderer = FrameRenderer()
    renderer.render(scene, timeline, fps=1, frame_count=1)

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        renderer.render(
            scene,
            timeline,
            fps=1,
            frame_count=1,
            clear_output=False,
        )


def test_each_frame_has_a_structured_debug_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame debugging records timing, object progress, and sample PNGs."""

    configure_temp_frames(tmp_path, monkeypatch)
    recorder = DebugRecorder(True, tmp_path / "debug")
    run_dir = recorder.start_run("Frame Trace")
    assert run_dir is not None
    scene, timeline = frame_inputs()

    FrameRenderer(recorder).render(scene, timeline, fps=2)

    trace_path = run_dir / "frames" / "frame_trace.jsonl"
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["frame_number"] for record in records] == [1, 2]
    assert records[0]["scene_number"] == 1
    assert records[0]["objects"][0]["animation"] == "write"
    assert records[1]["objects"][0]["visible_text"] == "Hello"
    sample_files = list((run_dir / "frames" / "samples").glob("*.png"))
    assert len(sample_files) == 2
