"""Tests for end-to-end pipeline orchestration."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core import pipeline_runner
from app.core.animation_timeline_builder import AnimationTimelineBuilder
from app.core.asset_manager import AssetManager
from app.core.asset_planner import AssetPlanner
from app.core.audio_manager import AudioManager
from app.core.frame_renderer import FrameRenderer
from app.core.pipeline_runner import PipelineRunner
from app.core.scene_graph_builder import SceneGraphBuilder
from app.core.script_generator import ScriptGenerator
from app.core.timeline_synchronizer import TimelineSynchronizer
from app.core.video_composer import VideoComposer
from app.models.animation import AnimationTimeline, SceneTimeline
from app.models.audio import AudioMetadata, SceneAudio
from app.models.render import RenderScene
from app.models.video_manifest import SceneManifest, VideoManifest


def mocked_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PipelineRunner, list[str], dict[str, MagicMock]]:
    """Create a runner whose every pipeline module records execution."""

    temp_dir = tmp_path / "temp"
    monkeypatch.setattr(
        pipeline_runner,
        "settings",
        SimpleNamespace(TEMP_DIR=temp_dir),
    )
    events: list[str] = []
    script = SimpleNamespace(scenes=[SimpleNamespace(scene_number=1)])
    asset_plan = object()
    resolved_assets = object()
    scene = RenderScene(scene_number=1, objects=[])
    scenes = [scene]
    scene_timeline = SceneTimeline(
        scene_number=1,
        animations=[],
    )
    timeline = AnimationTimeline(scenes=[scene_timeline])
    manifest = VideoManifest(
        fps=30,
        total_frames=1,
        duration=1 / 30,
        scenes=[
            SceneManifest(
                scene_number=1,
                frame_start=1,
                frame_end=1,
                start_time=0,
                end_time=1 / 30,
                duration=1 / 30,
            )
        ],
    )
    audio = AudioMetadata(
        file_path="outputs/audio/narration.mp3",
        duration=1 / 30,
        sample_rate=24000,
        voice="en-US-AriaNeural",
        scenes=[
            SceneAudio(
                scene_number=1,
                duration=1 / 30,
                text="Narration",
                start_time=0,
                end_time=1 / 30,
            )
        ],
    )

    script_generator = MagicMock(spec=ScriptGenerator)
    asset_planner = MagicMock(spec=AssetPlanner)
    asset_manager = MagicMock(spec=AssetManager)
    scene_graph_builder = MagicMock(spec=SceneGraphBuilder)
    timeline_builder = MagicMock(spec=AnimationTimelineBuilder)
    synchronizer = MagicMock(spec=TimelineSynchronizer)
    frame_renderer = MagicMock(spec=FrameRenderer)
    audio_manager = MagicMock(spec=AudioManager)
    video_composer = MagicMock(spec=VideoComposer)

    def result(stage: str, value: object) -> object:
        events.append(stage)
        return value

    script_generator.generate.side_effect = lambda _topic: result(
        "generate_script", script
    )
    asset_planner.plan.side_effect = lambda _script: result(
        "plan_assets", asset_plan
    )
    asset_manager.resolve.side_effect = lambda _plan: result(
        "resolve_assets", resolved_assets
    )
    scene_graph_builder.build.side_effect = lambda *_args: result(
        "build_scene_graph", scenes
    )
    timeline_builder.build.side_effect = lambda _scenes: result(
        "build_timeline", timeline
    )
    synchronizer.synchronize.side_effect = lambda *_args, **_kwargs: result(
        "synchronize", manifest
    )

    def render_frame(*_args: object, **kwargs: object) -> None:
        events.append("render_frames")
        frames_directory = temp_dir / "frames"
        frames_directory.mkdir(parents=True, exist_ok=True)
        frame_start = int(kwargs["frame_start"])
        frame_count = int(kwargs["frame_count"])
        for frame_number in range(frame_start, frame_start + frame_count):
            frame_path = frames_directory / f"frame_{frame_number:06d}.png"
            frame_path.write_bytes(b"png")

    frame_renderer.render.side_effect = render_frame
    audio_manager.generate.side_effect = lambda _script: result(
        "generate_audio", audio
    )
    video_composer.compose.side_effect = lambda *_args: result(
        "compose_video", None
    )

    modules = {
        "script_generator": script_generator,
        "asset_planner": asset_planner,
        "asset_manager": asset_manager,
        "scene_graph_builder": scene_graph_builder,
        "timeline_builder": timeline_builder,
        "timeline_synchronizer": synchronizer,
        "frame_renderer": frame_renderer,
        "audio_manager": audio_manager,
        "video_composer": video_composer,
    }
    runner = PipelineRunner(**modules)
    return runner, events, modules


def test_pipeline_executes_modules_in_required_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every existing module executes once in the specified order."""

    runner, events, modules = mocked_runner(tmp_path, monkeypatch)
    output_dir = tmp_path / "outputs"

    output_file = runner.run("Test Topic", str(output_dir))

    assert events == [
        "generate_script",
        "generate_audio",
        "plan_assets",
        "resolve_assets",
        "build_scene_graph",
        "build_timeline",
        "synchronize",
        "render_frames",
        "compose_video",
    ]
    assert output_file == (output_dir / "final_video.mp4").as_posix()
    assert list(runner.stage_timings) == list(PipelineRunner.STAGE_NAMES)
    assert all(elapsed >= 0 for elapsed in runner.stage_timings.values())
    assert runner.last_execution_time >= 0
    modules["video_composer"].compose.assert_called_once()


def test_pipeline_stops_and_preserves_original_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed stage stops downstream work and re-raises the same error."""

    runner, events, modules = mocked_runner(tmp_path, monkeypatch)
    expected_error = RuntimeError("planning failed")

    def fail_planning(_script: object) -> None:
        events.append("plan_assets")
        raise expected_error

    modules["asset_planner"].plan.side_effect = fail_planning

    with pytest.raises(RuntimeError) as exc_info:
        runner.run("Test Topic", str(tmp_path / "outputs"))

    assert exc_info.value is expected_error
    assert events == ["generate_script", "generate_audio", "plan_assets"]
    modules["asset_manager"].resolve.assert_not_called()
    modules["video_composer"].compose.assert_not_called()
    assert list(runner.stage_timings) == [
        "Generate Script",
        "Generate Narration",
        "Plan Assets",
    ]
    assert runner.last_execution_time >= 0


def test_stage_and_total_execution_times_are_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A monotonic clock is sampled around every stage and the full run."""

    runner, _events, _modules = mocked_runner(tmp_path, monkeypatch)
    clock_values = iter(float(value) for value in range(20))
    monkeypatch.setattr(
        pipeline_runner,
        "perf_counter",
        lambda: next(clock_values),
    )

    runner.run("Timed Topic", str(tmp_path / "outputs"))

    assert runner.stage_timings == {
        stage_name: 1.0
        for stage_name in PipelineRunner.STAGE_NAMES
    }
    assert runner.last_execution_time == 19.0
