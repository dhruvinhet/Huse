"""End-to-end orchestration for the whiteboard video pipeline."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import TypeVar

from loguru import logger

from app.config.settings import PROJECT_ROOT, settings
from app.core.animation_timeline_builder import AnimationTimelineBuilder
from app.core.asset_manager import AssetManager
from app.core.asset_planner import AssetPlanner
from app.core.audio_manager import AudioManager
from app.core.frame_renderer import FrameRenderer
from app.core.scene_graph_builder import SceneGraphBuilder
from app.core.script_generator import ScriptGenerator
from app.core.timeline_synchronizer import TimelineSynchronizer
from app.core.video_composer import VideoComposer
from app.models.animation import AnimationTimeline, SceneTimeline
from app.models.audio import AudioMetadata
from app.models.render import RenderScene
from app.models.script import Script
from app.models.video_manifest import SceneManifest, VideoManifest
from app.services.gemini_client import GeminiClient
from app.utils.debug_recorder import DebugRecorder


StageResult = TypeVar("StageResult")


class PipelineRunner:
    """Execute the existing pipeline modules in their required order."""

    FPS = 30
    OUTPUT_FILENAME = "final_video.mp4"
    STAGE_NAMES = (
        "Generate Script",
        "Generate Narration",
        "Plan Assets",
        "Resolve Assets",
        "Build Scene Graph",
        "Build Animation Timeline",
        "Synchronize Timeline",
        "Render Frames",
        "Compose Video",
    )

    def __init__(
        self,
        script_generator: ScriptGenerator | None = None,
        asset_planner: AssetPlanner | None = None,
        asset_manager: AssetManager | None = None,
        scene_graph_builder: SceneGraphBuilder | None = None,
        timeline_builder: AnimationTimelineBuilder | None = None,
        timeline_synchronizer: TimelineSynchronizer | None = None,
        frame_renderer: FrameRenderer | None = None,
        audio_manager: AudioManager | None = None,
        video_composer: VideoComposer | None = None,
        debug_recorder: DebugRecorder | None = None,
    ) -> None:
        """Initialize defaults while allowing every stage to be mocked."""

        debug_enabled = (
            script_generator is None
            and getattr(settings, "DEBUG_ARTIFACTS", False)
        )
        self._debug_recorder = debug_recorder or DebugRecorder(
            enabled=debug_enabled,
            root_dir=getattr(settings, "DEBUG_DIR", Path("outputs/debug")),
        )
        self._script_generator = script_generator
        self._asset_planner = asset_planner or AssetPlanner()
        self._asset_manager = asset_manager or AssetManager()
        self._scene_graph_builder = scene_graph_builder or SceneGraphBuilder()
        self._timeline_builder = timeline_builder or AnimationTimelineBuilder()
        self._timeline_synchronizer = (
            timeline_synchronizer or TimelineSynchronizer()
        )
        self._frame_renderer = frame_renderer or FrameRenderer(
            self._debug_recorder
        )
        self._audio_manager = audio_manager or AudioManager(
            self._debug_recorder
        )
        self._video_composer = video_composer or VideoComposer(
            self._debug_recorder
        )

        self._stage_timings: dict[str, float] = {}
        self.last_execution_time = 0.0
        self.last_script: Script | None = None
        self.last_manifest: VideoManifest | None = None
        self.last_audio: AudioMetadata | None = None
        self.last_output_file: str | None = None

    @property
    def stage_timings(self) -> dict[str, float]:
        """Return a copy of the most recent per-stage elapsed times."""

        return self._stage_timings.copy()

    @property
    def debug_run_dir(self) -> Path | None:
        """Return the active run's debug artifact directory."""

        return self._debug_recorder.run_dir

    def run(self, topic: str, output_dir: str = "outputs") -> str:
        """Run the complete topic-to-MP4 workflow and return its path."""

        self._reset_run_state()
        self._debug_recorder.start_run(topic)
        pipeline_started = perf_counter()
        pipeline_error: Exception | None = None
        status = "failed"

        try:
            script = self._run_stage(
                "Generate Script",
                lambda: self._get_script_generator().generate(topic),
            )
            self.last_script = script
            self._record_debug_model("pipeline/script.json", script)
            audio = self._run_stage(
                "Generate Narration",
                lambda: self._audio_manager.generate(script),
            )
            self.last_audio = audio
            self._record_debug_model("pipeline/audio.json", audio)

            asset_plan = self._run_stage(
                "Plan Assets",
                lambda: self._asset_planner.plan(script),
            )
            self._record_debug_model(
                "pipeline/asset_plan.json",
                asset_plan,
            )
            resolved_assets = self._run_stage(
                "Resolve Assets",
                lambda: self._asset_manager.resolve(asset_plan),
            )
            self._record_debug_model(
                "pipeline/resolved_assets.json",
                resolved_assets,
            )
            scenes = self._run_stage(
                "Build Scene Graph",
                lambda: self._scene_graph_builder.build(
                    script,
                    resolved_assets,
                ),
            )
            self._debug_recorder.write_json(
                "pipeline/scene_graph.json",
                [scene.model_dump(mode="json") for scene in scenes],
            )
            timeline = self._run_stage(
                "Build Animation Timeline",
                lambda: self._timeline_builder.build(scenes),
            )
            self._record_debug_model(
                "pipeline/animation_timeline.json",
                timeline,
            )
            manifest = self._run_stage(
                "Synchronize Timeline",
                lambda: self._timeline_synchronizer.synchronize(
                    timeline,
                    audio,
                    fps=self.FPS,
                ),
            )
            self.last_manifest = manifest
            self._record_debug_model(
                "pipeline/video_manifest.json",
                manifest,
            )
            self._record_scene_correlation(
                script,
                audio,
                scenes,
                manifest,
            )

            frames_folder = self._run_stage(
                "Render Frames",
                lambda: self._render_frames(scenes, timeline, manifest),
            )

            output_path = Path(output_dir) / self.OUTPUT_FILENAME
            output_file = output_path.as_posix()
            self._run_stage(
                "Compose Video",
                lambda: self._video_composer.compose(
                    manifest,
                    audio,
                    frames_folder,
                    output_file,
                ),
            )
            self.last_output_file = output_file
            status = "complete"
            return output_file
        except Exception as exc:
            pipeline_error = exc
            raise
        finally:
            self.last_execution_time = perf_counter() - pipeline_started
            self._debug_recorder.finish_run(
                status=status,
                execution_time=self.last_execution_time,
                stage_timings=self._stage_timings,
                error=pipeline_error,
            )
            logger.info(
                "Pipeline finished (execution_time={:.3f}s).",
                self.last_execution_time,
            )

    def _run_stage(
        self,
        stage_name: str,
        action: Callable[[], StageResult],
    ) -> StageResult:
        """Run and time one stage while preserving its original exception."""

        logger.info("Pipeline stage started: {}.", stage_name)
        stage_started = perf_counter()
        try:
            result = action()
        except Exception:
            elapsed = perf_counter() - stage_started
            self._stage_timings[stage_name] = elapsed
            logger.exception(
                "Pipeline stage failed: {} ({:.3f}s).",
                stage_name,
                elapsed,
            )
            raise

        elapsed = perf_counter() - stage_started
        self._stage_timings[stage_name] = elapsed
        logger.info(
            "Pipeline stage completed: {} ({:.3f}s).",
            stage_name,
            elapsed,
        )
        return result

    def _render_frames(
        self,
        scenes: list[RenderScene],
        timeline: AnimationTimeline,
        manifest: VideoManifest,
    ) -> str:
        """Render scene frames and collect a continuous manifest sequence."""

        timelines = {
            scene_timeline.scene_number: scene_timeline
            for scene_timeline in timeline.scenes
        }
        manifests = {
            scene_manifest.scene_number: scene_manifest
            for scene_manifest in manifest.scenes
        }
        configured_output = Path(settings.TEMP_DIR) / "frames"

        for scene_index, scene in enumerate(scenes):
            scene_timeline = self._required_timeline(scene, timelines)
            scene_manifest = self._required_manifest(scene, manifests)
            frame_count = (
                scene_manifest.frame_end - scene_manifest.frame_start + 1
            )
            self._frame_renderer.render(
                scene,
                scene_timeline,
                fps=manifest.fps,
                frame_start=scene_manifest.frame_start,
                frame_count=frame_count,
                clear_output=scene_index == 0,
            )

        output_directory = self._working_path(configured_output)
        generated_frames = len(list(output_directory.glob("frame_*.png")))
        if generated_frames != manifest.total_frames:
            raise RuntimeError(
                "Rendered frame count does not match the video manifest."
            )
        return configured_output.as_posix()

    @staticmethod
    def _required_timeline(
        scene: RenderScene,
        timelines: dict[int, SceneTimeline],
    ) -> SceneTimeline:
        """Return the timeline matching a render scene."""

        try:
            return timelines[scene.scene_number]
        except KeyError as exc:
            raise ValueError(
                f"Timeline is missing for scene {scene.scene_number}."
            ) from exc

    @staticmethod
    def _required_manifest(
        scene: RenderScene,
        manifests: dict[int, SceneManifest],
    ) -> SceneManifest:
        """Return the manifest entry matching a render scene."""

        try:
            return manifests[scene.scene_number]
        except KeyError as exc:
            raise ValueError(
                f"Manifest is missing for scene {scene.scene_number}."
            ) from exc

    def _get_script_generator(self) -> ScriptGenerator:
        """Lazily initialize Gemini only when a real pipeline run starts."""

        if self._script_generator is None:
            self._script_generator = ScriptGenerator(
                GeminiClient(),
                self._debug_recorder,
            )
        return self._script_generator

    def _record_debug_model(self, relative_path: str, model: object) -> None:
        """Serialize a Pydantic pipeline model into the debug bundle."""

        model_dump = getattr(model, "model_dump", None)
        value = model_dump(mode="json") if callable(model_dump) else model
        self._debug_recorder.write_json(relative_path, value)

    def _record_scene_correlation(
        self,
        script: Script,
        audio: AudioMetadata,
        scenes: list[RenderScene],
        manifest: VideoManifest,
    ) -> None:
        """Correlate narration, visual instructions, objects, and frame ranges."""

        if not self._debug_recorder.enabled:
            return
        audio_by_scene = {
            scene.scene_number: scene
            for scene in audio.scenes
        }
        render_by_scene = {
            scene.scene_number: scene
            for scene in scenes
        }
        manifest_by_scene = {
            scene.scene_number: scene
            for scene in manifest.scenes
        }
        correlated_scenes: list[dict[str, object]] = []
        for script_scene in script.scenes:
            scene_number = script_scene.scene_number
            scene_audio = audio_by_scene[scene_number]
            render_scene = render_by_scene[scene_number]
            scene_manifest = manifest_by_scene[scene_number]
            correlated_scenes.append(
                {
                    "scene_number": scene_number,
                    "narration": script_scene.narration,
                    "audio": {
                        "start_time": scene_audio.start_time,
                        "end_time": scene_audio.end_time,
                        "duration": scene_audio.duration,
                    },
                    "visual_instructions": [
                        visual.model_dump(mode="json")
                        for visual in script_scene.visuals
                    ],
                    "rendered_objects": [
                        obj.model_dump(mode="json")
                        for obj in render_scene.objects
                    ],
                    "frames": {
                        "start": scene_manifest.frame_start,
                        "end": scene_manifest.frame_end,
                        "count": (
                            scene_manifest.frame_end
                            - scene_manifest.frame_start
                            + 1
                        ),
                    },
                }
            )
        self._debug_recorder.write_json(
            "pipeline/scene_correlation.json",
            {"scenes": correlated_scenes},
        )

    def _reset_run_state(self) -> None:
        """Clear metadata from a previous run."""

        self._stage_timings.clear()
        self.last_execution_time = 0.0
        self.last_script = None
        self.last_manifest = None
        self.last_audio = None
        self.last_output_file = None

    @staticmethod
    def _working_path(configured_path: Path) -> Path:
        """Resolve a configured relative path from the project root."""

        if configured_path.is_absolute():
            return configured_path
        return PROJECT_ROOT / configured_path
