"""Interactive command-line demo for the end-to-end video pipeline."""

from pathlib import Path

from app.application.orchestrators import V2PipelineRunner
from app.config.settings import settings
from app.core.pipeline_runner import PipelineRunner
from app.domain.generation import (
    AudienceProfile,
    GenerationRequest,
    PipelineVersion,
)
from app.utils.logger import initialize_logger
from uuid import uuid4


def main() -> None:
    """Prompt for a topic, execute the pipeline, and print its summary."""

    initialize_logger(settings.LOG_LEVEL)
    topic = input("Enter Topic: ").strip()
    version = PipelineVersion(settings.PIPELINE_VERSION)
    if version is PipelineVersion.V2:
        runner_v2 = V2PipelineRunner()
        result = runner_v2.run(
            GenerationRequest(
                run_id=f"v2_{uuid4().hex}",
                topic=topic,
                audience=AudienceProfile(
                    learning_goal=f"Understand {topic}",
                ),
            )
        )
        video_location = result.output_file
        print(f"Video location: {video_location}")
        print(f"Execution time: {runner_v2.last_execution_time:.2f} seconds")
        print()
        print("========== PIPELINE SUMMARY ==========")
        print(f"Pipeline: {version.value}")
        print(f"Topic: {topic}")
        print(f"Frames Generated: {result.total_frames}")
        print("Audio Generated: YES")
        print("Video Generated: YES")
        print(f"Quality Score: {result.quality_score:.3f}")
        print(f"Output Folder: {Path(video_location).parent.as_posix()}")
        print(f"Debug Folder: {runner_v2.debug_run_dir or 'DISABLED'}")
        print(f"Execution Time: {runner_v2.last_execution_time:.2f} seconds")
        print("=====================================")
        return

    runner = PipelineRunner()
    video_location = runner.run(topic)

    script = runner.last_script
    manifest = runner.last_manifest
    audio = runner.last_audio
    output_folder = Path(video_location).parent.as_posix()

    print(f"Video location: {video_location}")
    print(f"Execution time: {runner.last_execution_time:.2f} seconds")
    print()
    print("========== PIPELINE SUMMARY ==========")
    print(f"Pipeline: {version.value}")
    print(f"Topic: {topic}")
    print(f"Scenes: {len(script.scenes) if script else 0}")
    print(f"Frames Generated: {manifest.total_frames if manifest else 0}")
    print(f"Audio Generated: {'YES' if audio else 'NO'}")
    print(f"Video Generated: {'YES' if runner.last_output_file else 'NO'}")
    print(f"Output Folder: {output_folder}")
    print(f"Debug Folder: {runner.debug_run_dir or 'DISABLED'}")
    print(f"Execution Time: {runner.last_execution_time:.2f} seconds")
    print("=====================================")


if __name__ == "__main__":
    main()
