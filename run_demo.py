"""Interactive command-line demo for the end-to-end video pipeline."""

import argparse
from pathlib import Path
from uuid import uuid4

from app.application.orchestrators import V2PipelineRunner
from app.config.settings import settings
from app.core.pipeline_runner import PipelineRunner
from app.domain.generation import (
    AudienceProfile,
    GenerationRequest,
    PipelineVersion,
)
from app.observability import ArtifactCheckpointStore
from app.utils.logger import initialize_logger


def main(argv: list[str] | None = None) -> None:
    """Prompt for a topic, execute the pipeline, and print its summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", help="Topic for a new generation run.")
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="Resume a V2 run from its latest compatible checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Pipeline output directory (default: outputs).",
    )
    args = parser.parse_args(argv)
    initialize_logger(settings.LOG_LEVEL)
    version = PipelineVersion(settings.PIPELINE_VERSION)
    if version is PipelineVersion.V2:
        if args.resume:
            checkpoint_store = ArtifactCheckpointStore(
                Path(args.output_dir) / "checkpoints" / args.resume
            )
            request = checkpoint_store.load("v2 request", GenerationRequest)
            topic = request.topic
        else:
            topic = (args.topic or input("Enter Topic: ")).strip()
            if not topic:
                parser.error("topic cannot be empty")
            request = GenerationRequest(
                run_id=f"v2_{uuid4().hex}",
                topic=topic,
                audience=AudienceProfile(
                    learning_goal=f"Understand {topic}",
                ),
            )
        print(f"Run ID: {request.run_id}")
        runner_v2 = V2PipelineRunner()
        result = runner_v2.run(
            request,
            output_dir=args.output_dir,
            resume=bool(args.resume),
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

    if args.resume:
        parser.error("--resume is supported only by the V2 pipeline")
    topic = (args.topic or input("Enter Topic: ")).strip()
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
