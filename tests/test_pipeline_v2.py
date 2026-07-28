"""End-to-end orchestration tests for the semantic V2 pipeline."""

from pathlib import Path

from app.application.orchestrators import V2PipelineRunner
from app.domain.generation import (
    AudienceProfile,
    GenerationRequest,
    OutputProfile,
)
from app.domain.lesson import LessonPlan
from app.domain.narration import NarrationPhrase, NarrationPlan
from app.domain.rendering import FrameSequence, VideoArtifact
from app.models.audio import AudioMetadata, SceneAudio
from app.utils.debug_recorder import DebugRecorder
from tests.test_domain_v2 import concept_graph, storyboard


class FakeLessonPlanner:
    """Return a fixed valid lesson."""

    def plan(self, request: GenerationRequest) -> LessonPlan:
        return LessonPlan(
            title=request.topic,
            summary="A transformation lesson",
            concept_graph=concept_graph(),
        )


class FakeStoryboardPlanner:
    """Return a fixed semantic storyboard."""

    def plan(self, lesson, strategies, templates):
        del lesson, strategies, templates
        return storyboard()


class FakeNarrationWriter:
    """Return one phrase per storyboard beat."""

    def write(self, board, audience):
        del audience
        return NarrationPlan(
            title=board.title,
            phrases=[
                NarrationPhrase(
                    phrase_id="phrase_1",
                    beat_id="beat_1",
                    text="First, the input enters the pipeline.",
                ),
                NarrationPhrase(
                    phrase_id="phrase_2",
                    beat_id="beat_2",
                    text="Now the output is emphasized.",
                ),
            ],
        )


class FakeKnowledge:
    """Return no optional strategy hints."""

    def strategies_for(self, graph, audience):
        del graph, audience
        return []


class FakeTemplates:
    """Return no optional template matches."""

    def match(self, graph):
        del graph
        return []

    def instantiate(self, template_id, parameters):
        raise AssertionError((template_id, parameters))


class FakeSpeech:
    """Return measured two-phrase audio metadata."""

    def synthesize(self, narration, voice):
        del narration, voice
        return AudioMetadata(
            file_path="outputs/audio/test.mp3",
            duration=4,
            sample_rate=24000,
            voice="test",
            scenes=[
                SceneAudio(
                    scene_number=1,
                    duration=2,
                    text="First",
                    start_time=0,
                    end_time=2,
                ),
                SceneAudio(
                    scene_number=2,
                    duration=2,
                    text="Second",
                    start_time=2,
                    end_time=4,
                ),
            ],
        )


class FakeRenderer:
    """Record the compiled render job without writing full frames."""

    def __init__(self) -> None:
        self.job = None

    def render(self, job):
        self.job = job
        return FrameSequence(
            folder=job.output_folder,
            total_frames=job.manifest.total_frames,
            fps=job.manifest.fps,
        )


class FakeComposer:
    """Return verified composition metadata without FFmpeg."""

    def __init__(self) -> None:
        self.job = None

    def compose(self, job):
        self.job = job
        return VideoArtifact(
            path=job.output_file,
            duration=job.manifest.duration,
            total_frames=job.manifest.total_frames,
            fps=job.manifest.fps,
            video_codec="h264",
            audio_codec="aac",
        )


def test_v2_pipeline_compiles_semantics_to_composition(tmp_path: Path) -> None:
    """All V2 semantic stages feed the preserved media boundary correctly."""

    from app.audio import ScenePhraseAligner

    renderer = FakeRenderer()
    composer = FakeComposer()
    runner = V2PipelineRunner(
        lesson_planner=FakeLessonPlanner(),
        storyboard_planner=FakeStoryboardPlanner(),
        narration_writer=FakeNarrationWriter(),
        knowledge_base=FakeKnowledge(),
        template_library=FakeTemplates(),
        speech_synthesizer=FakeSpeech(),
        phrase_aligner=ScenePhraseAligner(),
        render_engine=renderer,
        composition_engine=composer,
        debug_recorder=DebugRecorder(enabled=False, root_dir=tmp_path),
    )
    request = GenerationRequest(
        run_id="integration_run",
        topic="Transformation",
        target_duration=4,
        audience=AudienceProfile(learning_goal="Understand transformation"),
        output=OutputProfile(width=640, height=360, fps=2),
    )

    result = runner.run(request, tmp_path.as_posix())

    assert result.total_frames == 8
    assert result.quality_score == 1
    assert renderer.job.document.states[-1].object_states["output_box"].lifecycle.value == "emphasized"
    assert composer.job.audio.duration == 4
    assert runner.last_execution_time > 0
    assert "Plan Lesson" in runner.stage_timings
