"""End-to-end orchestration tests for the semantic V2 pipeline."""

from pathlib import Path

import pytest

from app.application.orchestrators import V2PipelineRunner
from app.agents import StructuredAgentError
from app.domain.generation import (
    AudienceProfile,
    GenerationRequest,
    OutputProfile,
)
from app.domain.lesson import LessonPlan
from app.domain.narration import NarrationPhrase, NarrationPlan
from app.domain.rendering import FrameSequence, VideoArtifact
from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityFinding,
    QualityReport,
)
from app.domain.storyboard import Storyboard
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


class FailingStoryboardPlanner:
    """Simulate an NVIDIA storyboard that never satisfies its contract."""

    def plan(self, lesson, strategies, templates):
        del lesson, strategies, templates
        raise StructuredAgentError("invalid object lifecycle")


class ExplodingStoryboardPlanner:
    """Prove authoritative template compilation bypasses model planning."""

    def plan(self, lesson, strategies, templates):
        del lesson, strategies, templates
        raise AssertionError("storyboard model must not run for a compiled match")


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


class FailingNarrationWriter:
    """Simulate malformed provider narration after a valid storyboard."""

    def write(self, board, audience):
        del board, audience
        raise StructuredAgentError("invalid narration JSON")


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


class DynamicFakeSpeech:
    """Return one continuous audio scene for every narration phrase."""

    def synthesize(self, narration, voice):
        phrase_duration = 1.0
        return AudioMetadata(
            file_path="outputs/audio/test.mp3",
            duration=len(narration.phrases) * phrase_duration,
            sample_rate=24000,
            voice=voice,
            scenes=[
                SceneAudio(
                    scene_number=index,
                    duration=phrase_duration,
                    text=phrase.text,
                    start_time=(index - 1) * phrase_duration,
                    end_time=index * phrase_duration,
                )
                for index, phrase in enumerate(narration.phrases, start=1)
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


class RejectCompiledStoryboardQuality:
    """Force one authoritative-template rejection and accept its fallback."""

    def evaluate(self, artifact_id, artifact, context):
        board = artifact if isinstance(artifact, Storyboard) else context.get("storyboard")
        if isinstance(board, Storyboard) and board.document_id.startswith("compiled_"):
            return QualityReport(
                overall_score=0.5,
                scores={"forced": 0.5},
                findings=[QualityFinding(
                    code="forced_compiled_repair",
                    severity=FindingSeverity.ERROR,
                    artifact_id=artifact_id,
                    message="Exercise deterministic compiled-template fallback.",
                    repair_target="storyboard",
                )],
                decision=EvaluationDecision.REPAIR,
            )
        return QualityReport(
            overall_score=1,
            scores={"forced": 1},
            decision=EvaluationDecision.PASS,
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
        # Honest relation QA can replace the two-beat provider fixture with a
        # four-beat deterministic storyboard; synthesize per phrase so both
        # valid paths remain alignable.
        speech_synthesizer=DynamicFakeSpeech(),
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
    assert any(
        "output" in item.metadata.get("concept_ids", [])
        for state in renderer.job.document.states
        for item in state.object_states.values()
    )
    assert composer.job.audio.duration == 4
    assert runner.last_execution_time > 0
    assert "Plan Lesson" in runner.stage_timings


def test_layout_repair_reuses_every_successful_upstream_stage(
    tmp_path: Path,
) -> None:
    """A layout finding invalidates layout and downstream work only."""

    from app.audio import ScenePhraseAligner
    from app.camera import SemanticCameraPlanner
    from app.domain.assets import ResolvedAssetSet
    from app.domain.repair import RepairStage
    from app.layout import HierarchicalLayoutEngine
    from app.motion import SemanticAnimationPlanner
    from app.state import VisualStateTransitionEngine

    class CountingNarration(FakeNarrationWriter):
        calls = 0

        def write(self, board, audience):
            self.calls += 1
            return super().write(board, audience)

    class CountingAssets:
        calls = 0

        def resolve(self, board):
            del board
            self.calls += 1
            return ResolvedAssetSet()

    class CountingSpeech(DynamicFakeSpeech):
        calls = 0

        def synthesize(self, narration, voice):
            self.calls += 1
            return super().synthesize(narration, voice)

    class CountingState(VisualStateTransitionEngine):
        calls = 0

        def materialize(self, board):
            self.calls += 1
            return super().materialize(board)

    class CountingLayout(HierarchicalLayoutEngine):
        calls = 0

        def layout(self, document, assets, viewport):
            self.calls += 1
            return super().layout(document, assets, viewport)

    class CountingMotion(SemanticAnimationPlanner):
        calls = 0

        def plan(self, board, layout, alignment):
            self.calls += 1
            return super().plan(board, layout, alignment)

    class CountingCamera(SemanticCameraPlanner):
        calls = 0

        def plan(self, board, layout, alignment):
            self.calls += 1
            return super().plan(board, layout, alignment)

    class RepairLayoutOnce:
        compiled_calls = 0

        def evaluate(self, artifact_id, artifact, context):
            del artifact_id
            if isinstance(artifact, Storyboard):
                return QualityReport(
                    overall_score=1,
                    scores={"storyboard": 1},
                    decision=EvaluationDecision.PASS,
                )
            self.compiled_calls += 1
            if self.compiled_calls == 1:
                return QualityReport(
                    overall_score=0.5,
                    scores={"layout": 0.5},
                    findings=[QualityFinding(
                        code="object_clipped",
                        severity=FindingSeverity.ERROR,
                        artifact_id="layout",
                        message="One object is outside the viewport.",
                        repair_target="layout",
                        repair_scope="object",
                        object_ids=["output_box"],
                        patch_paths=["/layout"],
                    )],
                    decision=EvaluationDecision.REPAIR,
                )
            return QualityReport(
                overall_score=1,
                scores={"layout": 1},
                decision=EvaluationDecision.PASS,
            )

    narration = CountingNarration()
    assets = CountingAssets()
    speech = CountingSpeech()
    state = CountingState()
    layout = CountingLayout()
    motion = CountingMotion()
    camera = CountingCamera()
    renderer = FakeRenderer()
    runner = V2PipelineRunner(
        lesson_planner=FakeLessonPlanner(),
        storyboard_planner=FakeStoryboardPlanner(),
        narration_writer=narration,
        knowledge_base=FakeKnowledge(),
        template_library=FakeTemplates(),
        speech_synthesizer=speech,
        phrase_aligner=ScenePhraseAligner(),
        asset_resolver=assets,
        state_engine=state,
        layout_engine=layout,
        animation_planner=motion,
        camera_planner=camera,
        quality_evaluator=RepairLayoutOnce(),
        render_engine=renderer,
        composition_engine=FakeComposer(),
        debug_recorder=DebugRecorder(enabled=False, root_dir=tmp_path),
    )
    request = GenerationRequest(
        run_id="layout_local_repair",
        topic="Transformation",
        target_duration=4,
        audience=AudienceProfile(learning_goal="Understand transformation"),
        output=OutputProfile(width=640, height=360, fps=2),
    )

    runner.run(request, tmp_path.as_posix())

    assert narration.calls == 1
    assert assets.calls == 1
    assert speech.calls == 1
    assert state.calls == 1
    assert layout.calls == 2
    assert motion.calls == 2
    assert camera.calls == 2
    assert runner.repair_history[0].owner_stage is RepairStage.LAYOUT


def test_repeated_identical_repair_stops_with_no_progress_diagnostic(
    tmp_path: Path,
) -> None:
    """An unchanged finding cannot consume the repair budget blindly."""

    from app.application.orchestrators.pipeline_v2 import QualityGateError
    from app.planning import PedagogyRouter

    runner = V2PipelineRunner(
        lesson_planner=FakeLessonPlanner(),
        storyboard_planner=FakeStoryboardPlanner(),
        narration_writer=FakeNarrationWriter(),
        knowledge_base=FakeKnowledge(),
        template_library=FakeTemplates(),
        speech_synthesizer=DynamicFakeSpeech(),
        render_engine=FakeRenderer(),
        composition_engine=FakeComposer(),
        debug_recorder=DebugRecorder(enabled=False, root_dir=tmp_path),
    )
    request = GenerationRequest(
        run_id="no_progress",
        topic="Transformation",
        audience=AudienceProfile(learning_goal="Understand transformation"),
    )
    lesson = FakeLessonPlanner().plan(request)
    board = storyboard()
    pedagogy = PedagogyRouter().route(lesson, request.audience)
    report = QualityReport(
        overall_score=0.5,
        scores={"layout": 0.5},
        findings=[QualityFinding(
            code="object_clipped",
            severity=FindingSeverity.ERROR,
            artifact_id="layout",
            message="The same object remains clipped.",
            repair_target="layout",
            repair_scope="object",
            object_ids=["output_box"],
            measured_value=12,
            required_value=0,
            patch_paths=["/layout"],
        )],
        decision=EvaluationDecision.REPAIR,
    )

    assert runner._repair_or_raise(
        lesson,
        board,
        report,
        [],
        [],
        0,
        pedagogy,
    ) is board
    with pytest.raises(QualityGateError, match="repair made no progress"):
        runner._repair_or_raise(
            lesson,
            board,
            report,
            [],
            [],
            1,
            pedagogy,
        )


def test_storyboard_repair_dispatches_only_implicated_beats(tmp_path: Path) -> None:
    """Beat-scoped findings use the compact repair contract when available."""

    from app.planning import PedagogyRouter

    class BeatRepairPlanner(FakeStoryboardPlanner):
        requested: list[str] = []

        def repair(self, lesson, previous, report, strategies, templates):
            raise AssertionError("full storyboard repair must not run")

        def repair_beats(
            self,
            lesson,
            previous,
            report,
            strategies,
            templates,
            beat_ids,
        ):
            del lesson, report, strategies, templates
            self.requested = list(beat_ids)
            beats = [
                beat.model_copy(update={"phrase_intent": "Repaired evidence."})
                if beat.beat_id in beat_ids else beat
                for beat in previous.beats
            ]
            return previous.model_copy(update={"beats": beats})

    planner = BeatRepairPlanner()
    runner = V2PipelineRunner(
        lesson_planner=FakeLessonPlanner(),
        storyboard_planner=planner,
        narration_writer=FakeNarrationWriter(),
        knowledge_base=FakeKnowledge(),
        template_library=FakeTemplates(),
        speech_synthesizer=DynamicFakeSpeech(),
        render_engine=FakeRenderer(),
        composition_engine=FakeComposer(),
        debug_recorder=DebugRecorder(enabled=False, root_dir=tmp_path),
    )
    request = GenerationRequest(
        run_id="beat_repair",
        topic="Transformation",
        audience=AudienceProfile(learning_goal="Understand transformation"),
    )
    lesson = FakeLessonPlanner().plan(request)
    board = storyboard()
    report = QualityReport(
        overall_score=0.7,
        scores={"evidence": 0.7},
        findings=[QualityFinding(
            code="weak_focal_evidence",
            severity=FindingSeverity.ERROR,
            artifact_id="storyboard",
            message="The second beat needs concrete evidence.",
            repair_target="storyboard",
            repair_scope="beat",
            beat_id="beat_2",
            patch_paths=["/beats/1"],
        )],
        decision=EvaluationDecision.REPAIR,
    )

    repaired = runner._repair_or_raise(
        lesson,
        board,
        report,
        [],
        [],
        0,
        PedagogyRouter().route(lesson, request.audience),
    )

    assert planner.requested == ["beat_2"]
    assert repaired.beats[0].phrase_intent == board.beats[0].phrase_intent
    assert repaired.beats[1].phrase_intent == "Repaired evidence."


def test_v2_pipeline_survives_invalid_storyboard_and_narration(
    tmp_path: Path,
) -> None:
    """Provider formatting failures fall back without losing lesson concepts."""

    from app.audio import ScenePhraseAligner

    renderer = FakeRenderer()
    runner = V2PipelineRunner(
        lesson_planner=FakeLessonPlanner(),
        storyboard_planner=FailingStoryboardPlanner(),
        narration_writer=FailingNarrationWriter(),
        knowledge_base=FakeKnowledge(),
        template_library=FakeTemplates(),
        speech_synthesizer=DynamicFakeSpeech(),
        phrase_aligner=ScenePhraseAligner(),
        render_engine=renderer,
        composition_engine=FakeComposer(),
        debug_recorder=DebugRecorder(enabled=False, root_dir=tmp_path),
    )
    request = GenerationRequest(
        run_id="provider_fallback",
        topic="Transformation",
        audience=AudienceProfile(learning_goal="Understand transformation"),
        output=OutputProfile(width=640, height=360, fps=2),
    )

    result = runner.run(request, tmp_path.as_posix())

    assert result.total_frames > 0
    assert "Build Deterministic Storyboard Fallback" in runner.stage_timings
    assert renderer.job.document.states[-1].beat_id == "beat_summary"


def test_v2_pipeline_accepts_compiled_template_with_typed_action_without_model(
    tmp_path: Path,
) -> None:
    """A reviewed match bypasses the model and satisfies action QA explicitly."""

    from app.audio import ScenePhraseAligner
    from app.templates import TemplateRegistry
    from app.templates.builtins import builtin_templates

    renderer = FakeRenderer()
    runner = V2PipelineRunner(
        lesson_planner=FakeLessonPlanner(),
        storyboard_planner=ExplodingStoryboardPlanner(),
        narration_writer=FailingNarrationWriter(),
        knowledge_base=FakeKnowledge(),
        template_library=TemplateRegistry(builtin_templates()),
        speech_synthesizer=DynamicFakeSpeech(),
        phrase_aligner=ScenePhraseAligner(),
        render_engine=renderer,
        composition_engine=FakeComposer(),
        debug_recorder=DebugRecorder(enabled=False, root_dir=tmp_path),
    )
    request = GenerationRequest(
        run_id="compiled_template",
        topic="Transformation",
        audience=AudienceProfile(learning_goal="Understand transformation"),
        output=OutputProfile(width=640, height=360, fps=2),
    )

    result = runner.run(request, tmp_path.as_posix())

    assert result.total_frames > 0
    assert "Compile Matched Template" in runner.stage_timings
    assert "Plan Storyboard" not in runner.stage_timings
    assert renderer.job.document.document_id.startswith("compiled_")
    assert "Build Deterministic Storyboard Fallback" not in runner.stage_timings
    compiled_report = runner.last_artifacts[
        "v2/quality/compiled_attempt_0.json"
    ]
    assert "semantic_state_delta_missing" not in {
        finding.code for finding in compiled_report.findings
    }
    assert "semantic_action_missing" not in {
        finding.code for finding in compiled_report.findings
    }
    assert len(renderer.job.document.states) == 4


def test_storyboard_narration_repairs_instruction_like_phrase_intents() -> None:
    """Deterministic fallback speech cannot fail the final narration boundary."""

    board = storyboard().model_copy(
        update={
            "beats": [
                beat.model_copy(
                    update={
                        "phrase_intent": (
                            "Explain how force changes motion. "
                            "Evidence: acceleration increases with force."
                        )
                    }
                )
                for beat in storyboard().beats
            ]
        },
        deep=True,
    )

    narration = V2PipelineRunner._storyboard_narration(board)
    V2PipelineRunner._validate_narration(board, narration)

    assert narration.phrases[0].text.startswith("We can understand")
    assert "Evidence:" not in narration.phrases[0].text


def test_compiled_template_quality_failure_uses_deterministic_fallback(
    tmp_path: Path,
) -> None:
    """A rejected authoritative template cannot terminate the whole run."""

    from app.audio import ScenePhraseAligner
    from app.templates import TemplateRegistry
    from app.templates.builtins import builtin_templates

    renderer = FakeRenderer()
    runner = V2PipelineRunner(
        lesson_planner=FakeLessonPlanner(),
        storyboard_planner=ExplodingStoryboardPlanner(),
        narration_writer=FailingNarrationWriter(),
        knowledge_base=FakeKnowledge(),
        template_library=TemplateRegistry(builtin_templates()),
        speech_synthesizer=DynamicFakeSpeech(),
        phrase_aligner=ScenePhraseAligner(),
        quality_evaluator=RejectCompiledStoryboardQuality(),
        render_engine=renderer,
        composition_engine=FakeComposer(),
        debug_recorder=DebugRecorder(enabled=False, root_dir=tmp_path),
    )
    request = GenerationRequest(
        run_id="compiled_fallback",
        topic="Transformation",
        audience=AudienceProfile(learning_goal="Understand transformation"),
        output=OutputProfile(width=640, height=360, fps=2),
    )

    result = runner.run(request, tmp_path.as_posix())

    assert result.total_frames > 0
    assert not renderer.job.document.document_id.startswith("compiled_")
    assert "Build Deterministic Storyboard Fallback" in runner.stage_timings
