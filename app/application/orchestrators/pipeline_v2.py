"""End-to-end semantic V2 educational whiteboard pipeline."""

from collections.abc import Callable
from inspect import signature
from pathlib import Path
import re
from time import perf_counter
from typing import TypeVar, cast

from loguru import logger

from app.agents import (
    GeminiLessonPlanner,
    GeminiNarrationWriter,
    GeminiStoryboardPlanner,
    StructuredAgentError,
)
from app.application.ports import (
    AnimationPlanner,
    ConstraintLayoutEngine,
    LessonPlanner,
    NarrationWriter,
    PhraseAligner,
    QualityEvaluator,
    RenderEngine,
    SemanticAssetResolver,
    SpeechSynthesizer,
    StoryboardPlanner,
    TemplateLibrary,
    VideoCompositionEngine,
    VirtualCameraPlanner,
    VisualKnowledgeBase,
    VisualStateEngine,
)
from app.audio import ScenePhraseAligner, StoryboardSpeechSynthesizer
from app.camera import SemanticCameraPlanner
from app.config.settings import settings
from app.core.audio_manager import AudioManager
from app.core.video_composer import VideoComposer
from app.domain.generation import AudienceProfile, GenerationRequest, GenerationResult
from app.domain.lesson import LessonPlan
from app.domain.narration import NarrationPhrase, NarrationPlan
from app.domain.pedagogy import PedagogyPlan
from app.domain.quality import EvaluationDecision, QualityReport
from app.domain.rendering import CompositionJob, RenderJob
from app.domain.storyboard import Storyboard
from app.knowledge import InMemoryVisualKnowledgeBase
from app.layout import HierarchicalLayoutEngine
from app.motion import SemanticAnimationPlanner
from app.observability import ArtifactCheckpointStore
from app.planning import (
    AttentionPlanningEngine,
    ConceptGraphStoryboardBuilder,
    PedagogyRouter,
    SemanticAssetQueryPlanner,
    TemplateCompiler,
)
from app.planning.graph_semantics import normalize_lesson_structure
from app.quality import (
    CompositeQualityEvaluator,
    DeterministicQualityEvaluator,
    EducationalQualityEvaluator,
    QualityPolicy,
    RenderedFrameQualityEvaluator,
    VisualQualityEvaluator,
)
from app.rendering import ExistingVideoComposerAdapter, SemanticFrameRenderer
from app.semantic_assets import CatalogSemanticAssetResolver
from app.services.gemini_client import GeminiClient
from app.state import VisualStateTransitionEngine
from app.templates import TemplateRegistry, builtin_templates
from app.timeline import PhraseManifestBuilder
from app.utils.debug_recorder import DebugRecorder


StageT = TypeVar("StageT")


class QualityGateError(RuntimeError):
    """Raised when an artifact fails or exhausts its repair budget."""


class V2PipelineRunner:
    """Compile a topic into a semantically planned, verified MP4."""

    def __init__(
        self,
        lesson_planner: LessonPlanner | None = None,
        storyboard_planner: StoryboardPlanner | None = None,
        narration_writer: NarrationWriter | None = None,
        knowledge_base: VisualKnowledgeBase | None = None,
        template_library: TemplateLibrary | None = None,
        speech_synthesizer: SpeechSynthesizer | None = None,
        phrase_aligner: PhraseAligner | None = None,
        asset_resolver: SemanticAssetResolver | None = None,
        state_engine: VisualStateEngine | None = None,
        layout_engine: ConstraintLayoutEngine | None = None,
        animation_planner: AnimationPlanner | None = None,
        camera_planner: VirtualCameraPlanner | None = None,
        quality_evaluator: QualityEvaluator | None = None,
        render_engine: RenderEngine | None = None,
        composition_engine: VideoCompositionEngine | None = None,
        multimodal_evaluator: object | None = None,
        debug_recorder: DebugRecorder | None = None,
        max_repair_attempts: int | None = None,
        attention_planner: AttentionPlanningEngine | None = None,
        pedagogy_router: PedagogyRouter | None = None,
        template_compiler: TemplateCompiler | None = None,
        asset_query_planner: SemanticAssetQueryPlanner | None = None,
    ) -> None:
        """Configure replaceable ports and production defaults."""

        if max_repair_attempts is not None and max_repair_attempts < 0:
            raise ValueError("max_repair_attempts cannot be negative")
        self._debug = debug_recorder or DebugRecorder(
            enabled=getattr(settings, "DEBUG_ARTIFACTS", True),
            root_dir=getattr(settings, "DEBUG_DIR", Path("outputs/debug")),
        )
        client: GeminiClient | None = None
        if lesson_planner is None or storyboard_planner is None or narration_writer is None:
            client = GeminiClient()
        self._lesson_planner = lesson_planner or GeminiLessonPlanner(cast(GeminiClient, client))
        self._storyboard_planner = storyboard_planner or GeminiStoryboardPlanner(cast(GeminiClient, client))
        self._narration_writer = narration_writer or GeminiNarrationWriter(cast(GeminiClient, client))
        self._knowledge = knowledge_base or InMemoryVisualKnowledgeBase()
        self._templates = template_library or TemplateRegistry(builtin_templates())
        self._speech = speech_synthesizer or StoryboardSpeechSynthesizer(
            AudioManager(self._debug)
        )
        self._aligner = phrase_aligner or ScenePhraseAligner()
        self._assets = asset_resolver or CatalogSemanticAssetResolver()
        self._state = state_engine or VisualStateTransitionEngine()
        self._layout = layout_engine or HierarchicalLayoutEngine()
        self._motion = animation_planner or SemanticAnimationPlanner()
        self._camera = camera_planner or SemanticCameraPlanner()
        policy = QualityPolicy(
            maximum_repair_attempts=(
                max_repair_attempts
                if max_repair_attempts is not None
                else getattr(settings, "V2_MAX_REPAIR_ATTEMPTS", 2)
            )
        )
        self._quality = quality_evaluator or CompositeQualityEvaluator(
            [
                DeterministicQualityEvaluator(policy),
                EducationalQualityEvaluator(),
                VisualQualityEvaluator(),
            ]
        )
        self._renderer = render_engine or SemanticFrameRenderer(self._debug)
        self._composer = composition_engine or ExistingVideoComposerAdapter(
            VideoComposer(self._debug)
        )
        if multimodal_evaluator is None and getattr(
            settings,
            "V2_ENABLE_MULTIMODAL",
            False,
        ):
            from app.quality import GeminiVisionClient, MultimodalFrameEvaluator

            multimodal_evaluator = MultimodalFrameEvaluator(
                GeminiVisionClient()
            )
        self._multimodal = multimodal_evaluator
        self._max_repairs = policy.maximum_repair_attempts
        self._manifest_builder = PhraseManifestBuilder()
        self._attention = attention_planner or AttentionPlanningEngine()
        self._pedagogy = pedagogy_router or PedagogyRouter()
        self._template_compiler = template_compiler or TemplateCompiler()
        self._asset_query_planner = (
            asset_query_planner or SemanticAssetQueryPlanner()
        )
        self.stage_timings: dict[str, float] = {}
        self.last_quality_report: QualityReport | None = None
        self.last_execution_time = 0.0
        self._checkpoints: ArtifactCheckpointStore | None = None
        self._last_artifact_id: str | None = None

    @property
    def debug_run_dir(self) -> Path | None:
        """Return the active debug bundle location."""

        return self._debug.run_dir

    def run(
        self,
        request: GenerationRequest,
        output_dir: str = "outputs",
        resume: bool = False,
    ) -> GenerationResult:
        """Execute the complete V2 workflow with bounded targeted repair."""

        self.stage_timings.clear()
        self.last_quality_report = None
        self._debug.start_run(request.topic)
        self._checkpoints = ArtifactCheckpointStore(
            self._working_path(
                Path(output_dir) / "checkpoints" / request.run_id
            )
        )
        self._last_artifact_id = None
        started = perf_counter()
        error: Exception | None = None
        status = "failed"
        try:
            self._record("v2/request.json", request)
            lesson = (
                self._checkpoints.load("v2 lesson", LessonPlan)
                if resume and self._checkpoints.has("v2 lesson")
                else self._stage(
                    "Plan Lesson",
                    lambda: self._lesson_planner.plan(request),
                )
            )
            lesson = self._stage(
                "Normalize Lesson Structure",
                lambda: normalize_lesson_structure(lesson),
            )
            self._record("v2/lesson.json", lesson)
            pedagogy = self._stage(
                "Route Pedagogy",
                lambda: self._pedagogy.route(lesson, request.audience),
            )
            self._record("v2/pedagogy.json", pedagogy)
            strategies = self._stage(
                "Select Visual Strategies",
                lambda: self._knowledge.strategies_for(
                    lesson.concept_graph,
                    request.audience,
                ),
            )
            template_matches = self._stage(
                "Match Templates",
                lambda: self._match_templates(
                    lesson,
                    request.audience,
                ),
            )
            self._debug.write_json(
                "v2/strategy.json",
                {
                    "strategies": [item.model_dump(mode="json") for item in strategies],
                    "template_matches": [item.model_dump(mode="json") for item in template_matches],
                },
            )
            template_program = None
            if template_matches:
                template_program = self._stage(
                    "Compile Matched Template",
                    lambda: self._template_compiler.compile(
                        lesson,
                        strategies,
                        template_matches,
                        self._templates,
                        pedagogy,
                        request.target_duration,
                    ),
                )
            if template_program is not None:
                self._record("v2/template_program.json", template_program)
            resumed_storyboard = (
                resume and self._checkpoints.has("v2 storyboard accepted")
            )
            if resumed_storyboard:
                storyboard = self._checkpoints.load(
                    "v2 storyboard accepted",
                    Storyboard,
                )
            elif template_program is not None:
                storyboard = template_program.storyboard
            else:
                try:
                    storyboard = self._stage(
                        "Plan Storyboard",
                        lambda: self._plan_storyboard_with_pedagogy(
                            lesson,
                            strategies,
                            template_matches,
                            pedagogy,
                        ),
                        recoverable_exceptions=(StructuredAgentError,),
                    )
                    visual_intent = getattr(
                        self._storyboard_planner,
                        "last_intent",
                        None,
                    )
                    if visual_intent is not None:
                        self._record("v2/visual_intent.json", visual_intent)
                except StructuredAgentError:
                    logger.warning(
                        "Provider storyboard was unusable; compiling the validated "
                        "lesson concept graph deterministically."
                    )
                    storyboard = self._stage(
                        "Build Deterministic Storyboard Fallback",
                        lambda: ConceptGraphStoryboardBuilder().build(
                            lesson,
                            pedagogy=pedagogy,
                        ),
                    )
                else:
                    storyboard = self._stage(
                        "Ground Storyboard Concepts",
                        lambda: ConceptGraphStoryboardBuilder().ground(
                            storyboard,
                            lesson,
                            pedagogy,
                        ),
                    )

            if not resumed_storyboard:
                storyboard = self._stage(
                    "Plan Semantic Asset Queries",
                    lambda: self._asset_query_planner.enrich(
                        storyboard,
                        lesson,
                    ),
                )
                storyboard = self._stage(
                    "Plan Attention",
                    lambda: self._attention.enrich(storyboard),
                )

            for attempt in range(self._max_repairs + 1):
                storyboard = self._asset_query_planner.enrich(
                    storyboard,
                    lesson,
                )
                self._record(f"v2/storyboard/attempt_{attempt}.json", storyboard)
                storyboard_report = self._quality.evaluate(
                    "storyboard",
                    storyboard,
                    {
                        "concept_graph": lesson.concept_graph,
                        "lesson": lesson,
                        "audience": request.audience,
                        "pedagogy": pedagogy,
                    },
                )
                self._record(
                    f"v2/quality/storyboard_attempt_{attempt}.json",
                    storyboard_report,
                )
                if storyboard_report.decision is not EvaluationDecision.PASS:
                    storyboard = self._repair_or_raise(
                        lesson,
                        storyboard,
                        storyboard_report,
                        strategies,
                        template_matches,
                        attempt,
                        pedagogy,
                    )
                    continue
                self._record("v2/storyboard/accepted.json", storyboard)

                narration = self._stage(
                    "Write Narration",
                    lambda: self._write_narration_with_fallback(
                        storyboard,
                        request.audience,
                        pedagogy,
                        request.target_duration,
                    ),
                )
                self._validate_narration(storyboard, narration)
                self._record("v2/narration.json", narration)
                audio = self._stage(
                    "Generate Narration",
                    lambda: self._speech.synthesize(narration, request.voice),
                )
                alignment = self._stage(
                    "Align Phrases",
                    lambda: self._aligner.align(narration, audio),
                )
                self._record("v2/audio_alignment.json", alignment)
                assets = self._stage(
                    "Resolve Semantic Assets",
                    lambda: self._assets.resolve(storyboard),
                )
                self._record("v2/assets.json", assets)
                document = self._stage(
                    "Build Persistent States",
                    lambda: self._state.materialize(storyboard),
                )
                self._record("v2/visual_document.json", document)
                viewport = request.output.model_dump(
                    include={"width", "height"}
                )
                from app.domain.layout import Viewport

                layout = self._stage(
                    "Solve Layout",
                    lambda: self._layout.layout(
                        document,
                        assets,
                        Viewport(**viewport),
                    ),
                )
                self._record("v2/layout.json", layout)
                motion = self._stage(
                    "Plan Motion",
                    lambda: self._motion.plan(storyboard, layout, alignment),
                )
                camera = self._stage(
                    "Plan Camera",
                    lambda: self._camera.plan(storyboard, layout, alignment),
                )
                self._record("v2/motion.json", motion)
                self._record("v2/camera.json", camera)
                report = self._quality.evaluate(
                    "compiled_plan",
                    motion,
                    {
                        "concept_graph": lesson.concept_graph,
                        "storyboard": storyboard,
                        "assets": assets,
                        "document": document,
                        "layout": layout,
                        "motion": motion,
                        "camera": camera,
                        "narration": narration,
                        "alignment": alignment,
                        "audience": request.audience,
                        "pedagogy": pedagogy,
                    },
                )
                self.last_quality_report = report
                self._record(
                    f"v2/quality/compiled_attempt_{attempt}.json",
                    report,
                )
                if report.decision is not EvaluationDecision.PASS:
                    storyboard = self._repair_or_raise(
                        lesson,
                        storyboard,
                        report,
                        strategies,
                        template_matches,
                        attempt,
                        pedagogy,
                    )
                    continue

                manifest = self._stage(
                    "Build Manifest",
                    lambda: self._manifest_builder.build(
                        storyboard,
                        alignment,
                        fps=request.output.fps,
                    ),
                )
                self._record("v2/video_manifest.json", manifest)
                frames_folder = (
                    Path(settings.TEMP_DIR)
                    / "v2"
                    / "frames"
                    / request.run_id
                ).as_posix()
                frames = self._stage(
                    "Render Semantic Frames",
                    lambda: self._renderer.render(
                        RenderJob(
                            run_id=request.run_id,
                            document=document,
                            assets=assets,
                            layout=layout,
                            motion=motion,
                            camera=camera,
                            manifest=manifest,
                            output_folder=frames_folder,
                        )
                    ),
                )
                rendered_report = self._stage(
                    "Check Rendered Pixels",
                    lambda: RenderedFrameQualityEvaluator().evaluate(frames),
                )
                self._record(
                    f"v2/quality/rendered_attempt_{attempt}.json",
                    rendered_report,
                )
                if rendered_report.decision is not EvaluationDecision.PASS:
                    raise QualityGateError(
                        "rendered frame quality failed: "
                        f"{[item.code for item in rendered_report.findings]}"
                    )
                self.last_quality_report = rendered_report
                if self._multimodal is not None:
                    multimodal_report = self._stage(
                        "Evaluate Rendered Frames",
                        lambda: self._multimodal.evaluate_frames(
                            "rendered_frames",
                            [self._working_path(Path(path)) for path in frames.sample_paths],
                            storyboard,
                            narration,
                        ),
                    )
                    self._record(
                        f"v2/quality/multimodal_attempt_{attempt}.json",
                        multimodal_report,
                    )
                    if multimodal_report.decision is not EvaluationDecision.PASS:
                        storyboard = self._repair_or_raise(
                            lesson,
                            storyboard,
                            multimodal_report,
                            strategies,
                            template_matches,
                            attempt,
                            pedagogy,
                        )
                        continue
                    self.last_quality_report = multimodal_report

                output_file = (Path(output_dir) / "final_video.mp4").as_posix()
                video = self._stage(
                    "Compose Video",
                    lambda: self._composer.compose(
                        CompositionJob(
                            frames=frames,
                            manifest=manifest,
                            audio=audio,
                            output_file=output_file,
                        )
                    ),
                )
                result = GenerationResult(
                    run_id=request.run_id,
                    output_file=video.path,
                    duration=video.duration,
                    total_frames=video.total_frames,
                    quality_score=(
                        self.last_quality_report.overall_score
                        if self.last_quality_report is not None else 1.0
                    ),
                )
                self._record("v2/result.json", result)
                status = "complete"
                return result
            raise QualityGateError("V2 repair budget exhausted")
        except Exception as exc:
            error = exc
            raise
        finally:
            self.last_execution_time = perf_counter() - started
            self._debug.finish_run(
                status,
                self.last_execution_time,
                self.stage_timings,
                error,
            )

    def _repair_or_raise(
        self,
        lesson: LessonPlan,
        storyboard: Storyboard,
        report: QualityReport,
        strategies: list[object],
        templates: list[object],
        attempt: int,
        pedagogy: PedagogyPlan,
    ) -> Storyboard:
        """Invoke targeted storyboard repair within the configured budget."""

        if report.decision is EvaluationDecision.FAIL:
            raise QualityGateError(
                f"quality gate failed: {[item.code for item in report.findings]}"
            )
        if storyboard.document_id.startswith("compiled_"):
            logger.warning(
                "Authoritative template failed quality checks; replacing it "
                "with the provider-independent concept-graph compiler "
                "(findings={}).",
                [item.code for item in report.findings],
            )
            return self._attention.enrich(
                self._stage(
                    "Build Deterministic Storyboard Fallback",
                    lambda: ConceptGraphStoryboardBuilder().build(
                        lesson,
                        pedagogy=pedagogy,
                    ),
                )
            )
        finding_codes = {item.code for item in report.findings}
        if "semantic_coverage_low" in finding_codes:
            return self._attention.enrich(
                self._stage(
                    "Build Deterministic Storyboard Fallback",
                    lambda: ConceptGraphStoryboardBuilder().build(
                        lesson,
                        storyboard,
                        pedagogy,
                    ),
                )
            )
        if attempt >= self._max_repairs:
            raise QualityGateError(
                f"quality repair budget exhausted: "
                f"{[item.code for item in report.findings]}"
            )
        repair = getattr(self._storyboard_planner, "repair", None)
        if not callable(repair):
            raise QualityGateError(
                "storyboard planner does not support targeted repair"
            )
        try:
            repaired = self._stage(
                "Repair Storyboard",
                lambda: repair(
                    lesson,
                    storyboard,
                    report,
                    strategies,
                    templates,
                ),
                recoverable_exceptions=(StructuredAgentError,),
            )
        except StructuredAgentError:
            logger.warning(
                "Provider storyboard repair was unusable; compiling the lesson "
                "concept graph deterministically."
            )
            repaired = self._stage(
                "Build Deterministic Storyboard Fallback",
                lambda: ConceptGraphStoryboardBuilder().build(
                    lesson,
                    storyboard,
                    pedagogy,
                ),
            )
        return self._attention.enrich(repaired)

    @staticmethod
    def _validate_narration(storyboard: Storyboard, narration: object) -> None:
        """Require narration to cover only and every storyboard beat."""

        phrase_items = getattr(narration, "phrases", [])
        narrated = {phrase.beat_id for phrase in phrase_items}
        expected = {beat.beat_id for beat in storyboard.beats}
        if narrated != expected:
            raise ValueError(
                "narration phrases must cover every storyboard beat exactly by ID"
            )
        obligation_verbs = {
            "describe", "explain", "show", "display", "reveal", "state",
            "name", "identify", "highlight", "mark", "connect", "align",
        }
        leaked = [
            phrase.phrase_id
            for phrase in phrase_items
            if (
                any(
                    match.group(1).casefold() in obligation_verbs
                    for match in re.finditer(
                        r"(?:^|[.!?]\s+)([A-Za-z]+)",
                        phrase.text.strip(),
                    )
                )
                or "evidence:" in phrase.text.casefold()
            )
        ]
        if leaked:
            raise ValueError(
                "narration contains storyboard instructions instead of speech: "
                f"{leaked}"
            )

    def _write_narration_with_fallback(
        self,
        storyboard: Storyboard,
        audience: AudienceProfile,
        pedagogy: PedagogyPlan,
        target_duration: float | None = None,
    ) -> NarrationPlan:
        """Use storyboard phrase intents if provider narration is unusable."""

        fallback = self._storyboard_narration(storyboard)
        try:
            writer = getattr(
                self._narration_writer,
                "write_with_pedagogy",
                None,
            )
            narration = (
                self._call_narration_writer(
                    writer,
                    storyboard,
                    audience,
                    pedagogy,
                    target_duration,
                )
                if callable(writer)
                else self._narration_writer.write(storyboard, audience)
            )
            self._validate_narration(storyboard, narration)
        except (StructuredAgentError, ValueError, TypeError) as exc:
            logger.warning(
                "Provider narration was unusable; using storyboard phrase "
                "intents instead (error={}).",
                type(exc).__name__,
            )
            return self._expand_narration_for_duration(
                fallback,
                fallback,
                storyboard,
                target_duration,
            )
        if not self._narration_meets_minimum(narration, target_duration):
            logger.info(
                "Provider narration was shorter than the requested teaching "
                "depth; using the fuller deterministic narration."
            )
            return self._expand_narration_for_duration(
                narration,
                fallback,
                storyboard,
                target_duration,
            )
        return narration

    @staticmethod
    def _expand_narration_for_duration(
        primary: NarrationPlan,
        fallback: NarrationPlan,
        storyboard: Storyboard,
        target_duration: float | None,
    ) -> NarrationPlan:
        """Add factual accepted material when narration is measurably too thin."""

        if target_duration is None or target_duration < 20:
            return primary
        desired_words = round(target_duration * 2.45)
        fallback_by_beat = {
            phrase.beat_id: phrase.text
            for phrase in fallback.phrases
        }
        phrases = [phrase.model_copy(deep=True) for phrase in primary.phrases]

        def word_count() -> int:
            return sum(
                len(re.findall(r"\b[\w'-]+\b", phrase.text))
                for phrase in phrases
            )

        for phrase in phrases:
            if word_count() >= desired_words:
                break
            addition = fallback_by_beat.get(phrase.beat_id, "").strip()
            if addition and addition.casefold() not in phrase.text.casefold():
                phrase.text = f"{phrase.text.rstrip()} {addition}"
        if phrases and word_count() < desired_words:
            summary_text = " ".join(
                item.rstrip(".") + "."
                for item in storyboard.final_learning_summary
            ).strip()
            if summary_text:
                phrases[-1].text = (
                    f"{phrases[-1].text.rstrip()} {summary_text}"
                )
        return primary.model_copy(update={"phrases": phrases})

    @staticmethod
    def _storyboard_narration(storyboard: Storyboard) -> NarrationPlan:
        """Build complete listener-facing narration from accepted beat intents."""

        return NarrationPlan(
            title=storyboard.title,
            phrases=[
                NarrationPhrase(
                    phrase_id=f"phrase_{index:03d}",
                    beat_id=beat.beat_id,
                    text=V2PipelineRunner._listener_facing_text(
                        beat.phrase_intent
                    ),
                )
                for index, beat in enumerate(storyboard.beats, start=1)
            ],
        )

    @staticmethod
    def _listener_facing_text(text: str) -> str:
        """Rewrite leaked storyboard imperatives as natural spoken sentences."""

        replacements = {
            "describe": "We can examine",
            "explain": "We can understand",
            "show": "The visual presents",
            "display": "The visual presents",
            "reveal": "The visual reveals",
            "state": "The key point is",
            "name": "The key ideas are",
            "identify": "The important element is",
            "highlight": "The emphasis is on",
            "mark": "The emphasis is on",
            "connect": "The relationship links",
            "align": "The relationship aligns",
        }
        normalized = re.sub(
            r"\bevidence\s*:\s*",
            "For example, ",
            text.strip(),
            flags=re.IGNORECASE,
        )
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        spoken: list[str] = []
        for sentence in sentences:
            match = re.match(r"^([A-Za-z]+)\b[\s,:-]*(.*)$", sentence.strip())
            if match is None:
                continue
            replacement = replacements.get(match.group(1).casefold())
            if replacement is None:
                spoken.append(sentence.strip())
                continue
            remainder = match.group(2).strip()
            if remainder:
                spoken.append(f"{replacement} {remainder}")
        result = " ".join(spoken).strip()
        return result or "This visual reinforces the current idea."

    @staticmethod
    def _call_narration_writer(
        writer: Callable[..., NarrationPlan],
        storyboard: Storyboard,
        audience: AudienceProfile,
        pedagogy: PedagogyPlan,
        target_duration: float | None,
    ) -> NarrationPlan:
        """Pass duration when supported while preserving injected writers."""

        parameters = signature(writer).parameters.values()
        accepts_duration = any(
            item.name == "target_duration" or item.kind.name == "VAR_KEYWORD"
            for item in parameters
        )
        if accepts_duration:
            return writer(
                storyboard,
                audience,
                pedagogy,
                target_duration=target_duration,
            )
        return writer(storyboard, audience, pedagogy)

    @staticmethod
    def _narration_meets_minimum(
        narration: NarrationPlan,
        target_duration: float | None,
    ) -> bool:
        """Treat requested duration as a minimum depth preference, not a cap."""

        if target_duration is None or target_duration < 20:
            return True
        words = sum(
            len(re.findall(r"\b[\w'-]+\b", phrase.text))
            for phrase in narration.phrases
        )
        expected = target_duration * 2.30
        return words >= expected

    def _plan_storyboard_with_pedagogy(
        self,
        lesson: LessonPlan,
        strategies: list[object],
        templates: list[object],
        pedagogy: PedagogyPlan,
    ) -> Storyboard:
        """Pass routed rhetoric when supported without breaking injected ports."""

        planner = getattr(
            self._storyboard_planner,
            "plan_with_pedagogy",
            None,
        )
        if callable(planner):
            return planner(lesson, strategies, templates, pedagogy)
        return self._storyboard_planner.plan(lesson, strategies, templates)

    def _match_templates(
        self,
        lesson: LessonPlan,
        audience: AudienceProfile,
    ) -> list[object]:
        """Pass audience semantics while preserving older injected test ports."""

        matcher = self._templates.match
        parameters = signature(matcher).parameters.values()
        accepts_audience = any(
            item.name == "audience" or item.kind.name == "VAR_POSITIONAL"
            for item in parameters
        )
        if accepts_audience:
            return matcher(lesson.concept_graph, audience)
        return matcher(lesson.concept_graph)

    def _stage(
        self,
        name: str,
        action: Callable[[], StageT],
        recoverable_exceptions: tuple[type[Exception], ...] = (),
    ) -> StageT:
        """Run one timed stage while retaining its original exception."""

        logger.info("V2 pipeline stage started: {}.", name)
        started = perf_counter()
        try:
            result = action()
        except recoverable_exceptions as exc:
            self.stage_timings[name] = perf_counter() - started
            logger.warning(
                "V2 pipeline stage needs recovery: {} (error={}).",
                name,
                type(exc).__name__,
            )
            raise
        except Exception:
            self.stage_timings[name] = perf_counter() - started
            logger.exception("V2 pipeline stage failed: {}.", name)
            raise
        self.stage_timings[name] = perf_counter() - started
        logger.info(
            "V2 pipeline stage completed: {} ({:.3f}s).",
            name,
            self.stage_timings[name],
        )
        return result

    def _record(self, path: str, artifact: object) -> None:
        """Persist a Pydantic or plain artifact in the debug bundle."""

        dump = getattr(artifact, "model_dump", None)
        value = dump(mode="json") if callable(dump) else artifact
        self._debug.write_json(path, value)
        if self._checkpoints is not None and callable(dump):
            stage = path.removesuffix(".json").replace("/", " ")
            entry = self._checkpoints.write(
                stage,
                artifact,
                producer="V2PipelineRunner",
                parent_artifact_ids=(
                    [self._last_artifact_id]
                    if self._last_artifact_id is not None else []
                ),
            )
            self._last_artifact_id = entry.artifact_id

    @staticmethod
    def _working_path(path: Path) -> Path:
        """Resolve a configured path from project root."""

        from app.config.settings import PROJECT_ROOT

        return path if path.is_absolute() else PROJECT_ROOT / path
