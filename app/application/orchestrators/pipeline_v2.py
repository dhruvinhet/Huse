"""End-to-end semantic V2 educational whiteboard pipeline."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from inspect import signature
from pathlib import Path
import re
import json
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
from app.domain.assets import ResolvedAssetSet
from app.domain.camera import CameraPlan
from app.domain.generation import AudienceProfile, GenerationRequest, GenerationResult
from app.domain.layout import LayoutPlan
from app.domain.lesson import LessonPlan
from app.domain.motion import MotionPlan
from app.domain.narration import AlignedAudio, NarrationPhrase, NarrationPlan
from app.domain.pedagogy import PedagogyPlan
from app.domain.quality import EvaluationDecision, QualityReport
from app.domain.repair import RepairPlan, RepairStage
from app.domain.rendering import CompositionJob, FrameSequence, RenderJob
from app.domain.storyboard import ShotPlan, Storyboard
from app.domain.visual_document import VisualDocument
from app.knowledge import InMemoryVisualKnowledgeBase
from app.layout import HierarchicalLayoutEngine
from app.motion import SemanticAnimationPlanner
from app.novelty import NoveltyManager, StructuralFingerprintBuilder
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
    QualityRepairPlanner,
    QualityReviewPolicy,
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
from app.validation import DomainValidator, LessonGroundingValidator
from app.models.audio import AudioMetadata
from app.models.video_manifest import VideoManifest


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
        storyboard_reviewer: QualityEvaluator | None = None,
        review_policy: QualityReviewPolicy | None = None,
        repair_planner: QualityRepairPlanner | None = None,
        domain_validators: list[DomainValidator] | None = None,
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
        self._storyboard_reviewer = storyboard_reviewer
        self._review_policy = review_policy or QualityReviewPolicy()
        self._repair_planner = repair_planner or QualityRepairPlanner()
        self._grounding = LessonGroundingValidator(domain_validators)
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
        self.last_artifacts: dict[str, object] = {}
        self.repair_history: list[RepairPlan] = []
        self._pending_repair: RepairPlan | None = None
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
        self.last_artifacts.clear()
        self.repair_history.clear()
        self._pending_repair = None
        self._debug.start_run(request.topic)
        self._checkpoints = ArtifactCheckpointStore(
            self._working_path(
                Path(output_dir) / "checkpoints" / request.run_id
            )
        )
        self._last_artifact_id = None
        started = perf_counter()
        novelty = NoveltyManager(
            self._working_path(Path(output_dir) / "novelty" / "history.json")
        )
        current_fingerprint = None
        narration_cache: dict[str, NarrationPlan] = {}
        assets_cache: dict[str, object] = {}
        audio_cache: dict[str, object] = {}
        alignment_cache: dict[str, object] = {}
        document_cache: dict[str, object] = {}
        layout_cache: dict[str, object] = {}
        motion_cache: dict[str, object] = {}
        camera_cache: dict[str, object] = {}
        frames_cache: dict[str, object] = {}
        last_frames: object | None = None
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
            lesson, grounding_report = self._stage(
                "Validate Factual Grounding",
                lambda: self._grounding.validate(lesson, request.sources),
            )
            self._record("v2/grounding_report.json", grounding_report)
            self._grounding.raise_for_failure(grounding_report)
            self._record("v2/lesson.json", lesson)
            # These three planning products all depend only on the validated
            # lesson and audience.  Keep their contracts separate, but fan
            # them out so the lesson -> pedagogy -> storyboard path no longer
            # pays three avoidable deterministic round trips before the next
            # provider stage can begin.
            with ThreadPoolExecutor(
                max_workers=3,
                thread_name_prefix="v2-pre-storyboard",
            ) as executor:
                pedagogy_future = executor.submit(
                    self._stage,
                    "Route Pedagogy",
                    lambda: self._pedagogy.route(lesson, request.audience),
                )
                strategies_future = executor.submit(
                    self._stage,
                    "Select Visual Strategies",
                    lambda: self._knowledge.strategies_for(
                        lesson.concept_graph,
                        request.audience,
                    ),
                )
                template_matches_future = executor.submit(
                    self._stage,
                    "Match Templates",
                    lambda: self._match_templates(
                        lesson,
                        request.audience,
                    ),
                )
                pedagogy = pedagogy_future.result()
                strategies = strategies_future.result()
                template_matches = template_matches_future.result()
            template_matches = novelty.adjust_matches(
                template_matches, lesson.concept_graph
            )
            self._record("v2/pedagogy.json", pedagogy)
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

            storyboard = self._ensure_shot_plans(storyboard)

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
                active_repair = self._pending_repair
                invalidated = set(
                    active_repair.invalidated_stages
                    if active_repair is not None
                    else []
                )
                self._pending_repair = None
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
                storyboard_review_reasons = self._review_policy.storyboard_reasons(
                    storyboard_report,
                    template_matches,
                )
                self._debug.write_json(
                    f"v2/quality/storyboard_review_risk_{attempt}.json",
                    {"reasons": storyboard_review_reasons},
                )
                if self._storyboard_reviewer is not None and storyboard_review_reasons:
                    reviewer_report = self._stage(
                        "Review High-Risk Storyboard",
                        lambda: self._storyboard_reviewer.evaluate(
                            "storyboard",
                            storyboard,
                            {
                                "concept_graph": lesson.concept_graph,
                                "lesson": lesson,
                                "audience": request.audience,
                                "pedagogy": pedagogy,
                                "risk_reasons": storyboard_review_reasons,
                            },
                        ),
                    )
                    self._record(
                        f"v2/quality/storyboard_reviewer_attempt_{attempt}.json",
                        reviewer_report,
                    )
                    if reviewer_report.decision is not EvaluationDecision.PASS:
                        storyboard = self._repair_or_raise(
                            lesson,
                            storyboard,
                            reviewer_report,
                            strategies,
                            template_matches,
                            attempt,
                            pedagogy,
                        )
                        continue
                self._record("v2/storyboard/accepted.json", storyboard)

                narration_key = self._narration_cache_key(storyboard)
                storyboard_key = self._artifact_cache_key(storyboard)
                # Narration and asset resolution depend on the accepted
                # storyboard but not on one another.  Run them together so a
                # visual-only repair does not wait behind another model call.
                with ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="v2-post-storyboard",
                ) as executor:
                    narration_future = None
                    resumed_narration = (
                        self._resume_checkpoint("v2 narration", NarrationPlan)
                        if resume and attempt == 0
                        else None
                    )
                    if resumed_narration is not None:
                        narration = resumed_narration
                        narration_cache[narration_key] = narration.model_copy(deep=True)
                    elif (
                        narration_key in narration_cache
                        and RepairStage.NARRATION not in invalidated
                    ):
                        narration = narration_cache[narration_key].model_copy(
                            deep=True
                        )
                        logger.info(
                            "Reusing narration for unchanged storyboard (attempt={}).",
                            attempt,
                        )
                    else:
                        narration_future = executor.submit(
                            self._stage,
                            "Write Narration",
                            lambda: self._write_narration_with_fallback(
                                storyboard,
                                request.audience,
                                pedagogy,
                                request.target_duration,
                            ),
                        )
                    assets_future = None
                    resumed_assets = (
                        self._resume_checkpoint("v2 assets", ResolvedAssetSet)
                        if resume and attempt == 0
                        else None
                    )
                    if resumed_assets is not None:
                        assets = resumed_assets
                        assets_cache[storyboard_key] = assets
                    elif (
                        storyboard_key in assets_cache
                        and RepairStage.ASSETS not in invalidated
                    ):
                        assets = assets_cache[storyboard_key]
                        logger.info("Reusing semantic assets for unchanged storyboard.")
                    else:
                        assets_future = executor.submit(
                            self._stage,
                            "Resolve Semantic Assets",
                            lambda: self._assets.resolve(storyboard),
                        )
                    if narration_future is not None:
                        narration = narration_future.result()
                        narration_cache[narration_key] = narration.model_copy(
                            deep=True
                        )
                    if assets_future is not None:
                        assets = assets_future.result()
                        assets_cache[storyboard_key] = assets
                self._validate_narration(storyboard, narration)
                self._record("v2/narration.json", narration)
                self._record("v2/assets.json", assets)
                audio_key = self._artifact_cache_key(narration, request.voice)
                resumed_audio = (
                    self._resume_audio(output_dir, request.voice)
                    if resume and attempt == 0
                    else None
                )
                if resumed_audio is not None:
                    audio = resumed_audio
                    audio_cache[audio_key] = audio
                elif audio_key in audio_cache and RepairStage.AUDIO not in invalidated:
                    audio = audio_cache[audio_key]
                    logger.info("Reusing synthesized narration audio.")
                else:
                    audio = self._stage(
                        "Generate Narration",
                        lambda: self._speech.synthesize(narration, request.voice),
                    )
                    audio_cache[audio_key] = audio
                self._record("v2/audio.json", audio)
                alignment_key = self._artifact_cache_key(narration, audio)
                resumed_alignment = (
                    self._resume_checkpoint("v2 audio alignment", AlignedAudio)
                    if resume and attempt == 0
                    else None
                )
                if resumed_alignment is not None:
                    alignment = resumed_alignment
                    alignment_cache[alignment_key] = alignment
                elif (
                    alignment_key in alignment_cache
                    and RepairStage.AUDIO not in invalidated
                ):
                    alignment = alignment_cache[alignment_key]
                    logger.info("Reusing phrase alignment.")
                else:
                    alignment = self._stage(
                        "Align Phrases",
                        lambda: self._aligner.align(narration, audio),
                    )
                    alignment_cache[alignment_key] = alignment
                self._record("v2/audio_alignment.json", alignment)
                resumed_document = (
                    self._resume_checkpoint("v2 visual document", VisualDocument)
                    if resume and attempt == 0
                    else None
                )
                if resumed_document is not None:
                    document = resumed_document
                    document_cache[storyboard_key] = document
                elif (
                    storyboard_key in document_cache
                    and RepairStage.STATE not in invalidated
                ):
                    document = document_cache[storyboard_key]
                    logger.info("Reusing materialized visual states.")
                else:
                    document = self._stage(
                        "Build Persistent States",
                        lambda: self._state.materialize(storyboard),
                    )
                    document_cache[storyboard_key] = document
                self._record("v2/visual_document.json", document)
                viewport = request.output.model_dump(
                    include={"width", "height"}
                )
                from app.domain.layout import Viewport

                viewport_model = Viewport(**viewport)
                layout_key = self._artifact_cache_key(
                    document,
                    assets,
                    viewport_model,
                )
                resumed_layout = (
                    self._resume_checkpoint("v2 layout", LayoutPlan)
                    if resume and attempt == 0
                    else None
                )
                if resumed_layout is not None:
                    layout = resumed_layout
                    layout_cache[layout_key] = layout
                elif layout_key in layout_cache and RepairStage.LAYOUT not in invalidated:
                    layout = layout_cache[layout_key]
                    logger.info("Reusing solved layout.")
                else:
                    previous_layout = layout_cache.get(layout_key)
                    layout_repair = getattr(self._layout, "repair", None)
                    if (
                        active_repair is not None
                        and active_repair.owner_stage is RepairStage.LAYOUT
                        and previous_layout is not None
                        and callable(layout_repair)
                    ):
                        layout = self._stage(
                            "Repair Layout",
                            lambda: layout_repair(
                                document,
                                assets,
                                viewport_model,
                                previous_layout,
                                active_repair,
                            ),
                        )
                    else:
                        layout = self._stage(
                            "Solve Layout",
                            lambda: self._layout.layout(
                                document,
                                assets,
                                viewport_model,
                            ),
                        )
                    layout_cache[layout_key] = layout
                self._record("v2/layout.json", layout)
                motion_key = self._artifact_cache_key(storyboard, layout, alignment)
                resumed_motion = (
                    self._resume_checkpoint("v2 motion", MotionPlan)
                    if resume and attempt == 0
                    else None
                )
                if resumed_motion is not None:
                    motion = resumed_motion
                    motion_cache[motion_key] = motion
                elif motion_key in motion_cache and RepairStage.MOTION not in invalidated:
                    motion = motion_cache[motion_key]
                    logger.info("Reusing motion plan.")
                else:
                    previous_motion = motion_cache.get(motion_key)
                    motion_repair = getattr(self._motion, "repair", None)
                    if (
                        active_repair is not None
                        and active_repair.owner_stage is RepairStage.MOTION
                        and previous_motion is not None
                        and callable(motion_repair)
                    ):
                        motion = self._stage(
                            "Repair Motion",
                            lambda: motion_repair(
                                storyboard,
                                layout,
                                alignment,
                                previous_motion,
                                active_repair,
                            ),
                        )
                    else:
                        motion = self._stage(
                            "Plan Motion",
                            lambda: self._motion.plan(storyboard, layout, alignment),
                        )
                    motion_cache[motion_key] = motion
                camera_key = self._artifact_cache_key(storyboard, layout, alignment)
                resumed_camera = (
                    self._resume_checkpoint("v2 camera", CameraPlan)
                    if resume and attempt == 0
                    else None
                )
                if resumed_camera is not None:
                    camera = resumed_camera
                    camera_cache[camera_key] = camera
                elif camera_key in camera_cache and RepairStage.CAMERA not in invalidated:
                    camera = camera_cache[camera_key]
                    logger.info("Reusing camera plan.")
                else:
                    previous_camera = camera_cache.get(camera_key)
                    camera_repair = getattr(self._camera, "repair", None)
                    if (
                        active_repair is not None
                        and active_repair.owner_stage is RepairStage.CAMERA
                        and previous_camera is not None
                        and callable(camera_repair)
                    ):
                        camera = self._stage(
                            "Repair Camera",
                            lambda: camera_repair(
                                storyboard,
                                layout,
                                alignment,
                                previous_camera,
                                active_repair,
                            ),
                        )
                    else:
                        camera = self._stage(
                            "Plan Camera",
                            lambda: self._camera.plan(storyboard, layout, alignment),
                        )
                    camera_cache[camera_key] = camera
                self._record("v2/motion.json", motion)
                self._record("v2/camera.json", camera)
                current_fingerprint = StructuralFingerprintBuilder.build(
                    storyboard,
                    layout,
                    camera,
                    document,
                    template_program,
                )
                novelty_assessment = novelty.assess(current_fingerprint)
                self._record("v2/novelty.json", novelty_assessment)
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

                manifest = (
                    self._resume_checkpoint(
                        "v2 video manifest",
                        VideoManifest,
                        schema_version="1.0",
                    )
                    if resume and attempt == 0
                    else None
                )
                if manifest is None:
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
                render_job = RenderJob(
                    run_id=request.run_id,
                    document=document,
                    assets=assets,
                    layout=layout,
                    motion=motion,
                    camera=camera,
                    manifest=manifest,
                    output_folder=frames_folder,
                    keep_frames=request.output.keep_frames,
                )
                frames_key = self._artifact_cache_key(render_job)
                resumed_render_job = (
                    self._resume_checkpoint("v2 render job", RenderJob)
                    if resume and attempt == 0
                    else None
                )
                resumed_frames = None
                if resumed_render_job == render_job:
                    resumed_frames = self._resume_checkpoint(
                        "v2 frame sequence",
                        FrameSequence,
                        schema_version="1.0",
                    )
                self._record("v2/render_job.json", render_job)
                if resumed_frames is not None and self._frame_sequence_available(
                    resumed_frames
                ):
                    frames = resumed_frames
                    frames_cache[frames_key] = frames
                elif frames_key in frames_cache and RepairStage.RENDERER not in invalidated:
                    frames = frames_cache[frames_key]
                    logger.info("Reusing rendered frame sequence.")
                else:
                    previous_frames = frames_cache.get(frames_key) or last_frames
                    render_repair = getattr(self._renderer, "repair", None)
                    if (
                        active_repair is not None
                        and active_repair.owner_stage in {
                            RepairStage.LAYOUT,
                            RepairStage.MOTION,
                            RepairStage.CAMERA,
                            RepairStage.RENDERER,
                        }
                        and previous_frames is not None
                        and callable(render_repair)
                        and bool(
                            active_repair.beat_ids
                            or active_repair.frame_numbers
                        )
                    ):
                        frames = self._stage(
                            "Repair Rendered Frames",
                            lambda: render_repair(
                                render_job,
                                previous_frames,
                                active_repair,
                            ),
                        )
                    else:
                        frames = self._stage(
                            "Render Semantic Frames",
                            lambda: self._renderer.render(render_job),
                        )
                    frames_cache[frames_key] = frames
                last_frames = frames
                self._record("v2/frame_sequence.json", frames)
                rendered_report = self._stage(
                    "Check Rendered Pixels",
                    lambda: RenderedFrameQualityEvaluator().evaluate(
                        frames,
                        {
                            "storyboard": storyboard,
                            "document": document,
                            "layout": layout,
                            "camera": camera,
                            "manifest": manifest,
                        },
                    ),
                )
                self._record(
                    f"v2/quality/rendered_attempt_{attempt}.json",
                    rendered_report,
                )
                if rendered_report.decision is not EvaluationDecision.PASS:
                    storyboard = self._repair_or_raise(
                        lesson,
                        storyboard,
                        rendered_report,
                        strategies,
                        template_matches,
                        attempt,
                        pedagogy,
                    )
                    continue
                self.last_quality_report = rendered_report
                multimodal_reasons = self._review_policy.rendered_reasons(
                    report,
                    rendered_report,
                    assets,
                    template_matches,
                )
                self._debug.write_json(
                    f"v2/quality/multimodal_risk_{attempt}.json",
                    {"reasons": multimodal_reasons},
                )
                if self._multimodal is not None and multimodal_reasons:
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
                if current_fingerprint is not None:
                    novelty.record(current_fingerprint)
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

    def _resume_checkpoint(
        self,
        stage: str,
        model_type: type[StageT],
        *,
        schema_version: str = "2.0",
    ) -> StageT | None:
        """Load one compatible run checkpoint and safely fall back if stale."""

        if self._checkpoints is None or not self._checkpoints.has(
            stage,
            schema_version=schema_version,
        ):
            return None
        try:
            artifact = self._checkpoints.load(
                stage,
                model_type,
                schema_version=schema_version,
            )
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unusable checkpoint {}: {}", stage, exc)
            return None
        logger.info("Resumed checkpoint: {}.", stage)
        return artifact

    def _resume_audio(
        self,
        output_dir: str,
        voice: str,
    ) -> AudioMetadata | None:
        """Load run-scoped audio, with compatibility for older checkpoints."""

        checkpoint = self._resume_checkpoint(
            "v2 audio",
            AudioMetadata,
            schema_version="1.0",
        )
        if checkpoint is not None:
            return checkpoint if Path(checkpoint.file_path).is_file() else None

        # Older runs wrote the audio manifest beside the MP3 but did not add it
        # to the run checkpoint. Validate it against run-scoped alignment before
        # accepting the compatibility artifact.
        manifest_path = self._working_path(
            Path(output_dir) / "audio" / "audio_manifest.json"
        )
        if not manifest_path.is_file():
            return None
        try:
            audio = AudioMetadata.model_validate_json(manifest_path.read_bytes())
            alignment = self._resume_checkpoint(
                "v2 audio alignment",
                AlignedAudio,
            )
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring legacy audio checkpoint: {}", exc)
            return None
        if (
            alignment is None
            or audio.voice != voice
            or abs(audio.duration - alignment.duration) > 1e-6
            or audio.sample_rate != alignment.sample_rate
            or not Path(audio.file_path).is_file()
        ):
            return None
        logger.info("Resumed legacy narration audio manifest.")
        return audio

    @staticmethod
    def _frame_sequence_available(frames: FrameSequence) -> bool:
        """Require a retained stream or complete frame folder before reuse."""

        if frames.video_stream_path and Path(frames.video_stream_path).is_file():
            return all(Path(path).is_file() for path in frames.sample_paths)
        folder = Path(frames.folder)
        return folder.is_dir() and all(
            (folder / (frames.pattern % number)).is_file()
            for number in (1, frames.total_frames)
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
        repair_plan = self._repair_planner.plan(report)
        if any(
            prior.fingerprint == repair_plan.fingerprint
            for prior in self.repair_history
        ):
            raise QualityGateError(
                "quality repair made no progress: "
                f"fingerprint={repair_plan.fingerprint}, "
                f"stage={repair_plan.owner_stage.value}, "
                f"findings={repair_plan.finding_codes}"
            )
        self.repair_history.append(repair_plan)
        self._pending_repair = repair_plan
        self._record(
            f"v2/quality/repair_plan_{len(self.repair_history):02d}.json",
            repair_plan,
        )
        repair_targets = {
            item.repair_target
            for item in report.findings
            if item.repair_target
        }
        if repair_targets and repair_targets.isdisjoint({"storyboard"}):
            logger.info(
                "Quality findings target deterministic stages {}; keeping the "
                "accepted storyboard and recomputing those stages.",
                sorted(repair_targets),
            )
            return storyboard
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
        deterministic_rebuild_codes = {
            "semantic_coverage_low",
            "required_relations_not_visualized",
            "connector_endpoint_missing",
            "connector_relation_reversed",
            "connector_relation_ungrounded",
            "semantic_state_delta_missing",
            "visual_obligation_unrepresented",
            "visual_obligation_action_missing",
        }
        if finding_codes.intersection(deterministic_rebuild_codes):
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
        repair_beats = getattr(self._storyboard_planner, "repair_beats", None)
        if not callable(repair):
            raise QualityGateError(
                "storyboard planner does not support targeted repair"
            )
        try:
            if callable(repair_beats) and repair_plan.beat_ids:
                repaired = self._stage(
                    "Repair Storyboard Beats",
                    lambda: repair_beats(
                        lesson,
                        storyboard,
                        report,
                        strategies,
                        templates,
                        repair_plan.beat_ids,
                    ),
                    recoverable_exceptions=(StructuredAgentError,),
                )
            else:
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

    @staticmethod
    def _artifact_cache_key(*artifacts: object) -> str:
        """Hash immutable stage inputs for safe in-run artifact reuse."""

        payload = []
        for artifact in artifacts:
            dump = getattr(artifact, "model_dump", None)
            payload.append(
                dump(mode="json") if callable(dump) else artifact
            )
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    @staticmethod
    def _narration_cache_key(storyboard: Storyboard) -> str:
        """Key narration reuse to spoken content, not low-level geometry."""

        payload = [
            {
                "beat_id": beat.beat_id,
                "concept_ids": beat.concept_ids,
                "teaching_intent": beat.teaching_intent,
                "phrase_intent": beat.phrase_intent,
                "purpose": beat.purpose,
            }
            for beat in storyboard.beats
        ]
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

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

    @staticmethod
    def _ensure_shot_plans(storyboard: Storyboard) -> Storyboard:
        """Give provider- or checkpoint-loaded storyboards focused shot policies."""

        if all(beat.shot_plan is not None for beat in storyboard.beats):
            return storyboard
        return storyboard.model_copy(
            update={
                "beats": [
                    beat.model_copy(
                        update={
                            "shot_plan": beat.shot_plan
                            or ShotPlan.for_purpose(beat.purpose),
                        }
                    )
                    for beat in storyboard.beats
                ]
            }
        )

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

        self.last_artifacts[path] = artifact
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
