"""Offline and provider-backed execution of versioned benchmark suites."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Protocol

from app.benchmarking.metrics import collect_metrics
from app.benchmarking.models import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkReport,
    BenchmarkSuite,
)
from app.domain.assets import ResolvedAssetSet
from app.domain.generation import AudienceProfile, GenerationRequest, OutputProfile
from app.domain.layout import Viewport
from app.domain.lesson import ConceptEdge, ConceptGraph, ConceptNode, LessonPlan
from app.domain.narration import (
    AlignedAudio,
    NarrationPhrase,
    NarrationPlan,
    PhraseTiming,
)
from app.layout import HierarchicalLayoutEngine
from app.motion import SemanticAnimationPlanner
from app.camera import SemanticCameraPlanner
from app.planning import ConceptGraphStoryboardBuilder, PedagogyRouter
from app.quality import (
    CompositeQualityEvaluator,
    DeterministicQualityEvaluator,
    EducationalQualityEvaluator,
    VisualQualityEvaluator,
)
from app.state import VisualStateTransitionEngine


class CaseExecutor(Protocol):
    """Execute one benchmark case and return its complete result."""

    def execute(
        self,
        case: BenchmarkCase,
        config: BenchmarkConfig,
        output_dir: Path,
    ) -> BenchmarkCaseResult: ...


def load_benchmark_suite(path: str | Path) -> BenchmarkSuite:
    """Load and strictly validate a versioned suite file."""

    return BenchmarkSuite.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_report(path: str | Path) -> BenchmarkReport:
    """Load a prior benchmark report for comparison."""

    return BenchmarkReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


class OfflineCaseExecutor:
    """Exercise production deterministic planning without network or media I/O."""

    def execute(
        self,
        case: BenchmarkCase,
        config: BenchmarkConfig,
        output_dir: Path,
    ) -> BenchmarkCaseResult:
        """Build, lay out, animate, and evaluate one fixture lesson."""

        del output_dir
        started = perf_counter()
        timings: dict[str, float] = {}

        def stage(name: str, action: object) -> object:
            stage_started = perf_counter()
            result = action()  # type: ignore[operator]
            timings[name] = perf_counter() - stage_started
            return result

        lesson = stage("Build Fixture Lesson", lambda: self._lesson(case))
        audience = AudienceProfile(
            level=case.audience_level,
            learning_goal=case.learning_goal,
        )
        pedagogy = stage(
            "Route Pedagogy",
            lambda: PedagogyRouter().route(lesson, audience),
        )
        storyboard = stage(
            "Build Storyboard",
            lambda: ConceptGraphStoryboardBuilder().build(lesson, pedagogy=pedagogy),
        )
        narration = self._narration(storyboard)
        alignment = self._alignment(narration)
        document = stage(
            "Build Persistent States",
            lambda: VisualStateTransitionEngine().materialize(storyboard),
        )
        layout = stage(
            "Solve Layout",
            lambda: HierarchicalLayoutEngine().layout(
                document,
                ResolvedAssetSet(),
                Viewport(width=config.width, height=config.height),
            ),
        )
        motion = stage(
            "Plan Motion",
            lambda: SemanticAnimationPlanner().plan(storyboard, layout, alignment),
        )
        camera = stage(
            "Plan Camera",
            lambda: SemanticCameraPlanner().plan(storyboard, layout, alignment),
        )
        quality = stage(
            "Evaluate Quality",
            lambda: CompositeQualityEvaluator([
                DeterministicQualityEvaluator(),
                EducationalQualityEvaluator(),
                VisualQualityEvaluator(),
            ]).evaluate(
                "benchmark_plan",
                motion,
                {
                    "concept_graph": lesson.concept_graph,
                    "storyboard": storyboard,
                    "assets": ResolvedAssetSet(),
                    "document": document,
                    "layout": layout,
                    "motion": motion,
                    "camera": camera,
                    "narration": narration,
                    "alignment": alignment,
                    "audience": audience,
                    "pedagogy": pedagogy,
                },
            ),
        )
        operators = sorted({
            str(item.content.get("operator", item.kind))
            for beat in storyboard.beats
            for operation in beat.operations
            for raw in operation.arguments.get("objects", [])
            if isinstance(raw, dict)
            for item in self._objects(raw)
        })
        return BenchmarkCaseResult(
            case_id=case.case_id,
            status="completed",
            wall_time_seconds=perf_counter() - started,
            stage_timings_seconds=timings,
            metrics=collect_metrics(
                storyboard=storyboard,
                document=document,
                layout=layout,
                camera=camera,
                narration=narration,
                alignment=alignment,
                quality=quality,
                duration=alignment.duration,
            ),
            selected_operators=operators,
            quality_findings=[finding.code for finding in quality.findings],
        )

    @staticmethod
    def _lesson(case: BenchmarkCase) -> LessonPlan:
        concept_ids = [f"concept_{index:02d}" for index in range(len(case.expected_concepts))]
        nodes = [
            ConceptNode(
                concept_id=concept_id,
                label=label,
                definition=f"{label} is an essential part of {case.topic}.",
                importance=1.0 if index < 3 else 0.6,
                prerequisites=[concept_ids[index - 1]] if index else [],
                teaching_order=index,
                visual_affordances=["label", "relation"],
            )
            for index, (concept_id, label) in enumerate(
                zip(concept_ids, case.expected_concepts, strict=True)
            )
        ]
        edges = [
            ConceptEdge(
                edge_id=f"edge_{index:02d}",
                source_id=concept_ids[index],
                target_id=concept_ids[index + 1],
                relation=case.relation,
                label=case.expected_actions[min(index, len(case.expected_actions) - 1)]
                if case.expected_actions
                else case.relation.value.replace("_", " "),
            )
            for index in range(len(concept_ids) - 1)
        ]
        return LessonPlan(
            title=case.topic,
            summary=case.learning_goal,
            concept_graph=ConceptGraph(
                objectives=[case.learning_goal],
                nodes=nodes,
                edges=edges,
                teaching_sequence=concept_ids,
            ),
        )

    @staticmethod
    def _narration(storyboard: object) -> NarrationPlan:
        return NarrationPlan(
            title=storyboard.title,
            phrases=[
                NarrationPhrase(
                    phrase_id=f"phrase_{index:03d}",
                    beat_id=beat.beat_id,
                    text=beat.phrase_intent,
                )
                for index, beat in enumerate(storyboard.beats, start=1)
            ],
        )

    @staticmethod
    def _alignment(narration: NarrationPlan) -> AlignedAudio:
        timings = [
            PhraseTiming(
                phrase_id=phrase.phrase_id,
                beat_id=phrase.beat_id,
                audio_start=float(index),
                audio_end=float(index + 1),
                confidence=1.0,
            )
            for index, phrase in enumerate(narration.phrases)
        ]
        return AlignedAudio(
            audio_path="benchmark://silent",
            duration=float(len(timings)),
            sample_rate=24_000,
            phrases=timings,
        )

    @staticmethod
    def _objects(raw: dict[str, object]) -> list[object]:
        from app.domain.storyboard import VisualObjectSpec

        return VisualObjectSpec.model_validate(raw).flatten()


class ProviderCaseExecutor:
    """Run the full configured V2 pipeline and collect captured artifacts."""

    def execute(
        self,
        case: BenchmarkCase,
        config: BenchmarkConfig,
        output_dir: Path,
    ) -> BenchmarkCaseResult:
        """Execute a live provider case; callers must opt into this tier."""

        from app.application.orchestrators import V2PipelineRunner

        started = perf_counter()
        runner = V2PipelineRunner()
        run_id = f"benchmark_{case.case_id}_{case.seed + config.seed}"
        result = runner.run(
            GenerationRequest(
                run_id=run_id,
                topic=case.topic,
                target_duration=case.target_duration,
                audience=AudienceProfile(
                    level=case.audience_level,
                    learning_goal=case.learning_goal,
                ),
                output=OutputProfile(
                    width=config.width,
                    height=config.height,
                    fps=config.fps,
                    keep_frames=False,
                ),
            ),
            output_dir=(output_dir / case.case_id).as_posix(),
        )
        artifacts = runner.last_artifacts
        storyboard = artifacts.get("v2/storyboard/accepted.json")
        template_program = artifacts.get("v2/template_program.json")
        return BenchmarkCaseResult(
            case_id=case.case_id,
            status="completed",
            wall_time_seconds=perf_counter() - started,
            stage_timings_seconds=runner.stage_timings,
            metrics=collect_metrics(
                storyboard=storyboard,
                document=artifacts.get("v2/visual_document.json"),
                layout=artifacts.get("v2/layout.json"),
                camera=artifacts.get("v2/camera.json"),
                narration=artifacts.get("v2/narration.json"),
                alignment=artifacts.get("v2/audio_alignment.json"),
                quality=runner.last_quality_report,
                duration=result.duration,
            ),
            selected_templates=self._templates(artifacts),
            selected_operators=self._operators(storyboard),
            template_match_confidence=getattr(
                template_program, "match_confidence", None
            ),
            template_capability_evidence=list(getattr(
                template_program, "capability_evidence", []
            )),
            template_parameter_provenance=dict(getattr(
                template_program, "parameter_provenance", {}
            )),
            template_default_usage=list(getattr(
                template_program, "default_usage", []
            )),
            repair_attempts=sum(
                path.startswith("v2/storyboard/attempt_") for path in artifacts
            ) - 1,
            quality_findings=[
                finding.code for finding in runner.last_quality_report.findings
            ] if runner.last_quality_report is not None else [],
            artifact_paths={
                "video": result.output_file,
                **({"debug": runner.debug_run_dir.as_posix()} if runner.debug_run_dir else {}),
            },
        )

    @staticmethod
    def _templates(artifacts: dict[str, object]) -> list[str]:
        program = artifacts.get("v2/template_program.json")
        template_ids = getattr(program, "template_ids", None)
        if isinstance(template_ids, list):
            return [item for item in template_ids if isinstance(item, str)]
        template_id = getattr(program, "template_id", None)
        return [template_id] if isinstance(template_id, str) else []

    @staticmethod
    def _operators(storyboard: object) -> list[str]:
        from app.domain.storyboard import Storyboard

        if not isinstance(storyboard, Storyboard):
            return []
        values = {
            str(item.content.get("operator", item.kind))
            for beat in storyboard.beats
            for operation in beat.operations
            for raw in operation.arguments.get("objects", [])
            if isinstance(raw, dict)
            for item in [*OfflineCaseExecutor._objects(raw)]
        }
        return sorted(values)


class BenchmarkRunner:
    """Run a suite without allowing one failed case to hide other results."""

    def __init__(self, executor: CaseExecutor) -> None:
        self._executor = executor

    def run(
        self,
        suite: BenchmarkSuite,
        config: BenchmarkConfig,
        output_dir: str | Path,
        limit: int | None = None,
    ) -> BenchmarkReport:
        """Execute cases serially and persist auditable per-case records."""

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)
        cases: list[BenchmarkCaseResult] = []
        selected = suite.cases[:limit] if limit is not None else suite.cases
        for case in selected:
            case_started = perf_counter()
            try:
                result = self._executor.execute(case, config, root)
            except Exception as exc:
                result = BenchmarkCaseResult(
                    case_id=case.case_id,
                    status="failed",
                    wall_time_seconds=perf_counter() - case_started,
                    metrics=BenchmarkMetrics(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            cases.append(result)
            (root / f"{case.case_id}.json").write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )
        report = BenchmarkReport(
            suite_id=suite.suite_id,
            config=config,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor() or "unknown",
            },
            cases=cases,
        )
        (root / "report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return report
