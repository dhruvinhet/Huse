"""Deterministic preflight evaluation for semantic pipeline artifacts."""

from app.domain.assets import ResolvedAssetSet
from app.domain.generation import AudienceProfile
from app.domain.layout import LaidOutNode, LayoutPlan
from app.domain.lesson import ConceptGraph
from app.domain.motion import MotionPlan
from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityFinding,
    QualityReport,
)
from app.domain.storyboard import Storyboard
from app.domain.visual_document import VisualDocument
from app.quality.policy import QualityPolicy
from app.planning.density import VisualDensityPlanner


class DeterministicQualityEvaluator:
    """Score semantic coverage, assets, layout, timing, and state integrity."""

    def __init__(self, policy: QualityPolicy | None = None) -> None:
        """Use the supplied policy or production-safe initial defaults."""

        self._policy = policy or QualityPolicy()

    def evaluate(
        self,
        artifact_id: str,
        artifact: object,
        context: dict[str, object],
    ) -> QualityReport:
        """Evaluate available artifacts and return actionable findings."""

        graph = self._typed(context.get("concept_graph"), ConceptGraph)
        storyboard = (
            artifact if isinstance(artifact, Storyboard)
            else self._typed(context.get("storyboard"), Storyboard)
        )
        assets = (
            artifact if isinstance(artifact, ResolvedAssetSet)
            else self._typed(context.get("assets"), ResolvedAssetSet)
        )
        layout = (
            artifact if isinstance(artifact, LayoutPlan)
            else self._typed(context.get("layout"), LayoutPlan)
        )
        motion = (
            artifact if isinstance(artifact, MotionPlan)
            else self._typed(context.get("motion"), MotionPlan)
        )
        document = (
            artifact if isinstance(artifact, VisualDocument)
            else self._typed(context.get("document"), VisualDocument)
        )
        audience = self._typed(context.get("audience"), AudienceProfile)

        findings: list[QualityFinding] = []
        scores: dict[str, float] = {}
        scores["semantic_coverage"] = self._concept_coverage(
            artifact_id,
            graph,
            storyboard,
            findings,
        )
        scores["asset_readiness"] = self._asset_readiness(
            artifact_id,
            assets,
            findings,
        )
        scores["layout"] = self._layout_quality(
            artifact_id,
            layout,
            findings,
        )
        scores["animation_density"] = self._motion_quality(
            artifact_id,
            motion,
            findings,
        )
        scores["state_integrity"] = 1.0 if document is not None else 0.5
        scores["readability"] = self._readability(layout)
        scores["diagram_correctness"] = 1.0
        scores["alignment"] = 1.0 if motion is not None else 0.5
        scores["visual_density"] = self._density_quality(
            artifact_id,
            storyboard,
            audience,
            findings,
        )

        overall = sum(scores.values()) / len(scores)
        decision = self._decision(findings)
        return QualityReport(
            overall_score=overall,
            scores=scores,
            findings=findings,
            decision=decision,
        )

    def _concept_coverage(
        self,
        artifact_id: str,
        graph: ConceptGraph | None,
        storyboard: Storyboard | None,
        findings: list[QualityFinding],
    ) -> float:
        """Score important concepts referenced by storyboard beats."""

        if graph is None or storyboard is None:
            return 0.5
        important = {
            node.concept_id
            for node in graph.nodes
            if node.importance >= 0.5
        }
        represented = {
            concept_id
            for beat in storyboard.beats
            for concept_id in beat.concept_ids
        }
        score = 1.0 if not important else len(important & represented) / len(important)
        if score < self._policy.minimum_concept_coverage:
            findings.append(
                QualityFinding(
                    code="semantic_coverage_low",
                    severity=FindingSeverity.ERROR,
                    artifact_id=artifact_id,
                    message=(
                        f"Important concept coverage {score:.2f} is below "
                        f"{self._policy.minimum_concept_coverage:.2f}."
                    ),
                    repair_target="storyboard",
                )
            )
        return score

    def _asset_readiness(
        self,
        artifact_id: str,
        assets: ResolvedAssetSet | None,
        findings: list[QualityFinding],
    ) -> float:
        """Reject unresolved or placeholder assets."""

        if assets is None or not assets.assets:
            return 1.0
        ready = 0
        for asset in assets.assets:
            placeholder = "placeholder" in asset.path.lower()
            if asset.ready and not placeholder:
                ready += 1
                continue
            severity = (
                FindingSeverity.FATAL
                if self._policy.fail_on_placeholder
                else FindingSeverity.ERROR
            )
            findings.append(
                QualityFinding(
                    code="asset_not_ready",
                    severity=severity,
                    artifact_id=asset.asset_id,
                    message="Required semantic asset is unresolved or a placeholder.",
                    repair_target="assets",
                )
            )
        return ready / len(assets.assets)

    def _layout_quality(
        self,
        artifact_id: str,
        layout: LayoutPlan | None,
        findings: list[QualityFinding],
    ) -> float:
        """Convert layout diagnostics to quality findings."""

        if layout is None:
            return 0.5
        for diagnostic in layout.diagnostics:
            findings.append(
                QualityFinding(
                    code="layout_violation",
                    severity=FindingSeverity.ERROR,
                    artifact_id=artifact_id,
                    message=diagnostic,
                    repair_target="layout",
                )
            )
        return 1.0 if not layout.diagnostics else max(
            0.0,
            1.0 - 0.1 * len(layout.diagnostics),
        )

    def _motion_quality(
        self,
        artifact_id: str,
        motion: MotionPlan | None,
        findings: list[QualityFinding],
    ) -> float:
        """Measure unexplained static gaps in the motion timeline."""

        if motion is None or not motion.events:
            return 0.5
        intervals = sorted(
            (event.start_time, event.start_time + event.duration)
            for event in motion.events
        )
        cursor = 0.0
        maximum_gap = 0.0
        for start, end in intervals:
            maximum_gap = max(maximum_gap, max(0.0, start - cursor))
            cursor = max(cursor, end)
        maximum_gap = max(maximum_gap, motion.duration - cursor)
        if maximum_gap > self._policy.maximum_static_gap:
            findings.append(
                QualityFinding(
                    code="static_gap_exceeded",
                    severity=FindingSeverity.ERROR,
                    artifact_id=artifact_id,
                    message=(
                        f"Maximum static gap {maximum_gap:.2f}s exceeds "
                        f"{self._policy.maximum_static_gap:.2f}s."
                    ),
                    repair_target="motion",
                )
            )
        return min(1.0, self._policy.maximum_static_gap / max(maximum_gap, 0.001))

    def _readability(self, layout: LayoutPlan | None) -> float:
        """Score minimum widths for text-like semantic nodes."""

        if layout is None:
            return 0.5
        text_nodes = [
            node
            for root in layout.state_roots.values()
            for node in self._flatten(root)
            if node.kind in {"text", "label", "annotation", "equation"}
        ]
        if not text_nodes:
            return 1.0
        readable = sum(
            node.box.width >= self._policy.minimum_text_width
            for node in text_nodes
        )
        return readable / len(text_nodes)

    def _density_quality(
        self,
        artifact_id: str,
        storyboard: Storyboard | None,
        audience: AudienceProfile | None,
        findings: list[QualityFinding],
    ) -> float:
        """Enforce audience-aware object-density budgets."""

        if storyboard is None or audience is None:
            return 1.0
        violations = VisualDensityPlanner().violations(storyboard, audience)
        for violation in violations:
            findings.append(
                QualityFinding(
                    code="visual_density_exceeded",
                    severity=FindingSeverity.ERROR,
                    artifact_id=artifact_id,
                    message=violation,
                    repair_target="storyboard",
                )
            )
        return max(0.0, 1.0 - 0.15 * len(violations))

    def _flatten(self, root: LaidOutNode) -> list[LaidOutNode]:
        """Flatten one layout hierarchy."""

        result = [root]
        for child in root.children:
            result.extend(self._flatten(child))
        return result

    @staticmethod
    def _typed(value: object, expected: type[object]) -> object | None:
        """Return a context value only when it has the requested type."""

        return value if isinstance(value, expected) else None

    @staticmethod
    def _decision(findings: list[QualityFinding]) -> EvaluationDecision:
        """Map finding severity to a gate decision."""

        if any(item.severity is FindingSeverity.FATAL for item in findings):
            return EvaluationDecision.FAIL
        if any(item.severity is FindingSeverity.ERROR for item in findings):
            return EvaluationDecision.REPAIR
        return EvaluationDecision.PASS
