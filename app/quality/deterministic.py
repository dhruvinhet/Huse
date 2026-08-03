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
from app.domain.storyboard import Storyboard, VisualObjectSpec
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
            storyboard,
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
        scores["state_integrity"] = self._state_progression(
            artifact_id,
            document,
            findings,
        )
        scores["readability"] = self._readability(
            artifact_id,
            layout,
            document,
            findings,
        )
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
        missing = sorted(important - represented)
        score = 1.0 if not important else len(important & represented) / len(important)
        if score < self._policy.minimum_concept_coverage:
            findings.append(
                QualityFinding(
                    code="semantic_coverage_low",
                    severity=FindingSeverity.ERROR,
                    artifact_id=artifact_id,
                    message=(
                        f"Important concept coverage {score:.2f} is below "
                        f"{self._policy.minimum_concept_coverage:.2f}. "
                        f"Missing concept IDs: {missing}."
                    ),
                    repair_target="storyboard",
                )
            )
        return score

    def _asset_readiness(
        self,
        artifact_id: str,
        assets: ResolvedAssetSet | None,
        storyboard: Storyboard | None,
        findings: list[QualityFinding],
    ) -> float:
        """Reject unresolved or placeholder assets."""

        if assets is None:
            return 1.0
        expected: set[str] = set()
        if storyboard is not None:
            roots = list(storyboard.initial_objects)
            for beat in storyboard.beats:
                for operation in beat.operations:
                    raw = operation.arguments.get("objects")
                    if not isinstance(raw, list):
                        continue
                    roots.extend(
                        VisualObjectSpec.model_validate(item)
                        for item in raw
                        if isinstance(item, dict)
                    )
            expected = {
                f"asset_{item.object_id}"
                for root in roots
                for item in root.flatten()
                if item.asset_query is not None
            }
        resolved_ids = {item.asset_id for item in assets.assets}
        missing = sorted(expected - resolved_ids)
        if missing:
            findings.append(QualityFinding(
                code="semantic_assets_missing",
                severity=FindingSeverity.ERROR,
                artifact_id=artifact_id,
                message=f"Required semantic assets were not resolved: {missing}.",
                repair_target="assets",
            ))
        if not assets.assets:
            return 0.0 if expected else 1.0
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
        return min(
            ready / len(assets.assets),
            (len(expected & resolved_ids) / len(expected)) if expected else 1.0,
        )

    @staticmethod
    def _state_progression(
        artifact_id: str,
        document: VisualDocument | None,
        findings: list[QualityFinding],
    ) -> float:
        """Reject adjacent checkpoints that are visually identical."""

        if document is None:
            return 0.5
        signatures: list[tuple[tuple[object, ...], ...]] = []
        for state in document.states:
            signatures.append(tuple(sorted(
                (
                    object_id,
                    item.kind,
                    item.lifecycle.value,
                    repr(sorted(item.content.items())),
                )
                for object_id, item in state.object_states.items()
            )))
        repeated = [
            document.states[index].beat_id
            for index in range(1, len(signatures))
            if signatures[index] == signatures[index - 1]
        ]
        if repeated:
            findings.append(QualityFinding(
                code="visual_state_unchanged",
                severity=FindingSeverity.ERROR,
                artifact_id=artifact_id,
                message=f"Adjacent beats do not change visible state: {repeated}.",
                repair_target="storyboard",
            ))
            return max(0.0, 1.0 - len(repeated) / len(signatures))
        return 1.0

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

    def _readability(
        self,
        artifact_id: str,
        layout: LayoutPlan | None,
        document: VisualDocument | None,
        findings: list[QualityFinding],
    ) -> float:
        """Score geometry for every object that actually paints text."""

        if layout is None:
            return 0.5
        content_by_id = {
            object_id: object_state.content
            for state in (document.states if document is not None else [])
            for object_id, object_state in state.object_states.items()
        }
        nodes_by_id: dict[str, LaidOutNode] = {}
        for root in layout.state_roots.values():
            for node in self._flatten(root):
                nodes_by_id[node.object_id] = node
        candidates: list[tuple[LaidOutNode, bool, str]] = []
        for node in nodes_by_id.values():
            if node.kind == "connector":
                continue
            content = content_by_id.get(node.object_id, {})
            has_label = node.kind in {
                "text", "label", "annotation", "equation"
            } or any(
                isinstance(content.get(key), str)
                and bool(str(content.get(key)).strip())
                for key in ("label", "text", "value")
            )
            has_detail = isinstance(content.get("detail"), str) and bool(
                str(content.get("detail")).strip()
            )
            if has_label:
                label = str(
                    content.get("label")
                    or content.get("text")
                    or content.get("value")
                    or ""
                ).strip()
                candidates.append((node, has_detail, label))
        if not candidates:
            return 1.0
        unreadable: list[str] = []
        output_scale = min(
            1.0,
            layout.viewport.width / 1920.0,
            layout.viewport.height / 1080.0,
        )
        for node, has_detail, label in candidates:
            label_width = min(
                320.0,
                max(
                    self._policy.minimum_text_width,
                    28.0 + min(36, len(label)) * 8.0,
                ),
            )
            minimum_width = max(
                180.0 if has_detail else 0.0,
                label_width,
            ) * output_scale
            minimum_height = (110.0 if has_detail else 36.0) * output_scale
            if (
                node.box.width < minimum_width
                or node.box.height < minimum_height
            ):
                unreadable.append(node.object_id)
        if unreadable:
            findings.append(QualityFinding(
                code="semantic_text_geometry_unreadable",
                severity=FindingSeverity.ERROR,
                artifact_id=artifact_id,
                message=(
                    "Text-bearing semantic objects are too small for their "
                    f"content: {sorted(unreadable)}."
                ),
                repair_target="layout",
            ))
        return 1.0 - len(unreadable) / len(candidates)

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
