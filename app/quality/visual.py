"""Geometry-aware preflight gate for renderer-ready visual plans."""

from app.domain.layout import LaidOutNode, LayoutBox, LayoutPlan
from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityFinding,
    QualityReport,
)


class VisualQualityEvaluator:
    """Detect clipping, sibling collisions, and unreadable object geometry."""

    def evaluate(
        self,
        artifact_id: str,
        artifact: object,
        context: dict[str, object],
    ) -> QualityReport:
        """Evaluate every layout state using semantic hierarchy geometry."""

        layout = artifact if isinstance(artifact, LayoutPlan) else context.get("layout")
        if not isinstance(layout, LayoutPlan):
            return QualityReport(
                overall_score=1.0,
                scores={"visual_preflight": 1.0},
                decision=EvaluationDecision.PASS,
            )
        findings: list[QualityFinding] = []
        checked = 0
        violations = 0
        for state_id, root in layout.state_roots.items():
            for node in self._flatten(root):
                checked += 1
                if not self._inside(node.box, layout):
                    violations += 1
                    findings.append(
                        self._finding(
                            "object_clipped",
                            FindingSeverity.ERROR,
                            node.object_id,
                            f"Object {node.object_id} leaves the safe viewport in {state_id}.",
                        )
                    )
                if node.kind in {"text", "label", "annotation", "equation"} and (
                    node.box.width < 72 or node.box.height < 28
                ):
                    violations += 1
                    findings.append(
                        self._finding(
                            "text_box_unreadable",
                            FindingSeverity.ERROR,
                            node.object_id,
                            f"Text geometry for {node.object_id} is too small to render safely.",
                        )
                    )
            for parent in self._flatten(root):
                siblings = [
                    child
                    for child in parent.children
                    if child.kind != "connector"
                ]
                for index, first in enumerate(siblings):
                    for second in siblings[index + 1:]:
                        checked += 1
                        overlap = self._overlap_ratio(first.box, second.box)
                        if overlap <= 0.12:
                            continue
                        violations += 1
                        findings.append(
                            self._finding(
                                "sibling_overlap",
                                FindingSeverity.ERROR,
                                parent.object_id,
                                f"Sibling objects {first.object_id} and {second.object_id} overlap by {overlap:.0%}.",
                            )
                        )
        score = max(0.0, 1.0 - violations / max(1, checked))
        decision = (
            EvaluationDecision.REPAIR
            if any(item.severity is FindingSeverity.ERROR for item in findings)
            else EvaluationDecision.PASS
        )
        return QualityReport(
            overall_score=score,
            scores={"visual_preflight": score},
            findings=findings,
            decision=decision,
        )

    @staticmethod
    def _inside(box: LayoutBox, layout: LayoutPlan) -> bool:
        """Return whether geometry stays inside the physical canvas."""

        return (
            box.x >= -1e-6
            and box.y >= -1e-6
            and box.x + box.width <= layout.viewport.width + 1e-6
            and box.y + box.height <= layout.viewport.height + 1e-6
        )

    @staticmethod
    def _overlap_ratio(first: LayoutBox, second: LayoutBox) -> float:
        """Measure intersection against the smaller sibling area."""

        width = max(
            0.0,
            min(first.x + first.width, second.x + second.width)
            - max(first.x, second.x),
        )
        height = max(
            0.0,
            min(first.y + first.height, second.y + second.height)
            - max(first.y, second.y),
        )
        return width * height / max(
            1.0,
            min(first.width * first.height, second.width * second.height),
        )

    @staticmethod
    def _flatten(root: LaidOutNode) -> list[LaidOutNode]:
        """Return a stable pre-order hierarchy."""

        result = [root]
        for child in root.children:
            result.extend(VisualQualityEvaluator._flatten(child))
        return result

    @staticmethod
    def _finding(
        code: str,
        severity: FindingSeverity,
        artifact_id: str,
        message: str,
    ) -> QualityFinding:
        """Build a layout-repair finding."""

        return QualityFinding(
            code=code,
            severity=severity,
            artifact_id=artifact_id,
            message=message,
            repair_target="layout",
        )
