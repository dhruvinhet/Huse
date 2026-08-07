"""Geometry-aware preflight gate for renderer-ready visual plans."""

from app.domain.layout import LaidOutNode, LayoutBox, LayoutPlan
from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityFinding,
    QualityReport,
)
from app.domain.visual_document import ObjectLifecycle, VisualDocument


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
        document = context.get("document")
        document_states = (
            {state.state_id: state for state in document.states}
            if isinstance(document, VisualDocument)
            else {}
        )
        for state_id, root in layout.state_roots.items():
            for content_root in root.children:
                width_ratio = content_root.box.width / layout.viewport.width
                height_ratio = content_root.box.height / layout.viewport.height
                area_ratio = width_ratio * height_ratio
                linear_kind = content_root.kind in {
                    "array", "timeline", "pipeline", "linked_list", "queue"
                }
                if area_ratio < 0.12 or (
                    not linear_kind and height_ratio < 0.20
                ):
                    violations += 1
                    findings.append(
                        self._finding(
                            "canvas_underused",
                            FindingSeverity.ERROR,
                            content_root.object_id,
                            (
                                f"Primary visual uses only {area_ratio:.0%} of the "
                                f"canvas area in {state_id}; its structure is too "
                                "small or flat for the selected operator."
                            ),
                        )
                    )
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
            state = document_states.get(state_id)
            if state is not None:
                crossings = self._connector_crossings(root, state.object_states)
                checked += max(1, len(crossings))
                if crossings:
                    violations += len(crossings)
                    connector_ids = sorted({
                        connector_id
                        for pair in crossings
                        for connector_id in pair
                    })
                    findings.append(QualityFinding(
                        code="connector_crossing",
                        # This preflight uses endpoint centerlines, while the
                        # renderer uses boundary anchors and obstacle-aware
                        # orthogonal routes. Preserve the signal for review,
                        # but do not trigger an impossible scale-only repair
                        # for geometry that is rerouted during rendering.
                        severity=FindingSeverity.WARNING,
                        artifact_id=artifact_id,
                        message=(
                            f"{len(crossings)} connector pairs geometrically cross "
                            f"in {state_id}: {crossings}."
                        ),
                        repair_target="layout",
                        repair_scope="object",
                        object_ids=connector_ids,
                        measured_value=float(len(crossings)),
                        required_value=0.0,
                        patch_paths=["/layout/connectors"],
                    ))
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

    @classmethod
    def _connector_crossings(
        cls,
        root: LaidOutNode,
        object_states: dict[str, object],
    ) -> list[tuple[str, str]]:
        """Detect proper crossings between connector endpoint centerlines."""

        boxes = {
            node.object_id: node.box
            for node in cls._flatten(root)
        }
        segments: list[
            tuple[str, str, str, tuple[float, float], tuple[float, float]]
        ] = []
        for object_id, state in object_states.items():
            if (
                getattr(state, "kind", None) != "connector"
                or getattr(state, "lifecycle", None)
                in {ObjectLifecycle.HIDDEN, ObjectLifecycle.REMOVED}
            ):
                continue
            content = getattr(state, "content", {})
            source_id = content.get("source_id")
            target_id = content.get("target_id")
            if source_id not in boxes or target_id not in boxes:
                continue
            segments.append((
                object_id,
                source_id,
                target_id,
                cls._center(boxes[source_id]),
                cls._center(boxes[target_id]),
            ))
        crossings: list[tuple[str, str]] = []
        for index, first in enumerate(segments):
            for second in segments[index + 1:]:
                if {first[1], first[2]}.intersection({second[1], second[2]}):
                    continue
                if cls._segments_cross(first[3], first[4], second[3], second[4]):
                    crossings.append((first[0], second[0]))
        return crossings

    @staticmethod
    def _center(box: LayoutBox) -> tuple[float, float]:
        return box.x + box.width / 2, box.y + box.height / 2

    @staticmethod
    def _segments_cross(
        first_start: tuple[float, float],
        first_end: tuple[float, float],
        second_start: tuple[float, float],
        second_end: tuple[float, float],
    ) -> bool:
        """Return whether two segments have a proper interior intersection."""

        def orientation(
            first: tuple[float, float],
            second: tuple[float, float],
            third: tuple[float, float],
        ) -> float:
            return (
                (second[0] - first[0]) * (third[1] - first[1])
                - (second[1] - first[1]) * (third[0] - first[0])
            )

        first_side = orientation(first_start, first_end, second_start)
        second_side = orientation(first_start, first_end, second_end)
        third_side = orientation(second_start, second_end, first_start)
        fourth_side = orientation(second_start, second_end, first_end)
        epsilon = 1e-6
        return (
            first_side * second_side < -epsilon
            and third_side * fourth_side < -epsilon
        )

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
            repair_scope="object",
            object_ids=[artifact_id],
            patch_paths=["/layout"],
        )
