"""Deterministic educational-coherence evaluation."""

from app.domain.generation import AudienceLevel, AudienceProfile
from app.domain.lesson import ConceptGraph, ConceptRelation
from app.domain.narration import NarrationPlan
from app.domain.operations import OperationType
from app.domain.pedagogy import PedagogyMode, PedagogyPlan
from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityFinding,
    QualityReport,
)
from app.domain.storyboard import Storyboard, VisualObjectSpec
from app.planning.graph_semantics import analyze_graph


class EducationalQualityEvaluator:
    """Score teaching progression, examples, summaries, and narration fit."""

    def evaluate(
        self,
        artifact_id: str,
        artifact: object,
        context: dict[str, object],
    ) -> QualityReport:
        """Return advisory educational findings without inventing facts."""

        storyboard = artifact if isinstance(artifact, Storyboard) else context.get("storyboard")
        graph = context.get("concept_graph")
        audience = context.get("audience")
        narration = context.get("narration")
        pedagogy = context.get("pedagogy")
        if not isinstance(storyboard, Storyboard):
            return QualityReport(
                overall_score=1.0,
                scores={"educational_coherence": 1.0},
                decision=EvaluationDecision.PASS,
            )
        findings: list[QualityFinding] = []
        progression = self._progression(storyboard, graph, artifact_id, findings)
        examples = self._examples(storyboard, artifact_id, findings)
        summary = self._summary(storyboard, graph, artifact_id, findings)
        narration_fit = self._narration_fit(
            narration,
            audience,
            artifact_id,
            findings,
        )
        semantic_fidelity = self._semantic_fidelity(
            storyboard,
            graph,
            artifact_id,
            findings,
        )
        visual_progression = self._visual_progression(
            storyboard,
            artifact_id,
            findings,
        )
        graph_sufficiency = self._graph_sufficiency(
            graph,
            pedagogy,
            artifact_id,
            findings,
        )
        scores = {
            "teaching_progression": progression,
            "worked_examples": examples,
            "learning_summary": summary,
            "audience_narration_fit": narration_fit,
            "semantic_fidelity": semantic_fidelity,
            "visual_progression": visual_progression,
            "graph_sufficiency": graph_sufficiency,
        }
        decision = (
            EvaluationDecision.REPAIR
            if any(item.severity is FindingSeverity.ERROR for item in findings)
            else EvaluationDecision.PASS
        )
        return QualityReport(
            overall_score=sum(scores.values()) / len(scores),
            scores=scores,
            findings=findings,
            decision=decision,
        )

    @staticmethod
    def _semantic_fidelity(
        storyboard: Storyboard,
        graph: object,
        artifact_id: str,
        findings: list[QualityFinding],
    ) -> float:
        """Verify that hierarchy and connectors preserve graph relationships."""

        if not isinstance(graph, ConceptGraph):
            return 1.0
        roots: list[VisualObjectSpec] = []
        for beat in storyboard.beats:
            for operation in beat.operations:
                if operation.operation is not OperationType.CREATE:
                    continue
                raw_objects = operation.arguments.get("objects", [])
                if isinstance(raw_objects, list):
                    roots.extend(
                        VisualObjectSpec.model_validate(item)
                        for item in raw_objects
                        if isinstance(item, dict)
                    )
        if not roots:
            return 0.5

        objects: dict[str, VisualObjectSpec] = {}
        parents: dict[str, str] = {}
        for root in roots:
            def visit(item: VisualObjectSpec, parent: str | None) -> None:
                objects[item.object_id] = item
                if parent is not None:
                    parents[item.object_id] = parent
                for child in item.children:
                    visit(child, item.object_id)
            visit(root, None)

        nodes = {node.concept_id: node for node in graph.nodes}
        by_label = {
            node.label.strip().casefold(): node.concept_id
            for node in graph.nodes
        }
        concept_objects: dict[str, list[str]] = {}
        violations = 0
        for item in objects.values():
            if item.kind == "connector":
                continue
            if len(item.concept_ids) == 1:
                concept_objects.setdefault(item.concept_ids[0], []).append(
                    item.object_id
                )
            label = str(item.content.get("label", "")).strip().casefold()
            expected = by_label.get(label)
            if (
                expected is not None
                and item.concept_ids
                and expected not in item.concept_ids
            ):
                violations += 1
                findings.append(QualityFinding(
                    code="concept_object_mismatch",
                    severity=FindingSeverity.ERROR,
                    artifact_id=item.object_id,
                    message=(
                        f"Object labelled {nodes[expected].label!r} is not "
                        f"grounded to concept {expected!r}."
                    ),
                    repair_target="storyboard",
                ))

        connector_pairs: set[tuple[str, str, str | None]] = set()
        for item in objects.values():
            if item.kind != "connector":
                continue
            source = objects.get(str(item.content.get("source_id", "")))
            target = objects.get(str(item.content.get("target_id", "")))
            if source is None or target is None:
                continue
            relation = item.content.get("relation")
            relation_name = str(relation) if relation is not None else None
            for source_id in source.concept_ids:
                for target_id in target.concept_ids:
                    connector_pairs.add((source_id, target_id, relation_name))

        def descends(child_id: str, parent_id: str) -> bool:
            cursor = parents.get(child_id)
            visited: set[str] = set()
            while cursor is not None and cursor not in visited:
                if cursor == parent_id:
                    return True
                visited.add(cursor)
                cursor = parents.get(cursor)
            return False

        for edge in graph.edges:
            source_objects = concept_objects.get(edge.source_id, [])
            target_objects = concept_objects.get(edge.target_id, [])
            if not source_objects or not target_objects:
                continue
            represented = False
            if edge.relation is ConceptRelation.PART_OF:
                represented = any(
                    descends(source_id, target_id)
                    for source_id in source_objects
                    for target_id in target_objects
                )
                represented = represented or any(
                    source == edge.source_id
                    and target == edge.target_id
                    and relation == edge.relation.value
                    for source, target, relation in connector_pairs
                )
            else:
                represented = any(
                    source == edge.source_id
                    and target == edge.target_id
                    and (relation is None or relation == edge.relation.value)
                    for source, target, relation in connector_pairs
                )
                if edge.relation is ConceptRelation.CONTRASTS_WITH:
                    represented = represented or any(
                        source == edge.target_id and target == edge.source_id
                        for source, target, _ in connector_pairs
                    )
            if represented:
                continue
            violations += 1
            findings.append(QualityFinding(
                code="semantic_relation_missing",
                severity=FindingSeverity.ERROR,
                artifact_id=artifact_id,
                message=(
                    f"Relationship {edge.source_id} {edge.relation.value} "
                    f"{edge.target_id} is not represented with the required "
                    f"hierarchy or connector."
                ),
                repair_target="storyboard",
            ))
        return max(0.0, 1.0 - violations / max(1, len(graph.edges) + len(nodes)))

    @staticmethod
    def _visual_progression(
        storyboard: Storyboard,
        artifact_id: str,
        findings: list[QualityFinding],
    ) -> float:
        """Reject lessons whose later shots only pulse the original picture."""

        if len(storyboard.beats) < 3:
            return 1.0
        later = [
            operation.operation
            for beat in storyboard.beats[1:]
            for operation in beat.operations
        ]
        meaningful = {
            OperationType.SHOW,
            OperationType.HIDE,
            OperationType.UPDATE,
            OperationType.MORPH,
            OperationType.CONNECT,
            OperationType.DISCONNECT,
            OperationType.MOVE,
            OperationType.GROUP,
            OperationType.UNGROUP,
        }
        if later and not set(later).intersection(meaningful):
            findings.append(QualityFinding(
                code="visual_progression_highlight_only",
                severity=FindingSeverity.ERROR,
                artifact_id=artifact_id,
                message=(
                    "Later shots only highlight the initial picture; they do not "
                    "reveal, transform, connect, or reorganize semantic state."
                ),
                repair_target="storyboard",
            ))
            return 0.35
        focus_signatures: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        for beat in storyboard.beats:
            emphasized = tuple(sorted({
                target
                for operation in beat.operations
                if operation.operation is OperationType.HIGHLIGHT
                for target in operation.target_ids
            }))
            changed = tuple(sorted({
                f"{operation.operation.value}:{target}"
                for operation in beat.operations
                if operation.operation in {
                    OperationType.HIDE,
                    OperationType.UPDATE,
                    OperationType.MORPH,
                    OperationType.CONNECT,
                    OperationType.DISCONNECT,
                    OperationType.MOVE,
                    OperationType.GROUP,
                    OperationType.UNGROUP,
                }
                for target in operation.target_ids
            }))
            focus_signatures.append((emphasized, changed))
        repeated = [
            storyboard.beats[index].beat_id
            for index in range(1, len(focus_signatures))
            if focus_signatures[index] == focus_signatures[index - 1]
            and focus_signatures[index] != ((), ())
        ]
        if repeated:
            findings.append(QualityFinding(
                code="visual_state_repeated",
                severity=FindingSeverity.ERROR,
                artifact_id=artifact_id,
                message=(
                    "Adjacent shots produce the same focus and transformation "
                    f"state: {repeated}."
                ),
                repair_target="storyboard",
            ))
            return 0.35
        return 1.0

    @staticmethod
    def _graph_sufficiency(
        graph: object,
        pedagogy: object,
        artifact_id: str,
        findings: list[QualityFinding],
    ) -> float:
        """Ensure the concept graph can support the routed explanation mode."""

        if not isinstance(graph, ConceptGraph) or not isinstance(
            pedagogy, PedagogyPlan
        ):
            return 1.0
        if pedagogy.mode is not PedagogyMode.MECHANISM_FIRST:
            return 1.0
        if analyze_graph(graph).is_process:
            return 1.0
        findings.append(QualityFinding(
            code="mechanism_relation_missing",
            severity=FindingSeverity.ERROR,
            artifact_id=artifact_id,
            message=(
                "The lesson is routed as a mechanism explanation, but its concept "
                "graph has no connected, factually labelled causal sequence."
            ),
            repair_target="lesson",
        ))
        return 0.0

    def _progression(
        self,
        storyboard: Storyboard,
        graph: object,
        artifact_id: str,
        findings: list[QualityFinding],
    ) -> float:
        purposes = [beat.purpose for beat in storyboard.beats]
        score = 1.0
        if len(purposes) >= 3 and len(set(purposes)) < 2:
            score -= 0.3
            findings.append(self._warning(
                "teaching_arc_flat",
                artifact_id,
                "The lesson uses one beat purpose throughout; add demonstration, connection, or summary beats.",
            ))
        if isinstance(graph, ConceptGraph):
            first_seen = {
                concept_id: index
                for index, beat in enumerate(storyboard.beats)
                for concept_id in beat.concept_ids
                if concept_id not in {
                    concept
                    for prior in storyboard.beats[:index]
                    for concept in prior.concept_ids
                }
            }
            inversions = sum(
                prerequisite in first_seen
                and node.concept_id in first_seen
                and first_seen[prerequisite] > first_seen[node.concept_id]
                for node in graph.nodes
                for prerequisite in node.prerequisites
            )
            if inversions:
                score -= min(0.4, inversions * 0.1)
                findings.append(self._warning(
                    "prerequisite_order",
                    artifact_id,
                    f"{inversions} prerequisite relationship(s) appear after the dependent concept.",
                ))
        return max(0.0, score)

    @staticmethod
    def _examples(
        storyboard: Storyboard,
        artifact_id: str,
        findings: list[QualityFinding],
    ) -> float:
        if len(storyboard.beats) < 3:
            return 1.0
        has_example = any(
            beat.purpose in {"demonstrate", "transform", "compare"}
            or "example" in beat.teaching_intent.lower()
            for beat in storyboard.beats
        )
        if not has_example and len(storyboard.beats) >= 3:
            findings.append(EducationalQualityEvaluator._warning(
                "worked_example_missing",
                artifact_id,
                "No beat visibly demonstrates, transforms, or compares the concept.",
            ))
        return 1.0 if has_example else 0.7

    @staticmethod
    def _summary(
        storyboard: Storyboard,
        graph: object,
        artifact_id: str,
        findings: list[QualityFinding],
    ) -> float:
        if len(storyboard.beats) < 3:
            return 1.0
        has_summary_beat = any(beat.purpose == "summarize" for beat in storyboard.beats)
        has_points = bool(storyboard.final_learning_summary)
        score = (0.5 if has_summary_beat else 0.0) + (0.5 if has_points else 0.0)
        if score < 1.0:
            findings.append(EducationalQualityEvaluator._warning(
                "learning_summary_incomplete",
                artifact_id,
                "Finish with a summary beat and concise final learning points.",
            ))
        if isinstance(graph, ConceptGraph) and has_points:
            important_labels = [
                node.label.lower()
                for node in graph.nodes
                if node.importance >= 0.7
            ]
            summary_text = " ".join(storyboard.final_learning_summary).lower()
            if important_labels and not any(label in summary_text for label in important_labels):
                score = max(0.5, score - 0.2)
        return score

    @staticmethod
    def _narration_fit(
        narration: object,
        audience: object,
        artifact_id: str,
        findings: list[QualityFinding],
    ) -> float:
        if not isinstance(narration, NarrationPlan) or not isinstance(audience, AudienceProfile):
            return 1.0
        obligation_verbs = {
            "describe", "explain", "show", "display", "reveal", "state",
            "name", "identify", "highlight", "mark", "connect", "align",
        }
        leaked = [
            phrase.phrase_id
            for phrase in narration.phrases
            if (
                phrase.text.strip().split(maxsplit=1)[0].strip(".,:;!?").casefold()
                in obligation_verbs
                or "evidence:" in phrase.text.casefold()
            )
        ]
        if leaked:
            findings.append(QualityFinding(
                code="narration_instruction_leak",
                severity=FindingSeverity.ERROR,
                artifact_id=artifact_id,
                message=(
                    "Narration contains internal storyboard instructions instead "
                    f"of listener-facing speech in phrases: {leaked}."
                ),
                repair_target="narration",
            ))
        words = [word.strip(".,:;!?()[]") for phrase in narration.phrases for word in phrase.text.split()]
        average = sum(len(word) for word in words) / max(1, len(words))
        ceiling = {
            AudienceLevel.BEGINNER: 6.4,
            AudienceLevel.INTERMEDIATE: 7.2,
            AudienceLevel.ADVANCED: 8.5,
        }[audience.level]
        if average > ceiling:
            findings.append(EducationalQualityEvaluator._warning(
                "audience_language_dense",
                artifact_id,
                f"Average word length {average:.1f} may be dense for a {audience.level.value} audience.",
            ))
            language_score = max(0.5, ceiling / average)
        else:
            language_score = 1.0
        return min(language_score, 0.25 if leaked else 1.0)

    @staticmethod
    def _warning(code: str, artifact_id: str, message: str) -> QualityFinding:
        return QualityFinding(
            code=code,
            severity=FindingSeverity.WARNING,
            artifact_id=artifact_id,
            message=message,
            repair_target="storyboard",
        )
