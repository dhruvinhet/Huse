"""Deterministic educational-coherence evaluation."""

from app.domain.generation import AudienceLevel, AudienceProfile
from app.domain.lesson import ConceptGraph
from app.domain.narration import NarrationPlan
from app.domain.quality import (
    EvaluationDecision,
    FindingSeverity,
    QualityFinding,
    QualityReport,
)
from app.domain.storyboard import Storyboard


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
        scores = {
            "teaching_progression": progression,
            "worked_examples": examples,
            "learning_summary": summary,
            "audience_narration_fit": narration_fit,
        }
        return QualityReport(
            overall_score=sum(scores.values()) / len(scores),
            scores=scores,
            findings=findings,
            decision=EvaluationDecision.PASS,
        )

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
            return max(0.5, ceiling / average)
        return 1.0

    @staticmethod
    def _warning(code: str, artifact_id: str, message: str) -> QualityFinding:
        return QualityFinding(
            code=code,
            severity=FindingSeverity.WARNING,
            artifact_id=artifact_id,
            message=message,
            repair_target="storyboard",
        )
