"""Lazy, plugin-style factual and domain validation for lesson plans."""

from __future__ import annotations

from typing import Protocol
import re

from app.domain.grounding import (
    FactualGroundingError,
    GroundingDecision,
    GroundingIssue,
    GroundingReport,
)
from app.domain.lesson import ConceptRelation, LessonPlan, SourceReference


class DomainValidator(Protocol):
    """Optional deterministic validator loaded only by caller configuration."""

    validator_id: str

    def supports(self, lesson: LessonPlan) -> bool: ...

    def validate(self, lesson: LessonPlan) -> list[GroundingIssue]: ...


class SequentialProcessValidator:
    """Validate forward sequence for benchmark-style process explanations."""

    validator_id = "sequential-process-v1"
    _DYNAMIC = {
        ConceptRelation.CAUSES,
        ConceptRelation.FLOWS_TO,
        ConceptRelation.TRANSFORMS_TO,
    }
    _TOPIC_MARKERS = {
        "how", "trace", "derive", "milestones", "calculate", "execution",
        "treatment", "fulfillment", "respiration", "signal", "transfer",
    }

    def supports(self, lesson: LessonPlan) -> bool:
        terms = set(re.findall(r"[a-z0-9]+", lesson.title.casefold()))
        return bool(terms.intersection(self._TOPIC_MARKERS)) and any(
            edge.relation in self._DYNAMIC for edge in lesson.concept_graph.edges
        )

    def validate(self, lesson: LessonPlan) -> list[GroundingIssue]:
        order = {
            concept_id: index
            for index, concept_id in enumerate(lesson.concept_graph.teaching_sequence)
        }
        return [
            GroundingIssue(
                code="domain_process_direction_reversed",
                severity="error",
                message=(
                    f"Process edge {edge.edge_id!r} points backward from "
                    f"teaching position {order[edge.source_id]} to "
                    f"{order[edge.target_id]}."
                ),
                edge_ids=[edge.edge_id],
            )
            for edge in lesson.concept_graph.edges
            if edge.relation in self._DYNAMIC
            and order[edge.source_id] > order[edge.target_id]
        ]


class LessonGroundingValidator:
    """Validate claims and directed relations against supplied references."""

    MINIMUM_CLAIM_OVERLAP = 0.22
    MINIMUM_CONFIDENCE = 0.65

    def __init__(self, validators: list[DomainValidator] | None = None) -> None:
        self._validators = list(validators or [])

    def validate(
        self,
        lesson: LessonPlan,
        supplied_sources: list[SourceReference] | None = None,
    ) -> tuple[LessonPlan, GroundingReport]:
        sources = list(supplied_sources or lesson.sources)
        grounded = lesson.model_copy(update={"sources": sources}, deep=True)
        issues: list[GroundingIssue] = []
        validated_claims = 0
        validated_relations = 0
        if sources:
            claim_by_id = {claim.claim_id: claim for claim in grounded.claims}
            source_by_id = {source.source_id: source for source in sources}
            if not grounded.claims:
                issues.append(GroundingIssue(
                    code="source_claims_missing",
                    severity="error",
                    message=(
                        "Sources were supplied, but the lesson returned no "
                        "claim-level provenance."
                    ),
                    source_ids=sorted(source_by_id),
                ))
            for claim in grounded.claims:
                unknown_sources = sorted(set(claim.source_ids) - set(source_by_id))
                if unknown_sources:
                    issues.append(GroundingIssue(
                        code="claim_source_unknown",
                        severity="error",
                        message=(
                            f"Claim {claim.claim_id!r} cites unknown sources: "
                            f"{unknown_sources}."
                        ),
                        claim_ids=[claim.claim_id],
                        source_ids=unknown_sources,
                    ))
                    continue
                cited = [source_by_id[item] for item in claim.source_ids]
                overlap = self._claim_overlap(claim.text, cited)
                if claim.status == "review" or claim.confidence < self.MINIMUM_CONFIDENCE:
                    issues.append(GroundingIssue(
                        code="claim_confidence_low",
                        severity="review",
                        message=(
                            f"Claim {claim.claim_id!r} has confidence "
                            f"{claim.confidence:.2f} and requires review."
                        ),
                        claim_ids=[claim.claim_id],
                        source_ids=claim.source_ids,
                    ))
                elif overlap < self.MINIMUM_CLAIM_OVERLAP:
                    issues.append(GroundingIssue(
                        code="claim_unsupported_by_source",
                        severity="error",
                        message=(
                            f"Claim {claim.claim_id!r} has only {overlap:.1%} "
                            "meaningful-token overlap with its cited sources."
                        ),
                        claim_ids=[claim.claim_id],
                        source_ids=claim.source_ids,
                    ))
                else:
                    validated_claims += 1
            for node in grounded.concept_graph.nodes:
                if not node.claim_ids:
                    issues.append(GroundingIssue(
                        code="concept_provenance_missing",
                        severity="error",
                        message=f"Concept {node.concept_id!r} has no cited claim.",
                    ))
            labels = {
                node.concept_id: node.label for node in grounded.concept_graph.nodes
            }
            for edge in grounded.concept_graph.edges:
                if not edge.claim_ids:
                    issues.append(GroundingIssue(
                        code="relation_provenance_missing",
                        severity="error",
                        message=f"Relation {edge.edge_id!r} has no cited claim.",
                        edge_ids=[edge.edge_id],
                    ))
                    continue
                claim_sources = {
                    source_id
                    for claim_id in edge.claim_ids
                    for source_id in claim_by_id[claim_id].source_ids
                }
                relation_issue = self._validate_relation(
                    edge.edge_id,
                    edge.source_id,
                    labels[edge.source_id],
                    edge.relation,
                    edge.target_id,
                    labels[edge.target_id],
                    [source_by_id[item] for item in claim_sources],
                    list(edge.claim_ids),
                )
                if relation_issue is None:
                    validated_relations += 1
                else:
                    issues.append(relation_issue)

        for validator in self._validators:
            if validator.supports(grounded):
                issues.extend(validator.validate(grounded))
        decision = (
            GroundingDecision.FAIL
            if any(issue.severity == "error" for issue in issues)
            else GroundingDecision.REVIEW
            if issues
            else GroundingDecision.PASS
        )
        return grounded, GroundingReport(
            decision=decision,
            issues=issues,
            validated_claims=validated_claims,
            validated_relations=validated_relations,
        )

    @staticmethod
    def raise_for_failure(report: GroundingReport) -> None:
        if report.decision is GroundingDecision.PASS:
            return
        details = "; ".join(f"{item.code}: {item.message}" for item in report.issues)
        raise FactualGroundingError(
            f"lesson factual grounding requires {report.decision.value}: {details}"
        )

    @staticmethod
    def _tokens(value: str) -> set[str]:
        stop = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are"}
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.casefold())
            if len(token) > 2 and token not in stop
        }

    @classmethod
    def _claim_overlap(
        cls,
        claim: str,
        sources: list[SourceReference],
    ) -> float:
        claim_tokens = cls._tokens(claim)
        source_tokens = cls._tokens(" ".join(source.content for source in sources))
        return (
            len(claim_tokens.intersection(source_tokens)) / len(claim_tokens)
            if claim_tokens else 1.0
        )

    @classmethod
    def _validate_relation(
        cls,
        edge_id: str,
        source_id: str,
        source_label: str,
        relation: ConceptRelation,
        target_id: str,
        target_label: str,
        sources: list[SourceReference],
        claim_ids: list[str],
    ) -> GroundingIssue | None:
        source_names = {source_id.casefold(), source_label.casefold()}
        target_names = {target_id.casefold(), target_label.casefold()}
        assertions = [
            (source.source_id, assertion)
            for source in sources
            for assertion in source.relation_assertions
            if assertion.relation is relation
        ]
        exact = [
            source_id_value
            for source_id_value, assertion in assertions
            if assertion.subject.casefold() in source_names
            and assertion.object.casefold() in target_names
        ]
        if exact:
            return None
        reversed_sources = [
            source_id_value
            for source_id_value, assertion in assertions
            if assertion.subject.casefold() in target_names
            and assertion.object.casefold() in source_names
        ]
        if reversed_sources:
            return GroundingIssue(
                code="source_relation_reversed",
                severity="error",
                message=(
                    f"Relation {edge_id!r} reverses the supplied assertion: "
                    f"{target_label} {relation.value} {source_label}."
                ),
                claim_ids=claim_ids,
                edge_ids=[edge_id],
                source_ids=sorted(set(reversed_sources)),
            )
        return GroundingIssue(
            code="source_relation_unsupported",
            severity="error",
            message=(
                f"No cited source asserts {source_label} {relation.value} "
                f"{target_label}."
            ),
            claim_ids=claim_ids,
            edge_ids=[edge_id],
            source_ids=[source.source_id for source in sources],
        )
