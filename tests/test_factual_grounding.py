"""Source provenance and optional domain-validation regressions."""

import json
import pytest

from app.agents.lesson_planner import GeminiLessonPlanner
from app.domain.generation import AudienceProfile, GenerationRequest
from app.domain.grounding import FactualGroundingError, GroundingDecision
from app.domain.lesson import (
    ConceptEdge,
    ConceptGraph,
    ConceptNode,
    ConceptRelation,
    FactualClaim,
    LessonPlan,
    SourceReference,
    SourceRelationAssertion,
)
from app.validation import LessonGroundingValidator, SequentialProcessValidator


class _LessonClient:
    PROVIDER = "nvidia"

    def __init__(self, lesson: LessonPlan) -> None:
        self.lesson = lesson
        self.prompts: list[str] = []

    def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        del temperature
        self.prompts.append(prompt)
        return json.dumps(self.lesson.model_dump(mode="json"))


def _source() -> SourceReference:
    return SourceReference(
        source_id="source_transfer",
        title="Transfer protocol specification",
        content=(
            "The local file enters a network request. The network request "
            "carries the local file to the cloud object."
        ),
        relation_assertions=[SourceRelationAssertion(
            assertion_id="assert_transfer",
            subject="Local file",
            relation=ConceptRelation.FLOWS_TO,
            object="Cloud object",
        )],
    )


def _lesson(*, reversed_edge: bool = False) -> LessonPlan:
    source_id, target_id = (
        ("cloud", "local") if reversed_edge else ("local", "cloud")
    )
    return LessonPlan(
        title="How a file transfer works",
        summary="A local file travels to a cloud object.",
        sources=[_source()],
        claims=[
            FactualClaim(
                claim_id="claim_local",
                text="The local file enters a network request.",
                source_ids=["source_transfer"],
            ),
            FactualClaim(
                claim_id="claim_cloud",
                text="The network request carries the file to the cloud object.",
                source_ids=["source_transfer"],
            ),
            FactualClaim(
                claim_id="claim_relation",
                text="The local file flows to the cloud object.",
                source_ids=["source_transfer"],
            ),
        ],
        concept_graph=ConceptGraph(
            objectives=["Explain the sourced transfer"],
            nodes=[
                ConceptNode(
                    concept_id="local",
                    label="Local file",
                    definition="The file before transfer.",
                    importance=1,
                    teaching_order=0,
                    claim_ids=["claim_local"],
                ),
                ConceptNode(
                    concept_id="cloud",
                    label="Cloud object",
                    definition="The stored cloud representation.",
                    importance=1,
                    teaching_order=1,
                    claim_ids=["claim_cloud"],
                ),
            ],
            edges=[ConceptEdge(
                edge_id="file_flow",
                source_id=source_id,
                target_id=target_id,
                relation=ConceptRelation.FLOWS_TO,
                claim_ids=["claim_relation"],
            )],
            teaching_sequence=["local", "cloud"],
        ),
    )


def test_source_grounded_claims_and_relation_are_traceable() -> None:
    grounded, report = LessonGroundingValidator().validate(_lesson())

    assert report.decision is GroundingDecision.PASS
    assert report.validated_claims == 3
    assert report.validated_relations == 1
    assert {
        source_id
        for claim in grounded.claims
        for source_id in claim.source_ids
    } == {"source_transfer"}


def test_reversed_relation_fails_with_source_and_edge_evidence() -> None:
    _grounded, report = LessonGroundingValidator().validate(
        _lesson(reversed_edge=True)
    )

    issue = next(
        item for item in report.issues
        if item.code == "source_relation_reversed"
    )
    assert report.decision is GroundingDecision.FAIL
    assert issue.edge_ids == ["file_flow"]
    assert issue.source_ids == ["source_transfer"]
    with pytest.raises(FactualGroundingError, match="source_relation_reversed"):
        LessonGroundingValidator.raise_for_failure(report)


def test_unsupported_and_low_confidence_claims_route_to_failure_or_review() -> None:
    unsupported = _lesson()
    unsupported.sources[0].relation_assertions = []
    _grounded, unsupported_report = LessonGroundingValidator().validate(unsupported)
    assert "source_relation_unsupported" in {
        item.code for item in unsupported_report.issues
    }

    review = _lesson()
    review.claims[0].confidence = 0.4
    review.claims[0].status = "review"
    _grounded, review_report = LessonGroundingValidator().validate(review)
    assert review_report.decision is GroundingDecision.REVIEW
    assert "claim_confidence_low" in {
        item.code for item in review_report.issues
    }


def test_optional_domain_validator_has_no_eager_external_dependencies() -> None:
    lesson = _lesson(reversed_edge=True)
    # Remove source assertions so this assertion focuses on the plugin's
    # benchmark-process invariant and not source validation.
    graph = lesson.concept_graph.model_copy(deep=True)
    for node in graph.nodes:
        node.claim_ids = []
    for edge in graph.edges:
        edge.claim_ids = []
    lesson = lesson.model_copy(update={
        "sources": [],
        "claims": [],
        "concept_graph": graph,
    })

    _grounded, report = LessonGroundingValidator(
        [SequentialProcessValidator()]
    ).validate(lesson)

    assert "domain_process_direction_reversed" in {
        item.code for item in report.issues
    }


def test_lesson_prompt_requires_claim_provenance_when_sources_are_supplied() -> None:
    client = _LessonClient(_lesson())
    request = GenerationRequest(
        run_id="grounded_prompt",
        topic="Sourced file transfer",
        audience=AudienceProfile(learning_goal="Explain the sourced transfer"),
        sources=[_source()],
    )

    result = GeminiLessonPlanner(client, max_attempts=1).plan(request)

    assert result.claims
    assert "claim-level" in client.prompts[0]
    assert "relation_assertions" in client.prompts[0]
