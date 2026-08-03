"""Tests for deterministic topic-specific teaching rhetoric."""

import pytest

from app.domain.generation import AudienceProfile
from app.domain.lesson import ConceptGraph, ConceptNode, LessonPlan
from app.domain.pedagogy import PedagogyMode
from app.planning import PedagogyRouter


def _lesson(topic: str) -> LessonPlan:
    """Build a minimal valid lesson whose semantics contain the topic."""

    concept_id = topic.lower().replace(" ", "_")
    return LessonPlan(
        title=topic,
        summary=f"Explain {topic} clearly.",
        concept_graph=ConceptGraph(
            objectives=[f"Understand {topic}"],
            nodes=[
                ConceptNode(
                    concept_id=concept_id,
                    label=topic,
                    definition=f"The central idea in {topic}.",
                    importance=1,
                    teaching_order=0,
                )
            ],
            teaching_sequence=[concept_id],
        ),
    )


@pytest.mark.parametrize(
    ("topic", "expected"),
    [
        ("How a REST request flows", PedagogyMode.MECHANISM_FIRST),
        ("What is computer vision", PedagogyMode.CONCEPT_OVERVIEW),
        ("Calculate compound interest", PedagogyMode.WORKED_EXAMPLE),
        ("Common gravity misconception", PedagogyMode.MISCONCEPTION_CORRECTION),
        ("An analogy for electric current", PedagogyMode.ANALOGY),
        ("Derive the quadratic equation", PedagogyMode.PROOF_DERIVATION),
        ("History of the industrial revolution", PedagogyMode.CHRONOLOGICAL),
        ("Compare min heap versus max heap", PedagogyMode.COMPARISON),
        ("Probability simulation", PedagogyMode.SIMULATION),
        ("Anatomy of an atom", PedagogyMode.SPATIAL_ANATOMY),
        ("Execute a recursive function", PedagogyMode.CODE_EXECUTION),
    ],
)
def test_router_selects_all_supported_modes(
    topic: str,
    expected: PedagogyMode,
) -> None:
    """Stable lesson terms choose the intended explanatory grammar."""

    plan = PedagogyRouter().route(
        _lesson(topic),
        AudienceProfile(learning_goal="Understand the topic"),
    )

    assert plan.mode is expected
    assert len(plan.shots) == 4
    assert len({shot.shot_id for shot in plan.shots}) == 4
    assert all(shot.visual_obligation for shot in plan.shots)
    assert all(shot.narration_obligation for shot in plan.shots)


def test_modes_have_distinct_shot_grammars() -> None:
    """Modes cannot silently collapse back into one reusable lesson shape."""

    router = PedagogyRouter()
    plans = [
        router.route(
            _lesson(topic),
            AudienceProfile(learning_goal="Understand"),
        )
        for topic in (
            "Calculate an interest example",
            "Correct a common misconception",
            "An analogy for voltage",
            "Derive a theorem proof",
            "History timeline",
            "Compare two alternatives",
            "Probability simulation",
            "Anatomy of a cell",
            "Execute program code",
            "How a request flows",
            "What is computer vision",
        )
    ]
    plans.append(PedagogyRouter().route(
        LessonPlan(
            title="Three safety rules",
            summary="A collection of co-equal safety rules.",
            concept_graph=ConceptGraph(
                objectives=["Understand three safety rules"],
                nodes=[
                    ConceptNode(
                        concept_id=f"rule_{index}",
                        label=f"Safety Rule {label}",
                        definition=f"Independent rule {index}.",
                        importance=1,
                        teaching_order=index - 1,
                    )
                    for index, label in enumerate(
                        ("One", "Two", "Three"),
                        start=1,
                    )
                ],
                teaching_sequence=["rule_1", "rule_2", "rule_3"],
            ),
        ),
        AudienceProfile(learning_goal="Understand the collection"),
    ))
    grammars = {
        tuple(shot.shot_id for shot in plan.shots)
        for plan in plans
    }

    assert len(grammars) == len(PedagogyMode)
