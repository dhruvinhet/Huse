"""Broad acceptance matrix for reviewed template action compilation."""

import pytest

from app.domain.lesson import ConceptGraph, ConceptNode, LessonPlan
from app.domain.pedagogy import PedagogyMode, PedagogyPlan, PedagogyShot
from app.domain.strategy import TemplateMatch
from app.planning import TemplateCompiler
from app.state import VisualStateTransitionEngine
from app.templates import TemplateRegistry
from app.templates.builtins import builtin_templates


TOPIC_TEMPLATES = [
    "array.v1",
    "graph.v1",
    "matrix.v1",
    "pipeline.v1",
    "probability_distribution.v1",
    "transformer_block.v1",
    "tree.v1",
    "before_after.v1",
    "comparison.v1",
    "concept_set.v1",
    "timeline.v1",
    "cycle.v1",
    "cause_effect.v1",
    "input_output.v1",
    "layered_architecture.v1",
    "flowchart.v1",
    "funnel.v1",
    "venn.v1",
    "bar_chart.v1",
    "line_chart.v1",
    "equation_derivation.v1",
    "code_trace.v1",
    "binary_search.v1",
    "quick_sort.v1",
    "merge_sort.v1",
    "dfs.v1",
    "bfs.v1",
    "cnn.v1",
    "rnn.v1",
    "tcp_handshake.v1",
]


def _lesson(template_id: str) -> LessonPlan:
    label = template_id.removesuffix(".v1").replace("_", " ").title()
    nodes = [
        ConceptNode(
            concept_id=f"concept_{index}",
            label=f"{label} value {index + 1}",
            definition=f"Concrete {label.lower()} operand {index + 1}.",
            importance=1,
            teaching_order=index,
            visual_affordances=[label.lower()],
        )
        for index in range(4)
    ]
    return LessonPlan(
        title=label,
        summary=f"Use values 1, 2, 3, and 4 to demonstrate {label.lower()}.",
        concept_graph=ConceptGraph(
            objectives=[f"Demonstrate {label.lower()} with visible state changes"],
            nodes=nodes,
            teaching_sequence=[node.concept_id for node in nodes],
        ),
    )


def _pedagogy() -> PedagogyPlan:
    return PedagogyPlan(
        mode=PedagogyMode.WORKED_EXAMPLE,
        rationale="Exercise a concrete reviewed transformation.",
        shots=[
            PedagogyShot(
                shot_id="setup",
                purpose="introduce",
                visual_obligation="Show the concrete starting state.",
                narration_obligation="State the starting operands.",
            ),
            PedagogyShot(
                shot_id="change",
                purpose="transform",
                visual_obligation="Animate the state-changing operation.",
                narration_obligation="Explain the visible change.",
            ),
            PedagogyShot(
                shot_id="result",
                purpose="summarize",
                visual_obligation="Show the resulting state.",
                narration_obligation="State the result.",
            ),
        ],
    )


@pytest.mark.parametrize("template_id", TOPIC_TEMPLATES)
def test_thirty_topic_transformations_lower_to_resolved_non_trace_actions(
    template_id: str,
) -> None:
    """Every reviewed topic family binds a concrete non-trace transition."""

    lesson = _lesson(template_id)
    registry = TemplateRegistry(builtin_templates())
    parameters: dict[str, object] = {}
    if template_id == "array.v1":
        parameters["values"] = ["1", "2", "3", "4"]
    elif template_id == "matrix.v1":
        parameters.update({"rows": 2, "columns": 2})
    match = TemplateMatch(
        template_id=template_id,
        concept_ids=list(lesson.concept_graph.teaching_sequence),
        parameters=parameters,
        score=1,
        reason="Thirty-topic acceptance matrix",
        capabilities=registry.capabilities(template_id),
    )

    program = TemplateCompiler().compile(
        lesson,
        [],
        [match],
        registry,
        _pedagogy(),
    )

    assert program is not None
    transform = next(
        beat for beat in program.storyboard.beats if beat.purpose == "transform"
    )
    assert transform.semantic_actions
    assert all(action.action != "trace" for action in transform.semantic_actions)
    document = VisualStateTransitionEngine().materialize(program.storyboard)
    transition = document.states[1].transitions[0]
    assert set(transition.operand_ids).issubset(
        set(transition.previous_object_states)
        | set(transition.next_object_states)
    )
