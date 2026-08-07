"""Build deterministic shot-local concept graphs for template selection."""

from app.domain.lesson import ConceptGraph, LessonPlan
from app.domain.pedagogy import PedagogyPlan, PedagogyShot


def shot_concept_groups(
    lesson: LessonPlan,
    pedagogy: PedagogyPlan,
) -> list[list[str]]:
    """Ground rhetorical shots in the concepts they actually teach."""

    sequence = list(lesson.concept_graph.teaching_sequence)
    parent_of = {
        edge.source_id: edge.target_id
        for edge in lesson.concept_graph.edges
        if edge.relation.value == "part_of"
    }
    whole_ids = [
        concept_id
        for concept_id in sequence
        if concept_id not in parent_of and concept_id in set(parent_of.values())
    ]
    component_ids = [concept_id for concept_id in sequence if concept_id in parent_of]
    relational_ids = list(dict.fromkeys(
        concept_id
        for edge in lesson.concept_graph.edges
        for concept_id in (edge.source_id, edge.target_id)
    ))
    groups: list[list[str]] = []
    content_shots = [
        shot
        for shot in pedagogy.shots
        if shot.purpose not in {"introduce", "connect", "summarize"}
    ]
    content_index = {
        shot.shot_id: index for index, shot in enumerate(content_shots)
    }
    for index, shot in enumerate(pedagogy.shots):
        if shot.purpose == "summarize":
            groups.append(sequence)
        elif shot.purpose == "introduce":
            groups.append(
                sequence
                if pedagogy.mode.value in {"concept_overview", "concept_set"}
                else whole_ids or sequence[:1]
            )
        elif shot.purpose == "connect":
            groups.append(relational_ids or sequence)
        elif pedagogy.mode.value == "mechanism_first":
            groups.append(sequence[1:] or sequence)
        elif component_ids:
            groups.append(component_ids)
        else:
            width = max(
                1,
                (len(sequence) + max(1, len(content_shots)) - 1)
                // max(1, len(content_shots)),
            )
            start = min(
                content_index.get(shot.shot_id, index) * width,
                len(sequence) - 1,
            )
            groups.append(sequence[start:start + width] or [sequence[start]])
    return groups


def shot_query_graph(
    graph: ConceptGraph,
    shot: PedagogyShot,
    concept_ids: list[str],
) -> ConceptGraph:
    """Return the induced/one-hop graph required by one teaching shot."""

    selected = set(concept_ids)
    if not graph.edges and len(selected) < min(3, len(graph.nodes)):
        # Stateless algorithm/topic graphs still need enough local operands
        # to instantiate their reviewed example, while relation-bearing mixed
        # lessons remain strictly scoped to the current shot.
        selected.update(graph.teaching_sequence[:3])
    edges = [
        edge
        for edge in graph.edges
        if edge.source_id in selected and edge.target_id in selected
    ]
    if not edges and (
        shot.purpose in {"connect", "transform", "demonstrate"}
        or len(selected) < 2
    ):
        edges = [
            edge
            for edge in graph.edges
            if edge.source_id in selected or edge.target_id in selected
        ]
        selected.update(
            concept_id
            for edge in edges
            for concept_id in (edge.source_id, edge.target_id)
        )
    nodes = [node for node in graph.nodes if node.concept_id in selected][:12]
    if not nodes:
        nodes = graph.nodes[:1]
    selected = {node.concept_id for node in nodes}
    nodes = [
        node.model_copy(update={
            "prerequisites": [
                prerequisite
                for prerequisite in node.prerequisites
                if prerequisite in selected
            ]
        })
        for node in nodes
    ]
    sequence = [
        concept_id
        for concept_id in graph.teaching_sequence
        if concept_id in selected
    ]
    return ConceptGraph(
        objectives=[
            shot.visual_obligation,
            shot.narration_obligation,
            *graph.objectives,
        ],
        nodes=nodes,
        edges=[
            edge
            for edge in edges
            if edge.source_id in selected and edge.target_id in selected
        ],
        teaching_sequence=sequence,
    )
