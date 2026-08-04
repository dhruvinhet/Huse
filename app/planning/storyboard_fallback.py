"""High-level concept grounding and provider-independent intent fallback."""

from math import ceil

from app.domain.lesson import ConceptNode, ConceptRelation, LessonPlan
from app.domain.operations import OperationType
from app.domain.pedagogy import PedagogyMode, PedagogyPlan
from app.domain.storyboard import Storyboard, VisualBeat, VisualObjectSpec
from app.domain.visual_intent import RendererOperator, ShotSpec, VisualIntent
from app.planning.visual_intent_compiler import VisualIntentCompiler


class ConceptGraphStoryboardBuilder:
    """Build high-level visual intent from any validated concept graph."""

    def __init__(self) -> None:
        self._compiler = VisualIntentCompiler()

    def build(
        self,
        lesson: LessonPlan,
        previous: Storyboard | None = None,
        pedagogy: PedagogyPlan | None = None,
    ) -> Storyboard:
        """Compile deterministic intent without directly authoring scene objects."""

        intent = self._intent(lesson, pedagogy)
        board = self._compiler.compile(intent, lesson, pedagogy)
        beats: list[VisualBeat] = []
        for index, beat in enumerate(board.beats):
            prior = (
                previous.beats[index]
                if previous is not None and index < len(previous.beats)
                else None
            )
            updates: dict[str, object] = {}
            if prior is not None and self._useful_phrase(prior.phrase_intent):
                updates["phrase_intent"] = prior.phrase_intent
                updates["estimated_duration"] = prior.estimated_duration
            if index == len(board.beats) - 1:
                updates["beat_id"] = "beat_summary"
            beats.append(beat.model_copy(update=updates))
        return board.model_copy(
            update={
                "document_id": (
                    previous.document_id
                    if previous is not None
                    else board.document_id
                ),
                "beats": beats,
            }
        )

    def ground(
        self,
        storyboard: Storyboard,
        lesson: LessonPlan,
        pedagogy: PedagogyPlan | None = None,
    ) -> Storyboard:
        """Retain strong compiled output and replace incomplete teaching plans."""

        valid_ids = set(lesson.concept_graph.teaching_sequence)
        nodes = {node.concept_id: node for node in lesson.concept_graph.nodes}
        grounded_beats: list[VisualBeat] = []
        for index, beat in enumerate(storyboard.beats):
            concept_ids = [item for item in beat.concept_ids if item in valid_ids]
            searchable = (
                f"{beat.teaching_intent} {beat.phrase_intent} "
                f"{beat.model_dump(mode='json')}"
            ).lower()
            for concept_id in lesson.concept_graph.teaching_sequence:
                if (
                    concept_id not in concept_ids
                    and nodes[concept_id].label.lower() in searchable
                ):
                    concept_ids.append(concept_id)
            if not concept_ids:
                concept_ids = [
                    lesson.concept_graph.teaching_sequence[
                        min(index, len(lesson.concept_graph.teaching_sequence) - 1)
                    ]
                ]
            grounded_beats.append(beat.model_copy(update={"concept_ids": concept_ids}))

        grounded = storyboard.model_copy(update={"beats": grounded_beats})
        represented = {
            concept_id
            for beat in grounded.beats
            for concept_id in beat.concept_ids
        }
        important = {
            node.concept_id
            for node in lesson.concept_graph.nodes
            if node.importance >= 0.5
        }
        if not important.issubset(represented) or self._is_visually_weak(grounded):
            return self.build(lesson, grounded, pedagogy)
        if grounded.final_learning_summary:
            return grounded
        return grounded.model_copy(
            update={
                "final_learning_summary": [
                    f"{node.label}: {node.definition}"
                    for node in lesson.concept_graph.nodes
                    if node.importance >= 0.5
                ]
            }
        )

    def _intent(
        self,
        lesson: LessonPlan,
        pedagogy: PedagogyPlan | None,
    ) -> VisualIntent:
        """Represent graph facts as operator-level shots only."""

        nodes = self._ordered_nodes(lesson)
        if pedagogy is None:
            content_count = min(5, len(nodes))
            chunk = max(1, ceil(len(nodes) / content_count))
            groups = [
                [node.concept_id for node in nodes[index:index + chunk]]
                for index in range(0, len(nodes), chunk)
            ]
            groups.append([node.concept_id for node in nodes])
            shot_ids = [
                f"concept_{index + 1:03d}"
                for index in range(len(groups) - 1)
            ]
            shot_ids.append("summary")
        else:
            groups = self._pedagogy_groups(lesson, pedagogy)
            shot_ids = [shot.shot_id for shot in pedagogy.shots]

        operator = self._operator(lesson, pedagogy)
        by_id = {node.concept_id: node for node in nodes}
        shots: list[ShotSpec] = []
        for index, concept_ids in enumerate(groups):
            selected = [by_id[item] for item in concept_ids]
            relation = self._relation_for(lesson, concept_ids)
            routed = (
                pedagogy.shots[index]
                if pedagogy is not None and index < len(pedagogy.shots)
                else None
            )
            shots.append(
                ShotSpec(
                    shot_id=shot_ids[index],
                    concept_ids=concept_ids,
                    relation=relation,
                    focal_object=self._bounded_join(
                        [node.label for node in selected],
                        " / ",
                        120,
                    ),
                    evidence=self._bounded_join(
                        [
                            f"{node.label}: {node.definition}"
                            for node in selected
                        ],
                        " ",
                        280,
                    ),
                    transformation=(
                        routed.visual_obligation
                        if routed is not None
                        else "Reveal the concept and its factual relationship."
                    ),
                    renderer_operator=operator,
                )
            )
        return VisualIntent(
            lesson_focus=lesson.title,
            shots=shots,
        )

    @staticmethod
    def _pedagogy_groups(
        lesson: LessonPlan,
        pedagogy: PedagogyPlan,
    ) -> list[list[str]]:
        """Assign distinct semantic focus to each rhetorical shot."""

        sequence = list(lesson.concept_graph.teaching_sequence)
        relational = list(dict.fromkeys(
            concept_id
            for edge in lesson.concept_graph.edges
            for concept_id in (edge.source_id, edge.target_id)
        ))
        content_shots = [
            shot
            for shot in pedagogy.shots
            if shot.purpose not in {"introduce", "connect", "summarize"}
        ]
        width = max(1, ceil(len(sequence) / max(1, len(content_shots))))
        content_index = 0
        groups: list[list[str]] = []
        for shot in pedagogy.shots:
            if shot.purpose == "summarize":
                group = sequence
            elif shot.purpose == "connect":
                group = relational or sequence
            elif shot.purpose == "introduce":
                group = (
                    sequence
                    if pedagogy.mode in {
                        PedagogyMode.CONCEPT_OVERVIEW,
                        PedagogyMode.CONCEPT_SET,
                    }
                    else sequence[:1]
                )
            elif pedagogy.mode is PedagogyMode.MECHANISM_FIRST:
                group = sequence[1:] or sequence
            else:
                start = min(content_index * width, len(sequence) - 1)
                group = sequence[start:start + width] or [sequence[start]]
                content_index += 1
            groups.append(list(group))
        return groups

    @staticmethod
    def _bounded_join(
        parts: list[str],
        separator: str,
        limit: int,
    ) -> str:
        """Join complete factual clauses without exceeding provider contracts."""

        result: list[str] = []
        for part in parts:
            candidate = separator.join([*result, part])
            if len(candidate) > limit:
                break
            result.append(part)
        if result:
            return separator.join(result)
        return parts[0][:limit].rstrip() if parts else "Lesson concept"

    @staticmethod
    def _operator(
        lesson: LessonPlan,
        pedagogy: PedagogyPlan | None,
    ) -> RendererOperator:
        """Choose a supported high-level operator from semantics."""

        if pedagogy is not None:
            routed = {
                # Overview and collection lessons need a graph-preserving
                # group root.  Specialized spatial/card roots can discard
                # dependency edges, which makes a deterministic fallback fail
                # the semantic-fidelity gate even though the lesson graph is
                # valid.
                PedagogyMode.CONCEPT_OVERVIEW: RendererOperator.GROUP,
                PedagogyMode.CONCEPT_SET: RendererOperator.GROUP,
                PedagogyMode.COMPARISON: RendererOperator.COMPARISON,
                PedagogyMode.CHRONOLOGICAL: RendererOperator.TIMELINE,
                PedagogyMode.PROOF_DERIVATION: RendererOperator.EQUATION,
                PedagogyMode.CODE_EXECUTION: RendererOperator.CODE_TRACE,
                PedagogyMode.SIMULATION: RendererOperator.SIMULATION,
                PedagogyMode.SPATIAL_ANATOMY: RendererOperator.ANATOMY,
            }.get(pedagogy.mode)
            if routed is not None:
                return routed
        terms = {
            str(value).casefold().replace("-", "_").replace(" ", "_")
            for node in lesson.concept_graph.nodes
            for value in [node.label, *node.visual_affordances]
        }
        if terms & {"tree", "hierarchy", "heap"}:
            return RendererOperator.TREE
        if terms & {"array", "list"}:
            return RendererOperator.ARRAY
        if terms & {"timeline", "chronology"}:
            return RendererOperator.TIMELINE
        relations = {edge.relation for edge in lesson.concept_graph.edges}
        if ConceptRelation.CONTRASTS_WITH in relations:
            return RendererOperator.COMPARISON
        if ConceptRelation.CAUSES in relations:
            return RendererOperator.CAUSE_EFFECT
        if relations & {ConceptRelation.FLOWS_TO, ConceptRelation.TRANSFORMS_TO}:
            return RendererOperator.PROCESS
        if lesson.concept_graph.edges:
            return RendererOperator.GRAPH
        return RendererOperator.SPATIAL

    @staticmethod
    def _relation_for(
        lesson: LessonPlan,
        concept_ids: list[str],
    ) -> ConceptRelation | None:
        selected = set(concept_ids)
        for edge in lesson.concept_graph.edges:
            if edge.source_id in selected or edge.target_id in selected:
                return edge.relation
        return None

    @staticmethod
    def _ordered_nodes(lesson: LessonPlan) -> list[ConceptNode]:
        by_id = {node.concept_id: node for node in lesson.concept_graph.nodes}
        return [by_id[item] for item in lesson.concept_graph.teaching_sequence]

    @staticmethod
    def _is_visually_weak(storyboard: Storyboard) -> bool:
        """Detect invalid external storyboards without constructing replacements."""

        definitions = 0
        meaningful = 0
        for beat in storyboard.beats:
            for operation in beat.operations:
                if operation.operation is not OperationType.CREATE:
                    continue
                raw_objects = operation.arguments.get("objects")
                if not isinstance(raw_objects, list):
                    continue
                for raw in raw_objects:
                    if not isinstance(raw, dict):
                        continue
                    item = VisualObjectSpec.model_validate(raw)
                    for flattened in item.flatten():
                        definitions += 1
                        if (
                            flattened.children
                            or flattened.content.get("label")
                            or flattened.content.get("text")
                            or flattened.kind == "connector"
                        ):
                            meaningful += 1
        return definitions == 0 or meaningful / definitions < 0.6

    @staticmethod
    def _useful_phrase(value: str) -> bool:
        normalized = value.strip().casefold()
        return len(normalized.split()) >= 4 and "_visual" not in normalized
