"""Compile compact visual intent into a persistent semantic storyboard."""

import re

from app.domain.lesson import LessonPlan
from app.domain.operations import OperationType, VisualOperation
from app.domain.pedagogy import PedagogyPlan, PedagogyShot
from app.domain.storyboard import (
    AttentionCue,
    CameraIntent,
    ShotPlan,
    Storyboard,
    VisualBeat,
    VisualObjectSpec,
)
from app.domain.visual_intent import (
    OperatorInstance,
    ShotSpec,
    VisualIntent,
    VisualProgram,
    VisualProgramShot,
)
from app.templates.operator_templates import (
    GenericOperatorParameters,
    PARAMETER_MODELS,
    SemanticOperatorCompiler,
)


MAX_ROOT_OPERANDS = 10
MAX_ROOT_CONCEPTS = 12
MAX_ROOT_RELATIONS = 24


class VisualIntentCompiler:
    """Own all low-level IDs, hierarchy, constraints, and transitions."""

    def __init__(self) -> None:
        self.last_program: VisualProgram | None = None

    def compile(
        self,
        intent: VisualIntent,
        lesson: LessonPlan,
        pedagogy: PedagogyPlan | None = None,
    ) -> Storyboard:
        """Expand high-level shots into a validated persistent visual program."""

        self.validate_intent(intent, lesson, pedagogy)
        self.last_program = self._build_program(intent, lesson)
        distinct_operators = {
            item.operator for item in self.last_program.operator_instances
        }
        if len(distinct_operators) > 1 or any(
            shot.supporting_operators for shot in intent.shots
        ):
            return self._compile_multi_operator(
                intent,
                lesson,
                pedagogy,
                self.last_program,
            )
        ids = self._object_ids(lesson)
        first = intent.shots[0]
        root = self._initial_hierarchy(first, lesson, ids)
        self._compiled_objects = root.flatten()
        beats = [
            self._compile_shot(
                shot,
                index,
                lesson,
                pedagogy,
                ids,
                root if index == 0 else None,
            )
            for index, shot in enumerate(intent.shots)
        ]
        important = [
            f"{node.label}: {node.definition}"
            for node in lesson.concept_graph.nodes
            if node.importance >= 0.5
        ]
        return Storyboard(
            document_id=f"intent_{self._safe_id(lesson.title)}",
            title=lesson.title,
            beats=beats,
            final_learning_summary=important,
        )

    @staticmethod
    def validate_intent(
        intent: VisualIntent,
        lesson: LessonPlan,
        pedagogy: PedagogyPlan | None = None,
    ) -> None:
        """Reject unknown concepts or violations of routed shot grammar."""

        known = {node.concept_id for node in lesson.concept_graph.nodes}
        referenced = {
            concept_id
            for shot in intent.shots
            for concept_id in shot.concept_ids
        }
        unknown = sorted(referenced - known)
        if unknown:
            raise ValueError(f"visual intent references unknown concepts: {unknown}")
        important = {
            node.concept_id
            for node in lesson.concept_graph.nodes
            if node.importance >= 0.5
        }
        missing = sorted(important - referenced)
        if missing:
            raise ValueError(f"visual intent omits important concepts: {missing}")
        if pedagogy is None:
            return
        expected = [shot.shot_id for shot in pedagogy.shots]
        actual = [shot.shot_id for shot in intent.shots]
        if actual != expected:
            raise ValueError(
                "visual intent shots must exactly match routed shot IDs and order"
            )

    def _build_program(
        self,
        intent: VisualIntent,
        lesson: LessonPlan,
    ) -> VisualProgram:
        """Declare operator ownership and cleanup before lowering scene objects."""

        program_id = f"program_{self._safe_id(lesson.title)}"
        legacy_shared = (
            len({shot.renderer_operator for shot in intent.shots}) == 1
            and all(not shot.supporting_operators for shot in intent.shots)
        )
        instances: list[OperatorInstance] = []
        program_shots: list[VisualProgramShot] = []
        shared_ids: list[str] = []
        previous_local: list[str] = []
        legacy_id = f"{program_id}_shared_primary"
        if legacy_shared:
            shared_ids.append(legacy_id)
            instances.append(OperatorInstance(
                instance_id=legacy_id,
                operator=intent.shots[0].renderer_operator,
                concept_ids=list(dict.fromkeys(
                    concept_id
                    for shot in intent.shots
                    for concept_id in shot.concept_ids
                )),
                ownership="shared",
                action_obligations=list(dict.fromkeys(
                    obligation
                    for shot in intent.shots
                    for obligation in (
                        shot.action_obligations or [shot.transformation]
                    )
                ))[:6],
            ))
        for index, shot in enumerate(intent.shots):
            shot_instances: list[str] = []
            if legacy_shared:
                shot_instances.append(legacy_id)
            else:
                operators = [shot.renderer_operator, *shot.supporting_operators]
                regions = ["full", "right", "bottom"]
                for operator_index, operator in enumerate(operators):
                    instance_id = (
                        f"{program_id}_{index:02d}_{operator_index:02d}_"
                        f"{operator.value}"
                    )
                    instances.append(OperatorInstance(
                        instance_id=instance_id,
                        operator=operator,
                        concept_ids=shot.concept_ids,
                        ownership="shot",
                        layout_region=regions[operator_index],
                        state_ref=shot.state_ref,
                        action_obligations=(
                            shot.action_obligations or [shot.transformation]
                        ),
                    ))
                    shot_instances.append(instance_id)
            program_shots.append(VisualProgramShot(
                shot_id=shot.shot_id,
                operator_instance_ids=shot_instances,
                state_ref=shot.state_ref,
                cleanup_instance_ids=list(previous_local),
                action_obligations=(
                    shot.action_obligations or [shot.transformation]
                ),
            ))
            previous_local = [
                instance_id
                for instance_id in shot_instances
                if instance_id not in shared_ids
            ]
        return VisualProgram(
            program_id=program_id,
            shared_instance_ids=shared_ids,
            operator_instances=instances,
            shots=program_shots,
        )

    def _compile_multi_operator(
        self,
        intent: VisualIntent,
        lesson: LessonPlan,
        pedagogy: PedagogyPlan | None,
        program: VisualProgram,
    ) -> Storyboard:
        """Lower shot-local operator roots with deterministic ownership cleanup."""

        instances = {item.instance_id: item for item in program.operator_instances}
        roots: dict[str, VisualObjectSpec] = {}
        for instance in program.operator_instances:
            scoped = self._safe_id(instance.instance_id)
            ids = {
                "root": f"{scoped}_root",
                "evidence": f"{scoped}_evidence",
            }
            source_shot = next(
                shot
                for shot in intent.shots
                if instance.instance_id in next(
                    item.operator_instance_ids
                    for item in program.shots
                    if item.shot_id == shot.shot_id
                )
            )
            operator_shot = source_shot.model_copy(
                update={"renderer_operator": instance.operator}
            )
            root = self._initial_hierarchy(operator_shot, lesson, ids)
            root.content.update({
                "program_instance_id": instance.instance_id,
                "ownership": instance.ownership,
                "layout_region": instance.layout_region,
            })
            roots[instance.instance_id] = root

        created: set[str] = set()
        beats: list[VisualBeat] = []
        for index, (shot, program_shot) in enumerate(
            zip(intent.shots, program.shots, strict=True)
        ):
            route = self._routed_shot(pedagogy, index)
            operations: list[VisualOperation] = []
            for stale_id in program_shot.cleanup_instance_ids:
                stale = roots[stale_id]
                operations.append(VisualOperation(
                    operation_id=f"intent_{index:02d}_cleanup_{self._safe_id(stale_id)}",
                    operation=OperationType.HIDE,
                    target_ids=[item.object_id for item in stale.flatten()],
                ))
            entering = [
                instance_id
                for instance_id in program_shot.operator_instance_ids
                if instance_id not in created
            ]
            if entering:
                definitions = [roots[instance_id] for instance_id in entering]
                operations.append(VisualOperation(
                    operation_id=f"intent_{index:02d}_create_program_roots",
                    operation=OperationType.CREATE,
                    target_ids=[item.object_id for item in definitions],
                    arguments={
                        "objects": [item.model_dump(mode="json") for item in definitions]
                    },
                ))
                created.update(entering)
            retained = [
                instance_id
                for instance_id in program_shot.operator_instance_ids
                if instance_id in created and instance_id not in entering
            ]
            if retained:
                operations.append(VisualOperation(
                    operation_id=f"intent_{index:02d}_show_shared",
                    operation=OperationType.SHOW,
                    target_ids=[
                        item.object_id
                        for instance_id in retained
                        for item in roots[instance_id].flatten()
                    ],
                ))
            active_objects = [
                item
                for instance_id in program_shot.operator_instance_ids
                for item in roots[instance_id].flatten()
            ]
            targets = [
                item.object_id
                for item in active_objects
                if item.kind != "connector"
                and set(item.concept_ids).intersection(shot.concept_ids)
            ] or [roots[program_shot.operator_instance_ids[0]].object_id]
            operations.append(VisualOperation(
                operation_id=f"intent_{index:02d}_highlight_program",
                operation=OperationType.HIGHLIGHT,
                target_ids=list(dict.fromkeys(targets)),
            ))
            purpose = route.purpose if route is not None else (
                "introduce" if index == 0 else "summarize"
            )
            beats.append(VisualBeat(
                beat_id=f"shot_{index + 1:02d}_{self._safe_id(shot.shot_id)}",
                section_id=(
                    f"pedagogy_{pedagogy.mode.value}"
                    if pedagogy is not None else "visual_program"
                ),
                concept_ids=shot.concept_ids,
                teaching_intent=(
                    f"{route.visual_obligation} {shot.transformation}"
                    if route is not None else shot.transformation
                ),
                phrase_intent=self._natural_phrase_intent(shot, lesson, purpose),
                purpose=purpose,
                estimated_duration=10.0 if index == 0 else 6.0,
                operations=operations,
                attention=[AttentionCue(
                    cue="focus",
                    target_ids=list(dict.fromkeys(targets)),
                    intensity=0.85,
                )],
                camera_intent=CameraIntent(
                    operation=(route.camera_operation if route else "hold"),
                    target_ids=[roots[program_shot.operator_instance_ids[0]].object_id],
                ),
                shot_plan=ShotPlan.for_purpose(purpose).model_copy(
                    update={"cleanup_policy": "retain"}
                ),
            ))
        return Storyboard(
            document_id=f"intent_{self._safe_id(lesson.title)}",
            title=lesson.title,
            beats=beats,
            final_learning_summary=[
                f"{node.label}: {node.definition}"
                for node in lesson.concept_graph.nodes
                if node.importance >= 0.5
            ],
        )

    def _initial_hierarchy(
        self,
        shot: ShotSpec,
        lesson: LessonPlan,
        ids: dict[str, str],
    ) -> VisualObjectSpec:
        """Create the one stable scene graph reused by every shot."""

        by_id = {
            node.concept_id: node for node in lesson.concept_graph.nodes
        }
        overview_ids = set(lesson.concept_graph.teaching_sequence[:MAX_ROOT_CONCEPTS])
        overview_nodes = [
            node
            for node in lesson.concept_graph.nodes
            if node.concept_id in overview_ids
        ]
        overview_relations = [
            edge
            for edge in lesson.concept_graph.edges
            if edge.source_id in overview_ids and edge.target_id in overview_ids
        ][:MAX_ROOT_RELATIONS]
        parameters = {
            "object_id": ids["root"],
            "label": lesson.title,
            "operands": [
                by_id[item].label
                for item in lesson.concept_graph.teaching_sequence[:MAX_ROOT_OPERANDS]
                if item in by_id
            ],
            "concepts": [
                {
                    "concept_id": node.concept_id,
                    "label": node.label,
                    "definition": node.definition,
                    "importance": node.importance,
                    "order": node.teaching_order,
                    "visual_affordances": node.visual_affordances,
                }
                for node in overview_nodes
            ],
            "relations": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation": edge.relation.value,
                    "label": edge.label,
                }
                for edge in overview_relations
            ],
        }
        model = PARAMETER_MODELS.get(
            shot.renderer_operator,
            GenericOperatorParameters,
        )
        validated = model.model_validate(parameters)
        root = SemanticOperatorCompiler().compile(
            shot.renderer_operator,
            validated,
        )
        root = self._ensure_relation_connectors(
            root,
            overview_relations,
            lesson,
        )
        evidence = VisualObjectSpec(
            object_id=ids["evidence"],
            kind="callout",
            semantic_role="evidence",
            concept_ids=shot.concept_ids,
            content={
                "text": shot.evidence,
                "transformation": shot.transformation,
            },
            style_token="annotation",
            accessibility_label=shot.evidence,
        )
        root.children.append(evidence)
        return root.model_copy(
            update={
                "semantic_role": "compiled_visual_intent",
                "accessibility_label": f"{lesson.title} visual explanation",
            }
        )

    def _compile_shot(
        self,
        shot: ShotSpec,
        index: int,
        lesson: LessonPlan,
        pedagogy: PedagogyPlan | None,
        ids: dict[str, str],
        root: VisualObjectSpec | None,
    ) -> VisualBeat:
        """Compile one intent shot into only allowed deterministic mutations."""

        route = self._routed_shot(pedagogy, index)
        known_objects = getattr(self, "_compiled_objects", [])
        concept_targets = [
            item.object_id
            for item in known_objects
            if item.kind != "connector"
            and item.object_id != ids["root"]
            and set(item.concept_ids).intersection(shot.concept_ids)
        ]
        relation_targets = [
            item.object_id
            for item in known_objects
            if item.kind == "connector"
            and set(item.concept_ids).intersection(shot.concept_ids)
        ]
        targets = list(dict.fromkeys(
            [*concept_targets, *relation_targets]
        )) or [ids["root"]]
        reset_targets = [
            item.object_id
            for item in known_objects
            if item.object_id != ids["root"]
        ]
        if root is not None:
            operations = [
                VisualOperation(
                    operation_id=f"intent_{index:02d}_create",
                    operation=OperationType.CREATE,
                    target_ids=[ids["root"]],
                    arguments={"objects": [root.model_dump(mode="json")]},
                )
            ]
        else:
            operations = [
                VisualOperation(
                    operation_id=f"intent_{index:02d}_evidence",
                    operation=OperationType.UPDATE,
                    target_ids=[ids["evidence"]],
                    arguments={
                        "content": {
                            "text": shot.evidence,
                            "transformation": shot.transformation,
                        }
                    },
                )
            ]
        if reset_targets:
            operations.append(
                VisualOperation(
                    operation_id=f"intent_{index:02d}_show",
                    operation=OperationType.SHOW,
                    target_ids=reset_targets,
                )
            )
        purpose = route.purpose if route is not None else (
            "introduce" if index == 0 else "summarize"
        )
        if purpose != "summarize":
            if reset_targets:
                operations.append(
                    VisualOperation(
                        operation_id=f"intent_{index:02d}_dim",
                        operation=OperationType.DIM,
                        target_ids=reset_targets,
                    )
                )
            operations.append(
                VisualOperation(
                    operation_id=f"intent_{index:02d}_highlight",
                    operation=OperationType.HIGHLIGHT,
                    target_ids=targets,
                )
            )
        visual_obligation = (
            route.visual_obligation
            if route is not None
            else shot.transformation
        )
        return VisualBeat(
            beat_id=f"shot_{index + 1:02d}_{self._safe_id(shot.shot_id)}",
            section_id=(
                f"pedagogy_{pedagogy.mode.value}"
                if pedagogy is not None
                else "visual_intent"
            ),
            concept_ids=shot.concept_ids,
            teaching_intent=f"{visual_obligation} {shot.transformation}",
            phrase_intent=self._natural_phrase_intent(
                shot,
                lesson,
                purpose,
            ),
            purpose=purpose,
            estimated_duration=10.0 if index == 0 else 6.0,
            operations=operations,
            attention=[
                AttentionCue(
                    cue="focus",
                    target_ids=targets,
                    intensity=0.85,
                )
            ],
            camera_intent=CameraIntent(
                operation=(route.camera_operation if route else "hold"),
                target_ids=[ids["root"]],
            ),
            shot_plan=ShotPlan.for_purpose(purpose),
        )

    @staticmethod
    def _ensure_relation_connectors(
        root: VisualObjectSpec,
        relations: list[object],
        lesson: LessonPlan,
    ) -> VisualObjectSpec:
        """Ground every graph edge even when an operator has custom geometry.

        Some educational operators intentionally use specialised children and
        therefore do not expose every lesson edge.  The semantic contract is
        still authoritative: add a small explicit connector for any missing
        edge so quality validation and the renderer cannot silently lose the
        relationship.
        """

        objects = root.flatten()
        by_concept: dict[str, VisualObjectSpec] = {}
        existing: set[tuple[str, str, str]] = set()
        for item in objects:
            if item.kind == "connector":
                relation = str(item.content.get("relation", ""))
                for source in item.concept_ids[:1]:
                    for target in item.concept_ids[1:2]:
                        existing.add((source, target, relation))
                continue
            for concept_id in item.concept_ids:
                by_concept.setdefault(concept_id, item)

        labels = {
            node.concept_id: node.label
            for node in lesson.concept_graph.nodes
        }
        additions: list[VisualObjectSpec] = []
        for index, edge in enumerate(relations):
            source = by_concept.get(edge.source_id)
            target = by_concept.get(edge.target_id)
            relation = edge.relation.value
            if source is None or target is None:
                continue
            if (edge.source_id, edge.target_id, relation) in existing:
                continue
            label = edge.label or relation.replace("_", " ")
            additions.append(
                VisualObjectSpec(
                    object_id=f"{root.object_id}_grounded_relation_{index:03d}",
                    kind="connector",
                    semantic_role=f"grounded_relation_{relation}",
                    concept_ids=[edge.source_id, edge.target_id],
                    content={
                        "source_id": source.object_id,
                        "target_id": target.object_id,
                        "relation": relation,
                        "label": label,
                    },
                    accessibility_label=(
                        f"{labels.get(edge.source_id, edge.source_id)} "
                        f"{label} {labels.get(edge.target_id, edge.target_id)}"
                    ),
                )
            )
        if not additions:
            return root
        return root.model_copy(update={"children": [*root.children, *additions]})

    @staticmethod
    def _natural_phrase_intent(
        shot: ShotSpec,
        lesson: LessonPlan,
        purpose: str,
    ) -> str:
        """Produce factual fallback speech without exposing prompt obligations."""

        by_id = {
            node.concept_id: node for node in lesson.concept_graph.nodes
        }
        labels = ", ".join(
            by_id[item].label
            for item in shot.concept_ids
            if item in by_id
        )
        evidence = shot.evidence.rstrip(".")
        if purpose == "summarize":
            objective = lesson.concept_graph.objectives[0].rstrip(".")
            return f"In summary, {evidence}. This explains how to {objective}."
        if purpose == "connect" and shot.relation is not None:
            relation = shot.relation.value.replace("_", " ")
            return f"For {labels}, the key relationship is {relation}. {evidence}."
        if purpose in {"transform", "demonstrate"}:
            return f"Watch {labels}: {evidence}."
        return f"We begin with {labels}. {evidence}."

    @staticmethod
    def _object_ids(lesson: LessonPlan) -> dict[str, str]:
        prefix = VisualIntentCompiler._safe_id(lesson.title)
        return {
            "root": f"{prefix}_scene",
            "evidence": f"{prefix}_evidence",
        }

    @staticmethod
    def _routed_shot(
        pedagogy: PedagogyPlan | None,
        index: int,
    ) -> PedagogyShot | None:
        if pedagogy is None or index >= len(pedagogy.shots):
            return None
        return pedagogy.shots[index]

    @staticmethod
    def _safe_id(value: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_") or "lesson"
