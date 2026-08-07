"""Authoritative compilation of reviewed semantic templates."""

import re

from loguru import logger

from app.application.ports.knowledge import TemplateLibrary
from app.domain.lesson import LessonPlan
from app.domain.operations import (
    CompareAction,
    ConsumeAction,
    GroupAction,
    MergeAction,
    OperationType,
    ProduceAction,
    RouteAction,
    SemanticAction,
    SplitAction,
    TransferAction,
    TransformAction,
    VisualOperation,
)
from app.domain.generation import AudienceProfile
from app.domain.pedagogy import PedagogyPlan
from app.domain.semantic_bounds import (
    MAX_SEMANTIC_CONCEPTS,
    MAX_SEMANTIC_RELATIONS,
    bound_semantic_operands,
)
from app.domain.storyboard import (
    AttentionCue,
    CameraIntent,
    ShotPlan,
    Storyboard,
    VisualBeat,
    VisualObjectSpec,
)
from app.domain.strategy import (
    CompiledTemplateProgram,
    ParameterProvenance,
    TemplateMatch,
    VisualStrategy,
)
from app.domain.visual_intent import RendererOperator
from app.novelty import REVIEWED_LAYOUT_VARIANTS
from app.planning.shot_graph import shot_concept_groups


class TemplateCompiler:
    """Select, instantiate, ground, and animate a reviewed template."""

    _GENERIC_TEMPLATE_IDS = {
        "array.v1",
        "pipeline.v1",
        "tree.v1",
        "graph.v1",
        "matrix.v1",
        "probability_distribution.v1",
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
    }
    _COMPONENT_PARAMETER_TEMPLATES = {
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
    }
    _TEMPLATE_PARAMETER_FIELDS = {
        "array.v1": {"values"},
        "pipeline.v1": {"stages"},
        "tree.v1": {"nodes"},
        "graph.v1": {"nodes"},
        "matrix.v1": {"rows", "columns", "values"},
        "probability_distribution.v1": {"values", "labels"},
        "transformer_block.v1": {"components"},
    }

    def compile(
        self,
        lesson: LessonPlan,
        strategies: list[VisualStrategy],
        matches: list[TemplateMatch],
        library: TemplateLibrary,
        pedagogy: PedagogyPlan,
        target_duration: float = 60.0,
        audience: AudienceProfile | None = None,
        shot_local_matching: bool = False,
    ) -> CompiledTemplateProgram | None:
        """Return a deterministic program for the strongest match, if any."""

        groups = shot_concept_groups(lesson, pedagogy)
        local_matcher = getattr(library, "match_shots", None)
        if shot_local_matching and callable(local_matcher):
            local_matches = local_matcher(
                lesson.concept_graph,
                pedagogy,
                groups,
                audience,
            )
            if local_matches:
                matches = local_matches
        section_selections = self._select_for_shots(
            lesson,
            strategies,
            matches,
            pedagogy,
        )
        if len({item.template_id for item in section_selections}) > 1:
            return self._compile_sections(
                lesson,
                section_selections,
                library,
                pedagogy,
                target_duration,
            )
        selection = (
            max(section_selections, key=lambda item: item.score)
            if section_selections
            else self._select(lesson, strategies, matches)
        )
        if selection is None:
            return None
        parameters = self._validated_parameters(selection, lesson, library)
        parameter_provenance = dict(self._latest_parameter_provenance)
        default_usage = list(self._latest_default_usage)
        raw_concept_count = len(parameters.get("concepts", []))
        raw_relation_count = len(parameters.get("relations", []))
        parameters = bound_semantic_operands(
            parameters,
            preferred_concept_ids=selection.concept_ids,
        )
        if (
            raw_concept_count != len(parameters.get("concepts", []))
            or raw_relation_count != len(parameters.get("relations", []))
        ):
            logger.warning(
                "Bound template '{}' semantic operands from {} concepts/{} "
                "relations to {}/{} for one visual shot",
                selection.template_id,
                raw_concept_count,
                raw_relation_count,
                MAX_SEMANTIC_CONCEPTS,
                MAX_SEMANTIC_RELATIONS,
            )
        root = library.instantiate(selection.template_id, parameters)
        root = self._apply_layout_variant(root, selection)
        root = self._ground_object(root, lesson)
        root = self._mark_connector_ownership(root)
        storyboard = self._compile_storyboard(
            root,
            lesson,
            pedagogy,
            target_duration,
            selection,
        )
        return CompiledTemplateProgram(
            template_id=selection.template_id,
            template_ids=[selection.template_id],
            shot_template_ids={
                shot.shot_id: selection.template_id for shot in pedagogy.shots
            },
            parameters=parameters,
            parameter_provenance=parameter_provenance,
            default_usage=default_usage,
            match_confidence=selection.match_confidence,
            capability_evidence=selection.capability_evidence,
            pedagogy_mode=pedagogy.mode,
            storyboard=storyboard,
            layout_variants={selection.template_id: selection.layout_variant},
        )

    def _select_for_shots(
        self,
        lesson: LessonPlan,
        strategies: list[VisualStrategy],
        matches: list[TemplateMatch],
        pedagogy: PedagogyPlan,
    ) -> list[TemplateMatch]:
        """Choose a compatible reviewed family independently for each shot."""

        if not matches:
            return []
        preferred = {
            item.preferred_template
            for item in strategies
            if item.preferred_template is not None
        }
        groups = shot_concept_groups(lesson, pedagogy)
        selections: list[TemplateMatch] = []
        for shot, concepts in zip(pedagogy.shots, groups, strict=True):
            required = set(concepts)
            eligible = [
                match
                for match in matches
                if (not match.shot_ids or shot.shot_id in match.shot_ids)
                and shot.purpose in match.capabilities.pedagogy_roles
                and (
                    shot.purpose
                    not in {"transform", "demonstrate", "connect", "compare"}
                    or shot.purpose in match.capabilities.action_recipes
                )
                and min(len(required), MAX_SEMANTIC_CONCEPTS)
                <= match.capabilities.maximum_operands
            ]
            if not eligible:
                raise ValueError(
                    f"no reviewed template capability can satisfy shot "
                    f"{shot.shot_id!r} role={shot.purpose!r} "
                    f"operands={len(required)}"
                )

            def rank(match: TemplateMatch) -> tuple[float, float, str]:
                overlap = len(required.intersection(match.concept_ids)) / max(
                    1, len(required)
                )
                preference = 0.08 if match.template_id in preferred else 0.0
                return (
                    match.score - match.novelty_penalty
                    + 0.30 * overlap + preference,
                    overlap,
                    match.template_id,
                )

            selections.append(max(eligible, key=rank))
        return selections

    def _compile_sections(
        self,
        lesson: LessonPlan,
        selections: list[TemplateMatch],
        library: TemplateLibrary,
        pedagogy: PedagogyPlan,
        target_duration: float,
    ) -> CompiledTemplateProgram:
        """Instantiate shot-owned reviewed roots and lower deterministic cleanup."""

        groups = shot_concept_groups(lesson, pedagogy)
        roots: list[VisualObjectSpec] = []
        section_parameters: list[dict[str, object]] = []
        combined_provenance: dict[str, ParameterProvenance] = {}
        combined_defaults: list[str] = []
        for index, (selection, concepts) in enumerate(
            zip(selections, groups, strict=True)
        ):
            parameters = self._validated_parameters(selection, lesson, library)
            for key, provenance in self._latest_parameter_provenance.items():
                combined_provenance[f"{pedagogy.shots[index].shot_id}.{key}"] = provenance
            combined_defaults.extend(
                f"{pedagogy.shots[index].shot_id}.{key}"
                for key in self._latest_default_usage
            )
            parameters["object_id"] = (
                f"template_{index:02d}_{self._safe_id(selection.template_id)}"
            )
            parameters = bound_semantic_operands(
                parameters,
                preferred_concept_ids=concepts or selection.concept_ids,
            )
            root = library.instantiate(selection.template_id, parameters)
            root = self._apply_layout_variant(root, selection)
            root = self._mark_connector_ownership(
                self._ground_object(root, lesson)
            )
            root.content.update({
                "template_id": selection.template_id,
                "ownership": "shot",
                "shot_id": pedagogy.shots[index].shot_id,
            })
            roots.append(root)
            section_parameters.append(parameters)

        beats: list[VisualBeat] = []
        for index, (shot, root, concept_ids) in enumerate(
            zip(pedagogy.shots, roots, groups, strict=True)
        ):
            operations: list[VisualOperation] = []
            if index > 0:
                operations.append(VisualOperation(
                    operation_id=f"cleanup_template_{index:02d}",
                    operation=OperationType.HIDE,
                    target_ids=[item.object_id for item in roots[index - 1].flatten()],
                ))
            operations.append(VisualOperation(
                operation_id=f"create_template_{index:02d}",
                operation=OperationType.CREATE,
                target_ids=[root.object_id],
                arguments={"objects": [root.model_dump(mode="json")]},
            ))
            targets = [
                item.object_id
                for item in root.flatten()
                if item.kind != "connector"
                and set(item.concept_ids).intersection(concept_ids)
            ] or [root.object_id]
            operations.append(VisualOperation(
                operation_id=f"highlight_template_{index:02d}",
                operation=OperationType.HIGHLIGHT,
                target_ids=list(dict.fromkeys(targets)),
            ))
            semantic_actions = self._template_actions(
                root,
                shot.purpose,
                shot.shot_id,
                selections[index].capabilities,
            )
            beats.append(VisualBeat(
                beat_id=f"shot_{index + 1:02d}_{shot.shot_id}",
                section_id=f"pedagogy_{pedagogy.mode.value}",
                concept_ids=concept_ids,
                teaching_intent=shot.visual_obligation,
                phrase_intent=self._bind_action_narration(
                    self._phrase_intent(
                        shot.purpose,
                        concept_ids,
                        lesson,
                        shot.narration_obligation,
                        pedagogy.mode.value,
                    ),
                    semantic_actions,
                    root,
                ),
                purpose=shot.purpose,
                estimated_duration=max(2.0, target_duration / len(pedagogy.shots)),
                operations=operations,
                semantic_actions=semantic_actions,
                attention=[AttentionCue(
                    cue="focus",
                    target_ids=list(dict.fromkeys(targets)),
                    intensity=0.8,
                )],
                camera_intent=CameraIntent(
                    operation=shot.camera_operation,
                    target_ids=[root.object_id],
                ),
                shot_plan=ShotPlan.for_purpose(shot.purpose).model_copy(
                    update={"cleanup_policy": "retain"}
                ),
            ))
        storyboard = Storyboard(
            document_id=f"compiled_multi_{self._safe_id(lesson.title)}",
            title=lesson.title,
            beats=beats,
            final_learning_summary=[
                f"{node.label}: {node.definition}"
                for node in lesson.concept_graph.nodes
                if node.importance >= 0.5
            ],
        )
        template_ids = list(dict.fromkeys(item.template_id for item in selections))
        return CompiledTemplateProgram(
            template_id=template_ids[0],
            template_ids=template_ids,
            shot_template_ids={
                shot.shot_id: selection.template_id
                for shot, selection in zip(pedagogy.shots, selections, strict=True)
            },
            parameters={"sections": section_parameters},
            parameter_provenance=combined_provenance,
            default_usage=combined_defaults,
            match_confidence=sum(item.match_confidence for item in selections)
            / len(selections),
            capability_evidence=list(dict.fromkeys(
                evidence
                for selection in selections
                for evidence in selection.capability_evidence
            )),
            pedagogy_mode=pedagogy.mode,
            storyboard=storyboard,
            layout_variants={
                selection.template_id: selection.layout_variant
                for selection in selections
            },
        )

    @staticmethod
    def _apply_layout_variant(
        root: VisualObjectSpec,
        selection: TemplateMatch,
    ) -> VisualObjectSpec:
        """Apply only an explicitly reviewed layout alternative."""

        if selection.layout_variant == "canonical":
            return root
        expected = REVIEWED_LAYOUT_VARIANTS.get(selection.template_id)
        if selection.layout_variant != expected:
            raise ValueError(
                f"unreviewed layout variant {selection.layout_variant!r} for "
                f"template {selection.template_id!r}"
            )
        root.content["layout"] = selection.layout_variant
        root.content["layout_variant"] = selection.layout_variant
        return root

    def _select(
        self,
        lesson: LessonPlan,
        strategies: list[VisualStrategy],
        matches: list[TemplateMatch],
    ) -> TemplateMatch | None:
        """Rank reviewed families by score, coverage, and strategy evidence."""

        if not matches:
            return None
        preferred = {
            item.preferred_template
            for item in strategies
            if item.preferred_template is not None
        }
        concept_count = len(lesson.concept_graph.nodes)
        grouped: dict[str, list[TemplateMatch]] = {}
        for match in matches:
            grouped.setdefault(match.template_id, []).append(match)

        def rank(template_id: str) -> tuple[float, str]:
            candidates = grouped[template_id]
            coverage = len({
                concept_id
                for candidate in candidates
                for concept_id in candidate.concept_ids
            }) / concept_count
            score = max(item.score - item.novelty_penalty for item in candidates)
            score += 0.08 if template_id not in self._GENERIC_TEMPLATE_IDS else 0
            score += 0.12 if template_id in preferred else 0
            score += 0.10 * coverage
            return score, template_id

        selected_id = max(grouped, key=lambda item: (rank(item)[0], item))
        return max(
            grouped[selected_id],
            key=lambda item: (
                item.score - item.novelty_penalty,
                len(item.concept_ids),
            ),
        )

    def _validated_parameters(
        self,
        match: TemplateMatch,
        lesson: LessonPlan,
        library: TemplateLibrary,
    ) -> dict[str, object]:
        """Extract only reviewed fields and record every value's provenance."""

        self._latest_parameter_provenance = {
            key: value for key, value in match.parameter_provenance.items()
        }
        self._latest_default_usage: list[str] = []
        parameters: dict[str, object] = {
            "object_id": f"template_{self._safe_id(match.template_id)}",
            "label": lesson.title,
            "concepts": [
                {
                    "concept_id": node.concept_id,
                    "label": node.label,
                    "definition": node.definition,
                    "importance": node.importance,
                    "order": node.teaching_order,
                    "visual_affordances": node.visual_affordances,
                }
                for node in lesson.concept_graph.nodes
            ],
            "relations": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation": edge.relation.value,
                    "label": edge.label,
                }
                for edge in lesson.concept_graph.edges
            ],
        }
        self._latest_parameter_provenance.update({
            "object_id": ParameterProvenance(
                source="derived", source_field="template_id", confidence=1.0
            ),
            "label": ParameterProvenance(
                source="extracted", source_field="lesson.title", confidence=1.0
            ),
            "concepts": ParameterProvenance(
                source="extracted",
                source_field="lesson.concept_graph.nodes",
                confidence=1.0,
            ),
            "relations": ParameterProvenance(
                source="extracted",
                source_field="lesson.concept_graph.edges",
                confidence=1.0,
            ),
        })
        allowed = set(self._TEMPLATE_PARAMETER_FIELDS.get(match.template_id, set()))
        if match.template_id in self._COMPONENT_PARAMETER_TEMPLATES:
            allowed.add("components")
        schema: dict[str, object] = {}
        schema_provider = getattr(library, "parameter_schema", None)
        if callable(schema_provider):
            try:
                schema = schema_provider(match.template_id)
            except (KeyError, TypeError):
                schema = {}
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            allowed.update(properties)
            self._latest_default_usage.extend(
                key
                for key, definition in properties.items()
                if key not in match.parameters
                and isinstance(definition, dict)
                and "default" in definition
            )
        for key in allowed:
            value = match.parameters.get(key)
            if key in {"rows", "columns"} and isinstance(value, int):
                parameters[key] = max(1, min(10, value))
            elif isinstance(value, list):
                cleaned = [
                    str(item).strip()[:80]
                    for item in value[:10]
                    if str(item).strip()
                ]
                if cleaned:
                    parameters[key] = cleaned
            elif isinstance(value, str) and value.strip():
                parameters[key] = value.strip()[:160]
            if key in parameters and key not in self._latest_parameter_provenance:
                self._latest_parameter_provenance[key] = ParameterProvenance(
                    source="extracted",
                    source_field=f"template_match.parameters.{key}",
                    confidence=match.match_confidence,
                )

        labels = [node.label for node in lesson.concept_graph.nodes]
        if match.template_id == "pipeline.v1":
            parameters["dsl_version"] = "1.0"
            sequence = list(lesson.concept_graph.teaching_sequence)[:8]
            nodes = {
                node.concept_id: node
                for node in lesson.concept_graph.nodes
            }
            edges = {
                (edge.source_id, edge.target_id): edge
                for edge in lesson.concept_graph.edges
            }
            parameters["stages"] = [nodes[item].label for item in sequence]
            parameters["stage_details"] = [
                {
                    "concept_id": concept_id,
                    "detail": nodes[concept_id].definition,
                    "relation_to_next": (
                        edges[(concept_id, sequence[index + 1])].relation.value
                        if index + 1 < len(sequence)
                        and (concept_id, sequence[index + 1]) in edges
                        else None
                    ),
                    "relation_label": (
                        edges[(concept_id, sequence[index + 1])].label
                        if index + 1 < len(sequence)
                        and (concept_id, sequence[index + 1]) in edges
                        else None
                    ),
                }
                for index, concept_id in enumerate(sequence)
            ]
        elif match.template_id in {"tree.v1", "graph.v1"}:
            parameters["nodes"] = labels[:10]
        elif match.template_id in self._COMPONENT_PARAMETER_TEMPLATES:
            parameters["components"] = labels[:8]
        elif match.template_id == "array.v1":
            numbers = self._lesson_numbers(lesson)
            if len(numbers) >= 3:
                parameters["values"] = numbers[:10]
                self._latest_parameter_provenance["values"] = ParameterProvenance(
                    source="extracted",
                    source_field="lesson.numeric_literals",
                    confidence=1.0,
                )
        for key in match.capabilities.required_parameters:
            value = parameters.get(key)
            if value is None or value == [] or value == "":
                raise ValueError(
                    f"template {match.template_id!r} requires extracted "
                    f"parameter {key!r}; no generic default is permitted"
                )
        for key in parameters:
            self._latest_parameter_provenance.setdefault(
                key,
                ParameterProvenance(
                    source="derived",
                    source_field=f"compiler.{match.template_id}.{key}",
                    confidence=0.95,
                ),
            )
        return parameters

    def _ground_object(
        self,
        root: VisualObjectSpec,
        lesson: LessonPlan,
    ) -> VisualObjectSpec:
        """Attach lesson concepts to reviewed objects without altering structure."""

        concept_ids = list(lesson.concept_graph.teaching_sequence)
        known = set(concept_ids)
        by_label = {
            node.label.strip().casefold(): node.concept_id
            for node in lesson.concept_graph.nodes
        }
        grounded = root.model_copy(deep=True)
        primary = [
            item
            for item in grounded.flatten()
            if item.kind != "connector" and item.object_id != grounded.object_id
        ]
        assigned: set[str] = set()
        for item in primary:
            valid = [value for value in item.concept_ids if value in known]
            label = str(item.content.get("label", "")).strip().casefold()
            exact = by_label.get(label)
            if exact is not None:
                valid = [exact]
            item.concept_ids = list(dict.fromkeys(valid))
            assigned.update(item.concept_ids)

        available = [item for item in primary if not item.concept_ids]
        for concept_id, item in zip(
            [value for value in concept_ids if value not in assigned],
            available,
            strict=False,
        ):
            item.concept_ids = [concept_id]

        object_concepts = {
            item.object_id: item.concept_ids
            for item in grounded.flatten()
            if item.kind != "connector"
        }
        for item in grounded.flatten():
            if item.kind != "connector":
                continue
            endpoints = (
                str(item.content.get("source_id", "")),
                str(item.content.get("target_id", "")),
            )
            inferred = [
                concept_id
                for endpoint in endpoints
                for concept_id in object_concepts.get(endpoint, [])
            ]
            item.concept_ids = list(dict.fromkeys([
                *[value for value in item.concept_ids if value in known],
                *inferred,
            ]))
        grounded.concept_ids = concept_ids
        grounded.accessibility_label = f"{lesson.title} reviewed template"
        return grounded

    def _compile_storyboard(
        self,
        root: VisualObjectSpec,
        lesson: LessonPlan,
        pedagogy: PedagogyPlan,
        target_duration: float,
        selection: TemplateMatch,
    ) -> Storyboard:
        """Compile shot boundaries and allowed transitions from the pedagogy plan."""

        sequence = list(lesson.concept_graph.teaching_sequence)
        nodes = {node.concept_id: node for node in lesson.concept_graph.nodes}
        shots = pedagogy.shots
        groups = shot_concept_groups(lesson, pedagogy)

        flattened = root.flatten()
        primary = [
            child
            for child in flattened
            if child.kind != "connector" and child.object_id != root.object_id
        ]
        connectors = [child for child in flattened if child.kind == "connector"]
        reset_targets = [
            child.object_id
            for child in [*primary, *connectors]
            if child.object_id != root.object_id
        ]
        beats: list[VisualBeat] = []
        for index, shot in enumerate(shots):
            concept_ids = groups[index]
            targets = [
                child.object_id
                for child in primary
                if set(child.concept_ids).intersection(concept_ids)
            ] or [root.object_id]
            relation_targets = [
                child.object_id
                for child in connectors
                if set(child.concept_ids).intersection(concept_ids)
            ]
            if shot.purpose == "connect" and relation_targets:
                targets = list(dict.fromkeys([*targets, *relation_targets]))

            operations: list[VisualOperation] = []
            if index == 0:
                operations.append(
                    VisualOperation(
                        operation_id=f"create_{root.object_id}",
                        operation=OperationType.CREATE,
                        target_ids=[root.object_id],
                        arguments={"objects": [root.model_dump(mode="json")]},
                    )
                )
            if shot.purpose == "summarize":
                if reset_targets:
                    operations.append(
                        VisualOperation(
                            operation_id=f"show_{shot.shot_id}",
                            operation=OperationType.SHOW,
                            target_ids=reset_targets,
                        )
                    )
            else:
                if reset_targets:
                    operations.extend([
                        VisualOperation(
                            operation_id=f"show_{shot.shot_id}",
                            operation=OperationType.SHOW,
                            target_ids=reset_targets,
                        ),
                        VisualOperation(
                            operation_id=f"dim_{shot.shot_id}",
                            operation=OperationType.DIM,
                            target_ids=reset_targets,
                        ),
                    ])
                operations.append(
                    VisualOperation(
                        operation_id=f"highlight_{shot.shot_id}",
                        operation=OperationType.HIGHLIGHT,
                        target_ids=targets,
                    )
                )
            semantic_actions = self._template_actions(
                root,
                shot.purpose,
                shot.shot_id,
                selection.capabilities,
            )
            beats.append(
                VisualBeat(
                    beat_id=f"shot_{index + 1:02d}_{shot.shot_id}",
                    section_id=f"pedagogy_{pedagogy.mode.value}",
                    concept_ids=concept_ids,
                    teaching_intent=shot.visual_obligation,
                    phrase_intent=self._bind_action_narration(
                        self._phrase_intent(
                            shot.purpose,
                            concept_ids,
                            lesson,
                            shot.narration_obligation,
                            pedagogy.mode.value,
                        ),
                        semantic_actions,
                        root,
                    ),
                    purpose=shot.purpose,
                    estimated_duration=max(
                        2.0,
                        target_duration / max(1, len(shots)),
                    ),
                    operations=operations,
                    semantic_actions=semantic_actions,
                    attention=[
                        AttentionCue(
                            cue="focus",
                            target_ids=targets,
                            intensity=0.8,
                        )
                    ],
                    camera_intent=CameraIntent(
                        operation=shot.camera_operation,
                        target_ids=[root.object_id],
                    ),
                    shot_plan=ShotPlan.for_purpose(shot.purpose),
                )
            )
        return Storyboard(
            document_id=f"compiled_{self._safe_id(root.object_id)}",
            title=lesson.title,
            beats=beats,
            final_learning_summary=[
                f"{node.label}: {node.definition}"
                for node in lesson.concept_graph.nodes
                if node.importance >= 0.5
            ],
        )

    @staticmethod
    def _bind_action_narration(
        phrase: str,
        actions: list[SemanticAction],
        root: VisualObjectSpec,
    ) -> str:
        """State ordered route operands in the same order they animate."""

        route = next((action for action in actions if action.action == "route"), None)
        if route is None:
            return phrase
        labels = {
            item.object_id: str(
                item.content.get("label")
                or item.content.get("value")
                or item.accessibility_label
            ).strip()
            for item in root.flatten()
        }
        ordered_ids = [route.source_id, *route.path_ids, route.target_id]
        ordered_labels = [labels[item] for item in ordered_ids if labels.get(item)]
        if len(ordered_labels) < 2:
            return phrase
        return f"{phrase.rstrip()} Visit {', '.join(ordered_labels)} in that order."

    @staticmethod
    def _template_actions(
        root: VisualObjectSpec,
        purpose: str,
        shot_id: str,
        capabilities: object,
    ) -> list[SemanticAction]:
        """Bind a reviewed non-trace action recipe to concrete local objects."""

        if purpose not in {"transform", "demonstrate", "connect", "compare"}:
            return []
        recipes = getattr(capabilities, "action_recipes", {})
        recipe = recipes.get(purpose) if isinstance(recipes, dict) else None
        if not isinstance(recipe, str) or not recipe.strip():
            raise ValueError(
                f"template {root.object_id!r} has no typed action recipe for "
                f"{purpose!r}"
            )
        supported = set(getattr(capabilities, "semantic_actions", []))
        if recipe == "trace" or recipe not in supported:
            raise ValueError(
                f"template {root.object_id!r} action recipe {recipe!r} is "
                "unsupported or trace-only"
            )
        raw_operator = str(root.content.get("operator", "semantic_structure"))
        if raw_operator == "semantic_structure":
            raw_operator = str(root.content.get("source_operator", raw_operator))
        try:
            operator = RendererOperator(raw_operator)
        except ValueError:
            operator = RendererOperator.SEMANTIC_STRUCTURE
        connectors = [item for item in root.flatten() if item.kind == "connector"]
        objects = [
            item for item in root.flatten()
            if item.kind != "connector" and item.object_id != root.object_id
        ]
        if len(objects) < 2:
            objects = [item for item in root.flatten() if item.kind != "connector"]
        object_ids = [item.object_id for item in objects][:8]
        if len(object_ids) < 2:
            raise ValueError(
                f"template {root.object_id!r} cannot bind {recipe!r}; "
                "at least two visual operands are required"
            )
        action_id = f"{root.object_id}_{shot_id}_{recipe}"
        common = {
            "action_id": action_id,
            "action": recipe,
            "operator": operator,
            "duration_hint": 1.2,
        }
        source_id, target_id = object_ids[0], object_ids[-1]
        if connectors and recipe in {"transfer", "consume", "produce"}:
            source_id = str(connectors[0].content.get("source_id", source_id))
            target_id = str(connectors[-1].content.get("target_id", target_id))
        if recipe == "route":
            path = object_ids
            return [RouteAction(
                **common,
                operand_ids=path,
                source_id=path[0],
                target_id=path[-1],
                path_ids=path[1:-1],
            )]
        if recipe == "transfer":
            payload_id = object_ids[1] if len(object_ids) > 2 else source_id
            return [TransferAction(
                **common,
                operand_ids=list(dict.fromkeys([source_id, target_id, payload_id])),
                source_id=source_id,
                target_id=target_id,
                payload_ids=[payload_id],
            )]
        if recipe == "split":
            output_ids = object_ids[1:3]
            if len(output_ids) < 2:
                raise ValueError("split action recipes require two visible outputs")
            return [SplitAction(
                **common,
                operand_ids=[object_ids[0], *output_ids],
                source_id=object_ids[0],
                output_ids=output_ids,
            )]
        if recipe == "merge":
            input_ids = object_ids[:2]
            return [MergeAction(
                **common,
                operand_ids=list(dict.fromkeys([*input_ids, target_id])),
                input_ids=input_ids,
                target_id=target_id,
            )]
        if recipe == "group":
            members = object_ids[:6]
            return [GroupAction(
                **common,
                operand_ids=list(dict.fromkeys([root.object_id, *members])),
                member_ids=members,
                group_id=root.object_id,
            )]
        if recipe == "compare":
            return [CompareAction(
                **common,
                operand_ids=[object_ids[0], object_ids[1]],
                left_id=object_ids[0],
                right_id=object_ids[1],
            )]
        if recipe == "consume":
            return [ConsumeAction(
                **common,
                operand_ids=[target_id, source_id],
                consumer_id=target_id,
                item_ids=[source_id],
            )]
        if recipe == "produce":
            return [ProduceAction(
                **common,
                operand_ids=[source_id, target_id],
                producer_id=source_id,
                item_ids=[target_id],
            )]
        if recipe == "transform":
            return [TransformAction(
                **common,
                operand_ids=[source_id, target_id],
                source_id=source_id,
                target_id=target_id,
            )]
        raise ValueError(f"typed action recipe {recipe!r} has no compiler binding")

    @staticmethod
    def _phrase_intent(
        purpose: str,
        concept_ids: list[str],
        lesson: LessonPlan,
        obligation: str,
        pedagogy_mode: str,
    ) -> str:
        """Build concise factual fallback speech rather than prompt-like prose."""

        nodes = {node.concept_id: node for node in lesson.concept_graph.nodes}
        selected = [nodes[item] for item in concept_ids if item in nodes]
        del obligation
        labels = ", ".join(item.label for item in selected[:6])
        related = [
            edge
            for edge in lesson.concept_graph.edges
            if edge.source_id in concept_ids and edge.target_id in concept_ids
        ]
        if purpose == "connect" and related:
            relations = ". ".join(
                f"{nodes[edge.source_id].label} connects to "
                f"{nodes[edge.target_id].label}: "
                f"{(edge.label or edge.relation.value).replace('_', ' ')}"
                for edge in related[:5]
            )
            return f"The important relationships are explicit. {relations}."
        if purpose == "summarize":
            return (
                f"In summary, {lesson.summary.rstrip('.')}. "
                f"The key ideas are {labels}."
            )
        if purpose == "introduce":
            return (
                f"{lesson.summary.rstrip('.')}. The main ideas are {labels}."
            )
        if pedagogy_mode in {"concept_overview", "concept_set"}:
            return " ".join(
                f"{item.label}: {item.definition.rstrip('.')} ."
                for item in selected[:6]
            ).replace(" .", ".")
        steps = []
        transitions = ["First", "Next", "Then", "After that", "Finally"]
        for index, item in enumerate(selected[:6]):
            transition = transitions[min(index, len(transitions) - 1)]
            steps.append(
                f"{transition}, {item.label.lower()}: "
                f"{item.definition.rstrip('.')}"
            )
        return ". ".join(steps) + "."

    @classmethod
    def _mark_connector_ownership(
        cls,
        root: VisualObjectSpec,
    ) -> VisualObjectSpec:
        """Tell containers when declared connectors own relationship pixels."""

        marked = root.model_copy(deep=True)

        def visit(item: VisualObjectSpec) -> None:
            for child in item.children:
                visit(child)
            if any(child.kind == "connector" for child in item.children):
                item.content["connector_mode"] = "explicit"

        visit(marked)
        return marked

    @staticmethod
    def _lesson_numbers(lesson: LessonPlan) -> list[str]:
        source = " ".join(
            [lesson.summary]
            + [node.definition for node in lesson.concept_graph.nodes]
        )
        return list(dict.fromkeys(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", source)))

    @staticmethod
    def _safe_id(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "template"
