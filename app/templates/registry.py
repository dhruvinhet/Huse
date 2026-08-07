"""Relation-aware BM25 registry for reviewed educational templates."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import log
import re
from typing import Protocol

from app.domain.generation import AudienceLevel, AudienceProfile
from app.domain.lesson import ConceptGraph, ConceptRelation
from app.domain.storyboard import VisualObjectSpec
from app.domain.strategy import (
    ParameterProvenance,
    TemplateCapabilities,
    TemplateMatch,
)
from app.planning.graph_semantics import analyze_graph
from app.domain.pedagogy import PedagogyPlan
from app.planning.shot_graph import shot_query_graph


class SemanticTemplate(Protocol):
    """Contract implemented by one parameterized visual template."""

    template_id: str
    keywords: frozenset[str]
    capabilities: TemplateCapabilities

    def instantiate(self, parameters: dict[str, object]) -> VisualObjectSpec:
        """Build a semantic object hierarchy without coordinates."""


@dataclass(frozen=True, slots=True)
class _TemplateProfile:
    """Pre-tokenized local search metadata for one reviewed template."""

    template_id: str
    tokens: tuple[str, ...]
    relations: frozenset[ConceptRelation]
    minimum_operands: int
    diagram_kind: str
    capabilities: TemplateCapabilities


class TemplateRegistry:
    """Register, structurally rank, and instantiate approved templates."""

    _SYNONYMS = {
        "lookup": "search",
        "locate": "search",
        "finding": "search",
        "ordered": "sorted",
        "sequence": "array",
        "list": "array",
        "endpoint": "api",
        "website": "rest",
        "web": "rest",
        "call": "request",
        "response": "output",
        "difference": "compare",
        "comparison": "compare",
        "contrasts": "compare",
        "versus": "compare",
        "contrast": "compare",
        "alternative": "option",
        "hierarchy": "tree",
        "hierarchical": "tree",
        "network": "graph",
        "vertex": "node",
        "vertices": "node",
        "dependency": "depends",
        "component": "part",
        "architecture": "layer",
        "chronology": "timeline",
        "historical": "history",
        "calculate": "equation",
        "derivation": "equation",
        "program": "code",
        "runtime": "code",
        "causes": "cause",
        "transforms": "transform",
        "flows": "flow",
        "depends": "depend",
        "examples": "example",
    }
    _TEMPLATE_METADATA = {
        "binary_search.v1": (
            "ordered sorted lookup locate midpoint halve interval array target",
        ),
        "rest_api.v1": (
            "web endpoint http request response client server resource",
        ),
        "comparison.v1": (
            "contrast difference alternative option criteria tradeoff",
        ),
        "cause_effect.v1": (
            "cause consequence outcome because leads result",
        ),
        "input_output.v1": (
            "transform mapping source result before after",
        ),
        "layered_architecture.v1": (
            "part component hierarchy dependency system layer",
        ),
        "code_trace.v1": (
            "program runtime execute state variable control output",
        ),
        "equation_derivation.v1": (
            "calculate proof derive formula equality step result",
        ),
        "concept_set.v1": (
            "collection framework principle rule law type category member family",
        ),
    }
    _RELATION_METADATA = {
        ConceptRelation.CONTRASTS_WITH: {
            "comparison", "before_after", "venn", "table", "bar_chart",
        },
        ConceptRelation.CAUSES: {
            "cause_effect", "pipeline", "flowchart", "cycle", "simulation",
        },
        ConceptRelation.TRANSFORMS_TO: {
            "pipeline", "input_output", "equation_derivation", "array",
            "code_trace", "transformer",
        },
        ConceptRelation.FLOWS_TO: {
            "pipeline", "flowchart", "cycle", "timeline", "graph", "rest",
            "tcp", "authentication",
        },
        ConceptRelation.DEPENDS_ON: {
            "tree", "graph", "layered_architecture", "architecture",
            "decision_tree",
        },
        ConceptRelation.PART_OF: {
            "tree", "layered_architecture", "architecture", "system_design",
            "transformer_block",
        },
        ConceptRelation.EXAMPLE_OF: {
            "comparison", "tree", "graph", "table",
        },
    }
    _ACTIONS_BY_RELATION = {
        ConceptRelation.CONTRASTS_WITH: frozenset({"compare", "substitute"}),
        ConceptRelation.CAUSES: frozenset({"produce", "transform", "trace"}),
        ConceptRelation.TRANSFORMS_TO: frozenset({"transform", "transfer"}),
        ConceptRelation.FLOWS_TO: frozenset({"route", "transfer", "trace"}),
        ConceptRelation.DEPENDS_ON: frozenset({"route", "trace"}),
        ConceptRelation.PART_OF: frozenset({"group", "merge", "trace"}),
        ConceptRelation.EXAMPLE_OF: frozenset({"group", "compare"}),
    }
    _KIND_ACTIONS = {
        "process": {"transfer", "route", "split", "merge", "transform", "trace"},
        "flow": {"transfer", "route", "split", "merge", "trace"},
        "comparison": {"compare", "substitute", "transform", "trace"},
        "tree": {"route", "group", "split", "merge", "trace"},
        "graph": {"route", "group", "compare", "trace"},
        "array": {"split", "merge", "compare", "substitute", "trace"},
        "matrix": {"compare", "accumulate", "trace", "transform"},
    }
    _REQUIRED_PARAMETERS = {
        "array.v1": ["values"],
        "pipeline.v1": ["stages"],
        "tree.v1": ["nodes"],
        "graph.v1": ["nodes"],
        "matrix.v1": ["rows", "columns"],
    }

    def __init__(self, templates: list[SemanticTemplate] | None = None) -> None:
        """Register templates and precompute their local search profiles."""

        self._templates: dict[str, SemanticTemplate] = {}
        self._profiles: dict[str, _TemplateProfile] = {}
        for template in templates or []:
            self.register(template)

    def register(self, template: SemanticTemplate) -> None:
        """Register one uniquely named template and its search metadata."""

        if not template.template_id.strip():
            raise ValueError("template_id cannot be empty")
        if template.template_id in self._templates:
            raise ValueError(f"template already registered: {template.template_id}")
        if not template.keywords:
            raise ValueError("templates must declare matching keywords")
        self._templates[template.template_id] = template
        self._profiles[template.template_id] = self._profile(template)

    def match(
        self,
        graph: ConceptGraph,
        audience: AudienceProfile | None = None,
    ) -> list[TemplateMatch]:
        """Rank templates by relations/operands first and BM25 metadata second."""

        query_tokens = self._query_tokens(graph, audience)
        relation_set = {edge.relation for edge in graph.edges}
        structure = analyze_graph(graph)
        bm25 = self._bm25_scores(query_tokens)
        maximum_bm25 = max(bm25.values(), default=0.0)
        matches: list[TemplateMatch] = []
        for template_id, profile in self._profiles.items():
            if not self._structurally_eligible(
                template_id,
                profile,
                structure,
                relation_set,
            ):
                continue
            if not self._capability_eligible(profile.capabilities, graph):
                continue
            relation_score = self._relation_score(profile, relation_set)
            lexical_score = (
                bm25[template_id] / maximum_bm25
                if maximum_bm25 > 0
                else 0.0
            )
            if (
                template_id == "tree.v1"
                and self._binary_tree_required(graph)
            ):
                lexical_score = max(0.98, lexical_score)
            if template_id == "concept_set.v1" and structure.peer_collection:
                lexical_score = max(0.80, lexical_score)
            explicit_relation = (
                relation_score >= 1.0
                or bool(relation_set)
                and relation_set.issubset(set(profile.capabilities.relation_types))
            )
            if lexical_score <= 0 and not explicit_relation:
                continue
            operand_score = min(
                1.0,
                len(graph.nodes) / max(1, profile.minimum_operands),
            )
            audience_score = self._audience_score(profile, audience)
            score = min(
                1.0,
                0.50 * lexical_score
                + 0.30 * relation_score
                + 0.15 * operand_score
                + 0.05 * audience_score,
            )
            concept_ids = self._matched_concepts(
                graph,
                profile,
                explicit_relation,
            )
            parameters = self._parameters(graph, profile)
            missing_parameters = [
                name
                for name in profile.capabilities.required_parameters
                if name not in parameters
            ]
            if missing_parameters:
                continue
            relation_names = sorted(item.value for item in relation_set)
            parameter_provenance = {
                key: ParameterProvenance(
                    source=("derived" if key == "object_id" else "extracted"),
                    source_field=(
                        "template_id" if key == "object_id"
                        else "concept_graph.objectives" if key == "label"
                        else "concept_graph.nodes"
                    ),
                    confidence=1.0,
                )
                for key in parameters
            }
            evidence = [
                f"operands={len(graph.nodes)} within "
                f"{profile.capabilities.minimum_operands}-"
                f"{profile.capabilities.maximum_operands}",
                "relations=" + ",".join(relation_names or ["none"]),
                "actions=" + ",".join(profile.capabilities.semantic_actions),
            ]
            matches.append(
                TemplateMatch(
                    template_id=template_id,
                    concept_ids=concept_ids,
                    parameters=parameters,
                    prototype=None,
                    score=score,
                    capabilities=profile.capabilities,
                    capability_evidence=evidence,
                    match_confidence=score,
                    parameter_provenance=parameter_provenance,
                    reason=(
                        f"relation={relation_score:.2f}; "
                        f"bm25={bm25[template_id]:.3f}; "
                        f"operands={len(graph.nodes)}/{profile.minimum_operands}; "
                        f"detected_relations={relation_names or ['none']}"
                    ),
                )
            )
        ranked = sorted(
            matches,
            key=lambda item: (-item.score, item.template_id),
        )
        if not ranked:
            return []
        best = ranked[0].score
        return [
            item
            for item in ranked[:5]
            if item.score >= 0.45 and item.score >= best - 0.22
        ]

    def match_shots(
        self,
        graph: ConceptGraph,
        pedagogy: PedagogyPlan,
        concept_groups: list[list[str]],
        audience: AudienceProfile | None = None,
    ) -> list[TemplateMatch]:
        """Match capabilities against each shot's local semantic subgraph."""

        if len(concept_groups) != len(pedagogy.shots):
            raise ValueError("shot concept groups must align with pedagogy shots")
        queries = [
            shot_query_graph(graph, shot, concept_ids)
            for shot, concept_ids in zip(
                pedagogy.shots, concept_groups, strict=True
            )
        ]
        with ThreadPoolExecutor(
            max_workers=min(8, len(queries)),
            thread_name_prefix="template-shot-match",
        ) as executor:
            ranked = list(executor.map(
                lambda query: self.match(query, audience),
                queries,
            ))
        return [
            match.model_copy(update={"shot_ids": [shot.shot_id]})
            for shot, matches in zip(pedagogy.shots, ranked, strict=True)
            for match in matches
        ]

    @staticmethod
    def _structurally_eligible(
        template_id: str,
        profile: _TemplateProfile,
        structure: object,
        relations: set[ConceptRelation],
    ) -> bool:
        """Prevent lexical similarity from authorizing a false diagram shape."""

        process_family = (
            template_id in {"pipeline.v1", "input_output.v1", "flowchart.v1"}
            or profile.diagram_kind in {"process", "flowchart"}
        )
        if process_family and not structure.is_process:
            return False
        if template_id == "cause_effect.v1" and ConceptRelation.CAUSES not in relations:
            return False
        if template_id == "cycle.v1":
            return structure.is_process and structure.causal_edges >= 3
        if template_id == "concept_set.v1":
            return structure.peer_collection and not structure.is_process
        return True

    @classmethod
    def _capability_eligible(
        cls,
        capabilities: TemplateCapabilities,
        graph: ConceptGraph,
    ) -> bool:
        """Apply hard semantic obligations before any lexical ranking."""

        operand_count = len(graph.nodes)
        if not (
            capabilities.minimum_operands
            <= operand_count
            <= capabilities.maximum_operands
        ):
            return False
        supported_relations = set(capabilities.relation_types)
        supported_actions = set(capabilities.semantic_actions)
        for relation in {edge.relation for edge in graph.edges}:
            if supported_relations and relation not in supported_relations:
                return False
            required_actions = cls._ACTIONS_BY_RELATION.get(relation, frozenset())
            if required_actions and not required_actions.intersection(supported_actions):
                return False
        return True

    def instantiate(
        self,
        template_id: str,
        parameters: dict[str, object],
    ) -> VisualObjectSpec:
        """Instantiate a registered template."""

        try:
            template = self._templates[template_id]
        except KeyError as exc:
            raise KeyError(f"unknown template: {template_id}") from exc
        return template.instantiate(parameters)

    def template_ids(self) -> list[str]:
        """Return deterministic registered template IDs."""

        return sorted(self._templates)

    def capabilities(self, template_id: str) -> TemplateCapabilities:
        """Return the reviewed capabilities declared for one template."""

        try:
            return self._profiles[template_id].capabilities.model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"unknown template: {template_id}") from exc

    def parameter_schema(self, template_id: str) -> dict[str, object]:
        """Return a reviewed operator's explicit local parameter schema."""

        try:
            template = self._templates[template_id]
        except KeyError as exc:
            raise KeyError(f"unknown template: {template_id}") from exc
        schema_provider = getattr(template, "parameter_schema", None)
        if not callable(schema_provider):
            raise TypeError(
                f"template {template_id!r} does not expose an operator schema"
            )
        return schema_provider()

    def _profile(self, template: SemanticTemplate) -> _TemplateProfile:
        """Build metadata from declarative fields without model calls."""

        diagram_kind = str(
            getattr(template, "diagram_kind", "")
            or template.template_id.split(".", maxsplit=1)[0]
        )
        metadata: list[str] = [
            template.template_id,
            diagram_kind,
            str(getattr(template, "default_label", "")),
            *sorted(template.keywords),
            *self._TEMPLATE_METADATA.get(template.template_id, ()),
        ]
        components = getattr(template, "components", ())
        if isinstance(components, (tuple, list)):
            metadata.extend(str(item) for item in components)
        tokens = tuple(self._tokenize(" ".join(metadata)))
        relations = frozenset(
            relation
            for relation, indicators in self._RELATION_METADATA.items()
            if indicators.intersection(tokens)
            or any(indicator in template.template_id for indicator in indicators)
        )
        minimum_operands = 2 if relations else 1
        if any(
            term in tokens
            for term in {"cycle", "timeline", "venn", "graph", "tree"}
        ):
            minimum_operands = 3
        declared = getattr(template, "capabilities", None)
        if isinstance(declared, TemplateCapabilities):
            capabilities = declared.model_copy(update={
                "relation_types": (
                    declared.relation_types
                    or sorted(relations, key=lambda item: item.value)
                ),
                "minimum_operands": minimum_operands,
                "required_parameters": list(dict.fromkeys([
                    *declared.required_parameters,
                    *self._REQUIRED_PARAMETERS.get(template.template_id, []),
                ])),
            })
        else:
            actions = set(self._KIND_ACTIONS.get(diagram_kind, set()))
            for relation in relations:
                actions.update(self._ACTIONS_BY_RELATION.get(relation, ()))
            actions.add("trace")
            roles = [
                "introduce", "demonstrate", "transform", "connect",
                "emphasize", "summarize",
            ]
            if "compare" in actions:
                roles.append("compare")
            capabilities = TemplateCapabilities(
                relation_types=sorted(relations, key=lambda item: item.value),
                semantic_actions=sorted(actions),
                minimum_operands=minimum_operands,
                maximum_operands=12,
                pedagogy_roles=list(dict.fromkeys(roles)),
                layout_constraints=[
                    f"diagram_kind={diagram_kind}",
                    "semantic_ids_unique",
                    "connectors_reference_local_objects",
                ],
                required_parameters=self._REQUIRED_PARAMETERS.get(
                    template.template_id, []
                ),
                action_recipes=self._action_recipes(actions),
            )
        return _TemplateProfile(
            template_id=template.template_id,
            tokens=tokens,
            relations=relations,
            minimum_operands=minimum_operands,
            diagram_kind=diagram_kind,
            capabilities=capabilities,
        )

    @staticmethod
    def _action_recipes(actions: set[str]) -> dict[str, str]:
        """Declare non-trace action recipes supported by one template family."""

        def choose(*candidates: str) -> str | None:
            return next((item for item in candidates if item in actions), None)

        recipes = {
            "demonstrate": choose(
                "route", "transfer", "split", "compare", "transform", "group"
            ),
            "transform": choose(
                "transform", "transfer", "split", "merge", "produce", "consume"
            ),
            "connect": choose("route", "transfer", "group", "merge"),
            "compare": choose("compare", "substitute", "transform"),
        }
        return {purpose: action for purpose, action in recipes.items() if action}

    @staticmethod
    def _binary_tree_required(graph: ConceptGraph) -> bool:
        """Recognize an explicit binary-tree minimum visual obligation."""

        text = " ".join([
            *graph.objectives,
            *(node.label for node in graph.nodes),
            *(term for node in graph.nodes for term in node.visual_affordances),
        ]).casefold().replace("-", " ")
        return "binary tree" in text or "binary trees" in text

    def _bm25_scores(self, query_tokens: list[str]) -> dict[str, float]:
        """Score precomputed metadata with a small in-memory BM25 index."""

        profiles = list(self._profiles.values())
        if not profiles:
            return {}
        document_count = len(profiles)
        average_length = sum(len(item.tokens) for item in profiles) / document_count
        frequencies = {
            token: sum(token in profile.tokens for profile in profiles)
            for token in set(query_tokens)
        }
        scores: dict[str, float] = {}
        k1 = 1.5
        b = 0.75
        for profile in profiles:
            counts = Counter(profile.tokens)
            score = 0.0
            for token in query_tokens:
                frequency = counts[token]
                if frequency == 0:
                    continue
                document_frequency = frequencies[token]
                inverse_frequency = log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * len(profile.tokens) / max(1.0, average_length)
                )
                score += inverse_frequency * frequency * (k1 + 1) / denominator
            scores[profile.template_id] = score
        return scores

    @staticmethod
    def _relation_score(
        profile: _TemplateProfile,
        relations: set[ConceptRelation],
    ) -> float:
        """Evaluate structural compatibility before lexical relevance."""

        if not relations:
            return 0.35
        declared_relations = (
            profile.relations
            or frozenset(profile.capabilities.relation_types)
        )
        overlap = declared_relations.intersection(relations)
        if overlap:
            graph_recall = len(overlap) / len(relations)
            return 0.55 + 0.45 * graph_recall
        if not declared_relations:
            return 0.20
        return 0.0

    @staticmethod
    def _audience_score(
        profile: _TemplateProfile,
        audience: AudienceProfile | None,
    ) -> float:
        """Prefer readable structures for beginners and dense ones for experts."""

        if audience is None:
            return 0.5
        dense = profile.diagram_kind in {
            "graph", "matrix", "architecture", "layered_architecture",
        }
        if audience.level is AudienceLevel.BEGINNER:
            return 0.25 if dense else 1.0
        if audience.level is AudienceLevel.ADVANCED:
            return 1.0 if dense else 0.65
        return 0.75

    def _matched_concepts(
        self,
        graph: ConceptGraph,
        profile: _TemplateProfile,
        relation_compatible: bool,
    ) -> list[str]:
        """Ground matches in lexical nodes and structurally required operands."""

        metadata = set(profile.tokens)
        matched = [
            node.concept_id
            for node in graph.nodes
            if metadata.intersection(
                self._tokenize(
                    " ".join(
                        [node.label, node.definition, *node.visual_affordances]
                    )
                )
            )
        ]
        if relation_compatible:
            related = [
                concept_id
                for edge in graph.edges
                if edge.relation in profile.relations
                for concept_id in (edge.source_id, edge.target_id)
            ]
            matched.extend(related)
        ordered = [
            concept_id
            for concept_id in graph.teaching_sequence
            if concept_id in set(matched)
        ]
        return ordered or list(graph.teaching_sequence)

    @staticmethod
    def _parameters(
        graph: ConceptGraph,
        profile: _TemplateProfile,
    ) -> dict[str, object]:
        """Derive factual template operands from the validated lesson graph."""

        labels = [node.label for node in graph.nodes]
        object_id = re.sub(
            r"[^a-z0-9_]+",
            "_",
            profile.template_id.casefold(),
        ).strip("_")
        parameters: dict[str, object] = {
            "object_id": f"matched_{object_id}",
            "label": graph.objectives[0],
            "components": labels[:10],
            "stages": labels[:10],
            "nodes": labels[:10],
        }
        numbers = list(dict.fromkeys(re.findall(
            r"(?<![A-Za-z])\d+(?:\.\d+)?",
            " ".join([*labels, *[node.definition for node in graph.nodes]]),
        )))
        if numbers:
            parameters["values"] = numbers[:10]
        if profile.template_id == "matrix.v1":
            side = max(1, min(10, round(len(graph.nodes) ** 0.5)))
            parameters["rows"] = side
            parameters["columns"] = max(
                1, min(10, (len(graph.nodes) + side - 1) // side)
            )
        return parameters

    def _query_tokens(
        self,
        graph: ConceptGraph,
        audience: AudienceProfile | None,
    ) -> list[str]:
        """Include goals, definitions, affordances, and audience knowledge."""

        source = [
            *graph.objectives,
            *[
                f"{edge.relation.value} {edge.label or ''}"
                for edge in graph.edges
            ],
            *[
                " ".join(
                    [node.label, node.definition, *node.visual_affordances]
                )
                for node in graph.nodes
            ],
        ]
        if audience is not None:
            source.extend(
                [audience.learning_goal, *audience.assumed_knowledge]
            )
        return self._tokenize(" ".join(source))

    def _tokenize(self, value: str) -> list[str]:
        """Normalize tokens and a conservative local synonym vocabulary."""

        tokens = re.findall(r"[a-z0-9]+", value.casefold())
        return [self._SYNONYMS.get(token, token) for token in tokens]
