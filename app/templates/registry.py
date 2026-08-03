"""Relation-aware BM25 registry for reviewed educational templates."""

from collections import Counter
from dataclasses import dataclass
from math import log
import re
from typing import Protocol

from app.domain.generation import AudienceLevel, AudienceProfile
from app.domain.lesson import ConceptGraph, ConceptRelation
from app.domain.storyboard import VisualObjectSpec
from app.domain.strategy import TemplateMatch
from app.planning.graph_semantics import analyze_graph


class SemanticTemplate(Protocol):
    """Contract implemented by one parameterized visual template."""

    template_id: str
    keywords: frozenset[str]

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
            relation_score = self._relation_score(profile, relation_set)
            lexical_score = (
                bm25[template_id] / maximum_bm25
                if maximum_bm25 > 0
                else 0.0
            )
            if template_id == "concept_set.v1" and structure.peer_collection:
                lexical_score = max(0.80, lexical_score)
            explicit_relation = relation_score >= 1.0
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
            relation_names = sorted(item.value for item in relation_set)
            matches.append(
                TemplateMatch(
                    template_id=template_id,
                    concept_ids=concept_ids,
                    parameters=parameters,
                    prototype=None,
                    score=score,
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
        return _TemplateProfile(
            template_id=template.template_id,
            tokens=tokens,
            relations=relations,
            minimum_operands=minimum_operands,
            diagram_kind=diagram_kind,
        )

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
        overlap = profile.relations.intersection(relations)
        if overlap:
            graph_recall = len(overlap) / len(relations)
            return 0.55 + 0.45 * graph_recall
        if not profile.relations:
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
        return {
            "object_id": f"matched_{object_id}",
            "label": graph.objectives[0],
            "components": labels[:10],
            "stages": labels[:10],
            "nodes": labels[:10],
        }

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
