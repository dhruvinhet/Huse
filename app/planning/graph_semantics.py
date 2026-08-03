"""Shared structural interpretation for educational concept graphs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.lesson import ConceptGraph, ConceptRelation, LessonPlan


CAUSAL_RELATIONS = frozenset({
    ConceptRelation.CAUSES,
    ConceptRelation.FLOWS_TO,
    ConceptRelation.TRANSFORMS_TO,
})

_ORDINALS = frozenset({
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
})
_LABEL_STOPWORDS = frozenset({
    "a", "an", "and", "for", "in", "of", "on", "s", "the", "to",
})
_FLOW_TERMS = frozenset({
    "carry", "communicate", "connect", "deliver", "enter", "exit", "feed",
    "flow", "input", "interface", "move", "output", "pass", "propagate",
    "provide", "read", "receive", "request", "response", "return", "send",
    "signal", "store", "supply", "transfer", "transmit", "transport", "travel",
    "write", "display", "execute", "circulate", "route", "forward",
})
_TRANSFORM_TERMS = frozenset({
    "assemble", "become", "calculate", "change", "convert", "decode", "develop",
    "encode", "evolve", "map", "reshape", "produce", "transform", "turn", "update",
})
_CAUSE_TERMS = frozenset({
    "accelerate", "cause", "control", "create", "drive", "enable", "execute",
    "affect", "allow", "contribute", "determine", "form", "impact", "induce",
    "influence", "initiate", "lead", "make", "manage", "originate", "power",
    "prevent", "produce", "result", "set", "shape", "support", "trigger", "use",
})
_DEPENDENCY_TERMS = frozenset({
    "depend", "need", "monitor", "observe", "rely", "require", "watch",
})
_GENERIC_RELATION_TERMS = frozenset({
    "a", "an", "and", "because", "causes", "flow", "flows", "into", "is",
    "leads", "of", "on", "the", "to", "transforms",
})
_CONTAINMENT_REVERSE_TERMS = frozenset({
    "contain", "contains", "comprise", "comprises", "include", "includes",
    "has", "have", "holds", "house", "houses",
})
def _tokens(value: str) -> set[str]:
    raw = set(re.findall(r"[a-z0-9]+", value.casefold()))
    singulars: set[str] = set()
    for token in raw:
        if len(token) > 4 and token.endswith("ies"):
            singulars.add(token[:-3] + "y")
        elif len(token) > 3 and token.endswith("s"):
            singulars.add(token[:-1])
    return raw | singulars


@dataclass(frozen=True, slots=True)
class GraphStructure:
    """Describe graph shape independently of any named subject."""

    causal_edges: int
    causal_chain_coverage: float
    dependency_edges: int
    containment_edges: int
    contrast_edges: int
    peer_collection: bool

    @property
    def is_process(self) -> bool:
        """Return whether ordered concepts form a supported causal process."""

        return self.causal_edges > 0 and self.causal_chain_coverage >= 0.60


def analyze_graph(graph: ConceptGraph) -> GraphStructure:
    """Return deterministic topology features used across planning stages."""

    sequence = list(graph.teaching_sequence)
    causal_pairs = {
        (edge.source_id, edge.target_id)
        for edge in graph.edges
        if edge.relation in CAUSAL_RELATIONS
        and relation_label_is_compatible(edge.relation, edge.label)
    }
    consecutive = sum(
        (source, target) in causal_pairs
        for source, target in zip(sequence, sequence[1:])
    )
    denominator = max(1, len(sequence) - 1)
    causal_edges = sum(
        edge.relation in CAUSAL_RELATIONS
        and relation_label_is_compatible(edge.relation, edge.label)
        for edge in graph.edges
    )
    return GraphStructure(
        causal_edges=causal_edges,
        causal_chain_coverage=consecutive / denominator,
        dependency_edges=sum(
            edge.relation is ConceptRelation.DEPENDS_ON
            for edge in graph.edges
        ),
        containment_edges=sum(
            edge.relation is ConceptRelation.PART_OF
            for edge in graph.edges
        ),
        contrast_edges=sum(
            edge.relation is ConceptRelation.CONTRASTS_WITH
            for edge in graph.edges
        ),
        peer_collection=_looks_like_peer_collection(graph),
    )


def relation_label_is_compatible(
    relation: ConceptRelation,
    label: str | None,
) -> bool:
    """Reject vague labels that claim a different dynamic relationship."""

    if relation not in CAUSAL_RELATIONS:
        return True
    if not label or not label.strip():
        return True
    terms = _tokens(label)
    if terms & _DEPENDENCY_TERMS:
        return False
    vocabulary = {
        ConceptRelation.FLOWS_TO: _FLOW_TERMS,
        ConceptRelation.TRANSFORMS_TO: _TRANSFORM_TERMS,
        ConceptRelation.CAUSES: _CAUSE_TERMS,
    }[relation]
    if terms & vocabulary:
        return True
    other_vocabularies = [
        terms & candidate
        for candidate_relation, candidate in (
            (ConceptRelation.CAUSES, _CAUSE_TERMS),
            (ConceptRelation.FLOWS_TO, _FLOW_TERMS),
            (ConceptRelation.TRANSFORMS_TO, _TRANSFORM_TERMS),
        )
        if candidate_relation is not relation
    ]
    if any(other_vocabularies):
        return False
    # Preserve uncommon but substantive factual verbs instead of rejecting
    # every model synonym. Known cross-family terms above remain strict.
    return bool(terms - _GENERIC_RELATION_TERMS - _LABEL_STOPWORDS)


def normalize_dependency_direction(graph: ConceptGraph) -> ConceptGraph:
    """Align dependency edge direction with declared node prerequisites."""

    prerequisites = {
        (node.concept_id, prerequisite)
        for node in graph.nodes
        for prerequisite in node.prerequisites
    }
    normalized = graph.model_copy(deep=True)
    for edge in normalized.edges:
        if edge.relation is not ConceptRelation.DEPENDS_ON:
            continue
        pair = (edge.source_id, edge.target_id)
        reverse = (edge.target_id, edge.source_id)
        if pair not in prerequisites and reverse in prerequisites:
            edge.source_id, edge.target_id = edge.target_id, edge.source_id
    return normalized


def normalize_containment_direction(graph: ConceptGraph) -> ConceptGraph:
    """Align ``part_of`` edges with containment semantics and prerequisites.

    The model occasionally emits ``parent contains child`` while declaring
    the enum as ``parent part_of child``. A hierarchy compiler cannot safely
    represent that edge in the wrong direction. Labels such as ``contains``
    and prerequisite evidence provide a deterministic correction without
    inventing a relationship.
    """

    prerequisites = {
        (node.concept_id, prerequisite)
        for node in graph.nodes
        for prerequisite in node.prerequisites
    }
    normalized = graph.model_copy(deep=True)
    for edge in normalized.edges:
        if edge.relation is not ConceptRelation.PART_OF:
            continue
        label_tokens = _tokens(edge.label or "")
        reverse_pair = (edge.target_id, edge.source_id)
        label_reverses = bool(label_tokens & _CONTAINMENT_REVERSE_TERMS)
        prerequisite_reverses = reverse_pair in prerequisites
        if not (label_reverses or prerequisite_reverses):
            continue
        edge.source_id, edge.target_id = edge.target_id, edge.source_id
        if label_reverses:
            edge.label = "part of"
    return normalized


def normalize_dynamic_relation_labels(graph: ConceptGraph) -> ConceptGraph:
    """Align dynamic relations with the factual action named by each label.

    Language models sometimes select a valid relation token but pair it with a
    label from another relation family (for example ``flows_to`` + ``enables``
    or ``causes`` + ``provides``). Preserve the directed edge and use the
    label vocabulary to repair the relation before semantic validation.
    """

    normalized = graph.model_copy(deep=True)
    for edge in normalized.edges:
        if edge.relation not in CAUSAL_RELATIONS or not edge.label:
            continue
        if relation_label_is_compatible(edge.relation, edge.label):
            continue
        terms = _tokens(edge.label)
        if terms & _DEPENDENCY_TERMS:
            edge.relation = ConceptRelation.DEPENDS_ON
        elif terms & _CAUSE_TERMS:
            edge.relation = ConceptRelation.CAUSES
        elif terms & _FLOW_TERMS:
            edge.relation = ConceptRelation.FLOWS_TO
        elif terms & _TRANSFORM_TERMS:
            edge.relation = ConceptRelation.TRANSFORMS_TO
    return normalized


def _looks_like_peer_collection(graph: ConceptGraph) -> bool:
    """Detect co-equal named members without using topic-specific vocabulary."""

    if not 2 <= len(graph.nodes) <= 10:
        return False
    label_tokens = [
        _tokens(node.label) - _LABEL_STOPWORDS
        for node in graph.nodes
    ]
    shared = set.intersection(*label_tokens) if label_tokens else set()
    distinctive = [tokens - shared for tokens in label_tokens]
    ordinal_members = all(
        bool(tokens & _ORDINALS)
        or any(token.isdigit() for token in tokens)
        for tokens in distinctive
    )
    if shared and ordinal_members:
        return True

    # A model may add one supporting/example concept beside an otherwise clear
    # named collection. Detect the collection as a majority cluster instead of
    # requiring every graph node to share the same label stem.
    token_members: dict[str, list[set[str]]] = {}
    for tokens in label_tokens:
        for token in tokens - _ORDINALS:
            token_members.setdefault(token, []).append(tokens)
    clustered = any(
        len(members) >= 2
        and len(members) / len(label_tokens) >= 0.60
        and all(
            bool(member & _ORDINALS)
            or any(token.isdigit() for token in member)
            for member in members
        )
        for members in token_members.values()
    )
    if clustered:
        return True

    objective = " ".join(graph.objectives)
    objective_terms = _tokens(objective)
    explicit_count = bool(
        objective_terms & _ORDINALS
        or re.search(r"\b\d+\b", objective)
    )
    independently_named = len({node.label.casefold() for node in graph.nodes}) == len(
        graph.nodes
    )
    no_truthful_causal_chain = not any(
        edge.relation in CAUSAL_RELATIONS
        and relation_label_is_compatible(edge.relation, edge.label)
        for edge in graph.edges
    )
    return explicit_count and independently_named and no_truthful_causal_chain


def normalize_lesson_structure(lesson: LessonPlan) -> LessonPlan:
    """Repair mechanically inconsistent graph claims without inventing facts."""

    graph = normalize_dependency_direction(lesson.concept_graph)
    graph = normalize_containment_direction(graph)
    graph = normalize_dynamic_relation_labels(graph)
    structure = analyze_graph(graph)
    if structure.peer_collection and not _has_evidenced_causal_chain(graph):
        kept_edges = [
            edge
            for edge in graph.edges
            if edge.relation in {
                ConceptRelation.CONTRASTS_WITH,
                ConceptRelation.EXAMPLE_OF,
                ConceptRelation.PART_OF,
            }
        ]
        dependent_pairs = {
            edge.source_id
            for edge in kept_edges
            if edge.relation is ConceptRelation.DEPENDS_ON
        }
        nodes = [
            node.model_copy(
                update={
                    "prerequisites": (
                        node.prerequisites
                        if node.concept_id in dependent_pairs
                        else []
                    )
                }
            )
            for node in graph.nodes
        ]
        graph = graph.model_copy(update={"nodes": nodes, "edges": kept_edges})
    return lesson.model_copy(update={"concept_graph": graph})


def _has_evidenced_causal_chain(graph: ConceptGraph) -> bool:
    """Require explicit factual labels before ordering apparent peer members."""

    sequence = list(graph.teaching_sequence)
    evidenced = {
        (edge.source_id, edge.target_id)
        for edge in graph.edges
        if edge.relation in CAUSAL_RELATIONS
        and bool(edge.label and edge.label.strip())
        and relation_label_is_compatible(edge.relation, edge.label)
    }
    consecutive = sum(
        (source, target) in evidenced
        for source, target in zip(sequence, sequence[1:])
    )
    return consecutive / max(1, len(sequence) - 1) >= 0.60
