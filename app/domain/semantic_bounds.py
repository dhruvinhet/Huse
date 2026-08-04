"""Deterministic bounds for semantic operands entering visual templates."""

from collections.abc import Iterable, Mapping


MAX_SEMANTIC_CONCEPTS = 12
MAX_SEMANTIC_RELATIONS = 24


def bound_semantic_operands(
    parameters: Mapping[str, object],
    preferred_concept_ids: Iterable[str] = (),
) -> dict[str, object]:
    """Return a coherent, bounded copy of model-produced graph operands.

    A lesson model can validly describe more facts than one shot can display.
    Template schemas therefore keep a deliberate upper bound, but raw model
    output must be reduced before Pydantic validation. Matched concepts get
    priority, then importance and teaching order provide deterministic
    tie-breakers. Relations are deduplicated and never retain dangling ends.
    """

    normalized = dict(parameters)
    preferred = {
        str(item).strip()
        for item in preferred_concept_ids
        if str(item).strip()
    }

    raw_concepts = normalized.get("concepts")
    selected_ids: set[str] | None = None
    if isinstance(raw_concepts, list):
        concepts: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for item in raw_concepts:
            if not isinstance(item, Mapping):
                continue
            concept_id = str(item.get("concept_id", "")).strip()
            if not concept_id or concept_id in seen_ids:
                continue
            seen_ids.add(concept_id)
            concepts.append(dict(item))

        if len(concepts) > MAX_SEMANTIC_CONCEPTS:
            ranked = sorted(
                enumerate(concepts),
                key=lambda pair: (
                    0 if str(pair[1].get("concept_id", "")) in preferred else 1,
                    -_importance(pair[1].get("importance")),
                    _order(pair[1].get("order"), pair[0]),
                    str(pair[1].get("concept_id", "")),
                ),
            )
            keep_indexes = {
                index for index, _ in ranked[:MAX_SEMANTIC_CONCEPTS]
            }
            concepts = [
                item
                for index, item in enumerate(concepts)
                if index in keep_indexes
            ]

        normalized["concepts"] = concepts
        selected_ids = {
            str(item["concept_id"])
            for item in concepts
            if str(item.get("concept_id", "")).strip()
        }

    raw_relations = normalized.get("relations")
    if isinstance(raw_relations, list):
        relations: list[dict[str, object]] = []
        seen_relations: set[tuple[str, str, str, str]] = set()
        for item in raw_relations:
            if not isinstance(item, Mapping):
                continue
            source_id = str(item.get("source_id", "")).strip()
            target_id = str(item.get("target_id", "")).strip()
            if selected_ids is not None and (
                source_id not in selected_ids or target_id not in selected_ids
            ):
                continue
            relation = str(item.get("relation", "")).strip()
            label = str(item.get("label") or "").strip()
            key = (source_id, target_id, relation, label)
            if not source_id or not target_id or key in seen_relations:
                continue
            seen_relations.add(key)
            relations.append(dict(item))

        if len(relations) > MAX_SEMANTIC_RELATIONS:
            relations = relations[:MAX_SEMANTIC_RELATIONS]
        normalized["relations"] = relations

    return normalized


def _importance(value: object) -> float:
    """Read an optional model score without allowing malformed data to sort."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _order(value: object, fallback: int) -> int:
    """Read an optional teaching order without allowing malformed data to sort."""

    return value if isinstance(value, int) else fallback
