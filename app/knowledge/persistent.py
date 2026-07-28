"""Persistent visual strategy evidence that improves through reviewed outcomes."""

from hashlib import sha256
import json
from pathlib import Path

from pydantic import Field

from app.domain.generation import AudienceProfile
from app.domain.lesson import ConceptGraph
from app.domain.strategy import VisualStrategy
from app.knowledge.repository import InMemoryVisualKnowledgeBase
from app.models.base import BaseModel, NonEmptyString


class KnowledgeRecord(BaseModel):
    """Store one reviewed strategy and its accumulated evidence."""

    record_id: NonEmptyString
    topic_terms: list[NonEmptyString] = Field(min_length=1)
    template_id: str | None = None
    teaching_strategy: NonEmptyString
    animation_hints: list[NonEmptyString] = Field(default_factory=list)
    quality_total: float = Field(ge=0)
    review_count: int = Field(ge=1)


class JsonVisualKnowledgeBase:
    """Query and update reviewed strategies in an atomic JSON store."""

    def __init__(self, path: Path) -> None:
        """Store the path and retain built-in strategy fallbacks."""

        self._path = path
        self._builtins = InMemoryVisualKnowledgeBase()

    def strategies_for(
        self,
        graph: ConceptGraph,
        audience: AudienceProfile,
    ) -> list[VisualStrategy]:
        """Merge built-in strategies with matching reviewed evidence."""

        strategies = self._builtins.strategies_for(graph, audience)
        records = self._read()
        for node in graph.nodes:
            terms = set(
                " ".join([node.label, *node.visual_affordances])
                .lower()
                .replace("-", " ")
                .split()
            )
            for record in records:
                overlap = terms.intersection(record.topic_terms)
                if not overlap:
                    continue
                strategies.append(
                    VisualStrategy(
                        strategy_id=f"knowledge_{record.record_id}_{node.concept_id}",
                        concept_ids=[node.concept_id],
                        preferred_template=record.template_id,
                        teaching_strategy=record.teaching_strategy,
                        animation_hints=record.animation_hints,
                        visual_complexity=(
                            2 if audience.level.value == "beginner"
                            else 4 if audience.level.value == "intermediate"
                            else 6
                        ),
                        audience_levels=[audience.level.value],
                        evidence_score=min(
                            1.0,
                            record.quality_total / record.review_count,
                        ),
                    )
                )
        unique = {item.strategy_id: item for item in strategies}
        return sorted(
            unique.values(),
            key=lambda item: (-item.evidence_score, item.strategy_id),
        )

    def record_outcome(
        self,
        topic_terms: list[str],
        teaching_strategy: str,
        quality_score: float,
        template_id: str | None = None,
        animation_hints: list[str] | None = None,
    ) -> KnowledgeRecord:
        """Accumulate evidence only from an explicitly reviewed outcome."""

        normalized_terms = sorted(
            {term.strip().lower() for term in topic_terms if term.strip()}
        )
        if not normalized_terms or not teaching_strategy.strip():
            raise ValueError("topic terms and teaching strategy are required")
        if not 0 <= quality_score <= 1:
            raise ValueError("quality_score must be between zero and one")
        digest_source = json.dumps(
            [normalized_terms, teaching_strategy, template_id],
            sort_keys=True,
        )
        record_id = sha256(digest_source.encode("utf-8")).hexdigest()[:16]
        records = {record.record_id: record for record in self._read()}
        existing = records.get(record_id)
        record = KnowledgeRecord(
            record_id=record_id,
            topic_terms=normalized_terms,
            template_id=template_id,
            teaching_strategy=teaching_strategy,
            animation_hints=animation_hints or [],
            quality_total=(
                quality_score
                if existing is None else existing.quality_total + quality_score
            ),
            review_count=(1 if existing is None else existing.review_count + 1),
        )
        records[record_id] = record
        self._write([records[key] for key in sorted(records)])
        return record

    def _read(self) -> list[KnowledgeRecord]:
        """Read validated records or return an empty knowledge store."""

        if not self._path.is_file():
            return []
        value = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("visual knowledge store must contain an array")
        return [KnowledgeRecord.model_validate(item) for item in value]

    def _write(self, records: list[KnowledgeRecord]) -> None:
        """Atomically persist reviewed knowledge records."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                [record.model_dump(mode="json") for record in records],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._path)
