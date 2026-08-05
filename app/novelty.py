"""Build fingerprints, measure similarity, and safely vary template selection."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from app.domain.camera import CameraPlan
from app.domain.layout import LaidOutNode, LayoutPlan
from app.domain.lesson import ConceptGraph
from app.domain.novelty import NoveltyAssessment, StructuralFingerprint
from app.domain.storyboard import Storyboard, VisualObjectSpec
from app.domain.strategy import CompiledTemplateProgram, TemplateMatch
from app.domain.visual_document import VisualDocument


REVIEWED_LAYOUT_VARIANTS = {
    "pipeline.v1": "vertical",
    "comparison.v1": "vertical",
    "concept_set.v1": "grid",
    "input_output.v1": "vertical",
}


class StructuralFingerprintBuilder:
    """Create a stable structural signature from compiled visual artifacts."""

    @classmethod
    def build(
        cls,
        storyboard: Storyboard,
        layout: LayoutPlan,
        camera: CameraPlan,
        document: VisualDocument,
        program: CompiledTemplateProgram | None = None,
    ) -> StructuralFingerprint:
        operators: list[str] = []
        actions: list[str] = []
        for beat in storyboard.beats:
            actions.extend(str(action.action) for action in beat.semantic_actions)
            for operation in beat.operations:
                for raw in operation.arguments.get("objects", []):
                    if not isinstance(raw, dict):
                        continue
                    root = VisualObjectSpec.model_validate(raw)
                    operators.extend(
                        str(item.content.get("operator", item.kind))
                        for item in root.flatten()
                        if item.object_id == root.object_id
                        or item.content.get("operator")
                    )
        topology: list[str] = []

        def visit(node: LaidOutNode, depth: int) -> None:
            topology.append(f"{depth}:{node.kind}:{len(node.children)}")
            for child in node.children:
                visit(child, depth + 1)

        for root in layout.state_roots.values():
            visit(root, 0)
        templates = [
            template_id.split(".", maxsplit=1)[0]
            for template_id in (program.template_ids if program else [])
        ]
        styles = sorted({
            item.style_token
            for state in document.states
            for item in state.object_states.values()
        })
        payload = {
            "operator_sequence": operators,
            "template_families": templates,
            "layout_topology": topology,
            "camera_operations": [cue.operation.value for cue in camera.cues],
            "action_sequence": actions,
            "style_tokens": styles,
        }
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return StructuralFingerprint(digest=digest, **payload)


def structural_similarity(
    first: StructuralFingerprint,
    second: StructuralFingerprint,
) -> float:
    """Return a weighted Jaccard score across output-structure dimensions."""

    fields = {
        "operator_sequence": 0.25,
        "template_families": 0.25,
        "layout_topology": 0.20,
        "camera_operations": 0.10,
        "action_sequence": 0.15,
        "style_tokens": 0.05,
    }
    total = 0.0
    for field, weight in fields.items():
        left = set(getattr(first, field))
        right = set(getattr(second, field))
        union = left | right
        score = len(left & right) / len(union) if union else 1.0
        total += weight * score
    return min(1.0, max(0.0, total))


class NoveltyManager:
    """Persist recent successes and request only correctness-safe variants."""

    MAX_HISTORY = 20
    MAX_SCORE_GAP = 0.08

    def __init__(self, history_path: Path, threshold: float = 0.82) -> None:
        self.history_path = history_path
        self.threshold = threshold

    def load(self) -> list[StructuralFingerprint]:
        if not self.history_path.is_file():
            return []
        try:
            raw = json.loads(self.history_path.read_text(encoding="utf-8"))
            return [StructuralFingerprint.model_validate(item) for item in raw][-self.MAX_HISTORY:]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return []

    def adjust_matches(
        self,
        matches: list[TemplateMatch],
        graph: ConceptGraph | None = None,
    ) -> list[TemplateMatch]:
        """Penalize a repeated family only when an equivalent eligible match exists."""

        history = self.load()
        if not matches or not history:
            return matches
        best = max(matches, key=lambda item: item.score)
        best_family = best.template_id.split(".", maxsplit=1)[0]
        repeated = any(
            fingerprint.template_families
            and fingerprint.template_families[0] == best_family
            for fingerprint in history[-5:]
        )
        alternatives = [
            item for item in matches
            if item.template_id.split(".", maxsplit=1)[0] != best_family
            and item.score >= best.score - self.MAX_SCORE_GAP
            and self._compatible(item, graph)
        ]
        if not repeated:
            return matches
        if not alternatives and best.template_id in REVIEWED_LAYOUT_VARIANTS:
            return [
                item.model_copy(update={
                    "layout_variant": REVIEWED_LAYOUT_VARIANTS[item.template_id],
                    "novelty_evidence": [
                        "recent_family_repeated",
                        "reviewed_layout_variant",
                    ],
                }) if item.template_id == best.template_id else item
                for item in matches
            ]
        if not alternatives:
            return matches
        return [
            item.model_copy(update={
                "novelty_penalty": 0.10,
                "novelty_evidence": [
                    "recent_family_repeated",
                    "compatible_alternative_within_0.08",
                ],
            }) if item.template_id == best.template_id else item
            for item in matches
        ]

    @staticmethod
    def _compatible(
        match: TemplateMatch,
        graph: ConceptGraph | None,
    ) -> bool:
        """Recheck hard relation/cardinality obligations before varying."""

        if graph is None:
            return True
        count = len(graph.nodes)
        relations = {edge.relation for edge in graph.edges}
        return (
            match.capabilities.minimum_operands
            <= count
            <= match.capabilities.maximum_operands
            and relations.issubset(set(match.capabilities.relation_types))
        )

    def assess(self, fingerprint: StructuralFingerprint) -> NoveltyAssessment:
        history = self.load()
        if not history:
            return NoveltyAssessment(fingerprint=fingerprint)
        scored = [
            (structural_similarity(fingerprint, item), item.digest)
            for item in history
        ]
        similarity, digest = max(scored)
        return NoveltyAssessment(
            fingerprint=fingerprint,
            maximum_recent_similarity=similarity,
            closest_recent_digest=digest,
            threshold=self.threshold,
            variant_requested=similarity >= self.threshold,
            status="similar" if similarity >= self.threshold else "novel",
        )

    def record(self, fingerprint: StructuralFingerprint) -> None:
        history = [item for item in self.load() if item.digest != fingerprint.digest]
        history.append(fingerprint)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in history[-self.MAX_HISTORY:]],
                indent=2,
            ),
            encoding="utf-8",
        )
