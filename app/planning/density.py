"""Audience-aware visual density budgets."""

from dataclasses import dataclass

from app.domain.generation import AudienceLevel, AudienceProfile
from app.domain.storyboard import Storyboard, VisualObjectSpec


@dataclass(frozen=True, slots=True)
class DensityBudget:
    """Recommended semantic-object range for one beat."""

    minimum_objects: int
    maximum_objects: int
    maximum_simultaneous_events: int


class VisualDensityPlanner:
    """Estimate cognitive-load limits from audience and beat duration."""

    _BASE = {
        AudienceLevel.BEGINNER: DensityBudget(2, 8, 2),
        AudienceLevel.INTERMEDIATE: DensityBudget(3, 18, 4),
        AudienceLevel.ADVANCED: DensityBudget(4, 36, 6),
    }

    def budget(
        self,
        audience: AudienceProfile,
        duration: float,
    ) -> DensityBudget:
        """Scale a base budget conservatively for available explanation time."""

        if duration <= 0:
            raise ValueError("duration must be positive")
        base = self._BASE[audience.level]
        duration_factor = max(0.6, min(1.8, duration / 8.0))
        maximum = max(
            base.minimum_objects,
            round(base.maximum_objects * duration_factor),
        )
        if audience.max_visual_density is not None:
            maximum = min(maximum, audience.max_visual_density)
        return DensityBudget(
            minimum_objects=min(base.minimum_objects, maximum),
            maximum_objects=maximum,
            maximum_simultaneous_events=base.maximum_simultaneous_events,
        )

    def violations(
        self,
        storyboard: Storyboard,
        audience: AudienceProfile,
    ) -> list[str]:
        """Report beats whose create operations exceed their density budget."""

        violations: list[str] = []
        for beat in storyboard.beats:
            count = 0
            for operation in beat.operations:
                raw_objects = operation.arguments.get("objects")
                if not isinstance(raw_objects, list):
                    continue
                definitions = [
                    VisualObjectSpec.model_validate(item)
                    for item in raw_objects
                ]
                # An asset child is a visual treatment of its parent concept,
                # not another teaching object.  Counting it against the
                # cognitive-density budget would make adding a grounded SVG
                # fail lessons that previously passed with the same concepts.
                count += sum(
                    sum(
                        1
                        for item in root.flatten()
                        if item.kind != "semantic_asset"
                    )
                    for root in definitions
                )
            budget = self.budget(audience, beat.estimated_duration)
            if count > budget.maximum_objects:
                violations.append(
                    f"{beat.beat_id} creates {count} objects; "
                    f"maximum is {budget.maximum_objects}"
                )
        return violations
