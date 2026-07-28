"""Ensure every visual beat contains an explicit attention target."""

from app.domain.storyboard import AttentionCue, Storyboard


class AttentionPlanningEngine:
    """Add conservative focus cues when a planner omitted them."""

    def enrich(self, storyboard: Storyboard) -> Storyboard:
        """Return a copy with at least one focus cue per beat."""

        beats = []
        for beat in storyboard.beats:
            if beat.attention:
                beats.append(beat)
                continue
            targets = beat.operations[-1].target_ids
            beats.append(
                beat.model_copy(
                    update={
                        "attention": [
                            AttentionCue(
                                cue="focus",
                                target_ids=targets,
                                intensity=0.75,
                            )
                        ]
                    }
                )
            )
        return storyboard.model_copy(update={"beats": beats})
