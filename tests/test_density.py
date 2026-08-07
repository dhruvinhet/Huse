"""Regression tests for audience-aware visual density planning."""

from app.domain.generation import AudienceProfile
from app.domain.operations import OperationType, VisualOperation
from app.domain.storyboard import Storyboard, VisualBeat, VisualObjectSpec
from app.planning.density import VisualDensityPlanner


def test_connectors_do_not_consume_teaching_object_density_budget() -> None:
    """Relationship lines must not make an otherwise simple diagram fail."""

    root = VisualObjectSpec(
        object_id="diagram",
        kind="group",
        semantic_role="diagram",
        accessibility_label="Diagram",
        children=[
            VisualObjectSpec(
                object_id="concept",
                kind="component",
                semantic_role="concept",
                accessibility_label="Concept",
            ),
            *[
                VisualObjectSpec(
                    object_id=f"connector_{index}",
                    kind="connector",
                    semantic_role="relation",
                    accessibility_label=f"Relation {index}",
                )
                for index in range(18)
            ],
        ],
    )
    storyboard = Storyboard(
        document_id="density_test",
        title="Density test",
        beats=[
            VisualBeat(
                beat_id="beat_1",
                section_id="section",
                concept_ids=["concept"],
                teaching_intent="Introduce the concept.",
                phrase_intent="Show the concept.",
                estimated_duration=8,
                operations=[VisualOperation(
                    operation_id="create_diagram",
                    operation=OperationType.CREATE,
                    target_ids=[root.object_id],
                    arguments={"objects": [root.model_dump(mode="json")]},
                )],
            )
        ],
    )

    audience = AudienceProfile(learning_goal="Understand the concept")
    assert VisualDensityPlanner().violations(storyboard, audience) == []
