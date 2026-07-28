"""Built-in high-value educational templates."""

from app.templates.builtins.core import (
    ArrayTemplate,
    GraphTemplate,
    MatrixTemplate,
    PipelineTemplate,
    ProbabilityDistributionTemplate,
    TransformerBlockTemplate,
    TreeTemplate,
    educational_templates,
    topic_templates,
)


def builtin_templates() -> list[object]:
    """Return a fresh deterministic set of built-in templates."""

    return [
        ArrayTemplate(),
        GraphTemplate(),
        MatrixTemplate(),
        PipelineTemplate(),
        ProbabilityDistributionTemplate(),
        TransformerBlockTemplate(),
        TreeTemplate(),
        *educational_templates(),
        *topic_templates(),
    ]


__all__ = [
    "ArrayTemplate",
    "GraphTemplate",
    "MatrixTemplate",
    "PipelineTemplate",
    "ProbabilityDistributionTemplate",
    "TransformerBlockTemplate",
    "TreeTemplate",
    "builtin_templates",
]
