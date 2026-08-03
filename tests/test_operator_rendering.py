"""Tests for explicit fail-closed semantic renderer plugins."""

from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageChops

from app.domain.layout import LayoutBox
from app.domain.visual_document import ObjectState
from app.domain.visual_intent import RendererOperator
from app.rendering import SemanticFrameRenderer
from app.rendering.kinds import SemanticKindRegistry


def _draw(state: ObjectState) -> Image.Image:
    image = Image.new("RGBA", (640, 360), (255, 255, 255, 255))
    SemanticFrameRenderer()._draw_object(
        image,
        state,
        LayoutBox(x=70, y=50, width=500, height=250),
        {},
        MagicMock(),
    )
    return image


def test_every_declared_semantic_kind_has_explicit_plugin() -> None:
    """Declared JSON vocabulary cannot exceed pixel implementations."""

    renderer = SemanticFrameRenderer()

    assert all(
        renderer._operator_renderers.supports(kind)
        for kind in SemanticKindRegistry().names()
    )
    assert all(
        renderer._operator_renderers.supports_operator(operator.value)
        for operator in RendererOperator
    )


def test_unsupported_kind_fails_closed() -> None:
    """Unknown semantics never degrade into a rounded rectangle."""

    state = ObjectState(
        object_id="unsupported",
        kind="invented_visual_kind",
        content={"label": "Must fail"},
    )

    with pytest.raises(KeyError, match="unsupported semantic kind"):
        _draw(state)


def test_semantic_operators_produce_visibly_different_pixels() -> None:
    """Cycle, comparison, and binary search have different visual structures."""

    cycle = _draw(
        ObjectState(
            object_id="cycle",
            kind="nested_group",
            content={"label": "Lifecycle", "operator": "cycle"},
        )
    )
    comparison = _draw(
        ObjectState(
            object_id="comparison",
            kind="table",
            content={"label": "Tradeoffs", "operator": "comparison"},
        )
    )
    binary = _draw(
        ObjectState(
            object_id="binary",
            kind="array",
            child_ids=["c0", "c1", "c2", "c3", "c4"],
            content={
                "label": "Binary Search",
                "operator": "binary_search",
                "low": 0,
                "mid": 2,
                "high": 4,
                "active_code_line": "mid = (low + high) // 2",
            },
        )
    )

    cycle = cycle.convert("RGB")
    comparison = comparison.convert("RGB")
    binary = binary.convert("RGB")
    assert ImageChops.difference(cycle, comparison).getbbox() is not None
    assert ImageChops.difference(comparison, binary).getbbox() is not None
    assert ImageChops.difference(binary, cycle).getbbox() is not None
