"""Ports for persistent state and geometric layout."""

from typing import Protocol

from app.domain.assets import ResolvedAssetSet
from app.domain.layout import LayoutPlan, Viewport
from app.domain.storyboard import Storyboard
from app.domain.visual_document import VisualDocument


class VisualStateEngine(Protocol):
    """Materialize immutable visual checkpoints from operations."""

    def materialize(self, storyboard: Storyboard) -> VisualDocument:
        """Return the persistent document and its state history."""


class ConstraintLayoutEngine(Protocol):
    """Compute renderer geometry from semantics and constraints."""

    def layout(
        self,
        document: VisualDocument,
        assets: ResolvedAssetSet,
        viewport: Viewport,
    ) -> LayoutPlan:
        """Return deterministic geometry for each state."""
