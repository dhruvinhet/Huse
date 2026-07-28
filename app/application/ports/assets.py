"""Port for semantic asset resolution."""

from typing import Protocol

from app.domain.assets import ResolvedAssetSet
from app.domain.storyboard import Storyboard


class SemanticAssetResolver(Protocol):
    """Resolve all storyboard asset queries with provenance."""

    def resolve(self, storyboard: Storyboard) -> ResolvedAssetSet:
        """Return resolved, content-addressed semantic assets."""
