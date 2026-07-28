"""Explicit plugin registry for replaceable V2 subsystems."""

from app.infrastructure.plugins.registry import (
    PluginDescriptor,
    PluginRegistry,
)

__all__ = ["PluginDescriptor", "PluginRegistry"]
