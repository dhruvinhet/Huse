"""Safe explicit registry for pre-approved subsystem plugins."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar


PluginT = TypeVar("PluginT")


@dataclass(frozen=True, slots=True)
class PluginDescriptor(Generic[PluginT]):
    """Describe one installed plugin and its factory."""

    category: str
    name: str
    version: str
    contract_versions: frozenset[str]
    capabilities: frozenset[str]
    factory: Callable[[], PluginT]
    configuration_schema: dict[str, object] = field(default_factory=dict)


class PluginRegistry:
    """Register and resolve allowlisted plugins without dynamic imports."""

    def __init__(self) -> None:
        """Create an empty registry."""

        self._plugins: dict[tuple[str, str], PluginDescriptor[object]] = {}

    def register(self, descriptor: PluginDescriptor[object]) -> None:
        """Register a unique category/name pair."""

        category = descriptor.category.strip()
        name = descriptor.name.strip()
        if not category or not name or not descriptor.version.strip():
            raise ValueError("plugin category, name, and version are required")
        if not descriptor.contract_versions:
            raise ValueError("plugins must declare at least one contract version")
        key = (category, name)
        if key in self._plugins:
            raise ValueError(f"plugin is already registered: {category}/{name}")
        self._plugins[key] = descriptor

    def create(
        self,
        category: str,
        name: str,
        contract_version: str,
        required_capabilities: set[str] | None = None,
    ) -> object:
        """Create a compatible plugin after capability validation."""

        try:
            descriptor = self._plugins[(category, name)]
        except KeyError as exc:
            raise KeyError(f"unknown plugin: {category}/{name}") from exc
        if contract_version not in descriptor.contract_versions:
            raise ValueError(
                f"plugin {category}/{name} does not support contract "
                f"{contract_version}"
            )
        required = required_capabilities or set()
        if not required.issubset(descriptor.capabilities):
            missing = sorted(required.difference(descriptor.capabilities))
            raise ValueError(f"plugin is missing capabilities: {missing}")
        return descriptor.factory()

    def descriptors(self, category: str | None = None) -> list[PluginDescriptor[object]]:
        """Return registered descriptors in deterministic order."""

        descriptors = [
            descriptor
            for descriptor in self._plugins.values()
            if category is None or descriptor.category == category
        ]
        return sorted(descriptors, key=lambda item: (item.category, item.name))
