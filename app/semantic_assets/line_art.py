"""Relation-aware deterministic compositions for unresolved semantic assets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from app.config.settings import PROJECT_ROOT
from app.domain.assets import AssetQuery
from app.semantic_assets.catalog import AssetCatalog, CatalogAsset


@dataclass(frozen=True, slots=True)
class CompositionPlan:
    """Describe selected local components and their visual relation grammar."""

    asset_ids: tuple[str, ...]
    relation: str
    license_ids: tuple[str, ...] = ()
    diagnostic_fallback: bool = False


class DeterministicLineArtGenerator:
    """Compose one to three local SVGs before using a diagnostic lightbulb."""

    SIZE = 512
    _RELATION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("inside", ("inside", "within", "contains", "part of", "in a")),
        ("flows-to", ("flows to", "sends", "transmits", "leads to", "causes", "request", "response")),
        ("transforms-into", ("transforms", "converts", "becomes", "changes into", "turns into")),
        ("compares-with", ("compare", "comparison", "versus", " vs ", "difference between")),
        ("orbits", ("orbit", "revolves", "around")),
        ("blocks", ("blocks", "prevents", "stops", "guards", "protects")),
        ("repeats", ("repeat", "recursive", "recursion", "loop", "cycle")),
    )

    def __init__(self, catalog: AssetCatalog | None = None) -> None:
        """Use the same checked-in catalog as direct retrieval."""

        self._catalog = catalog or AssetCatalog()

    def plan(self, query: AssetQuery) -> CompositionPlan:
        """Retrieve diverse local components and select a relation grammar."""

        candidates = self._catalog.rank(query, limit=6)
        selected: list[CatalogAsset] = []
        represented_categories: set[str] = set()
        for candidate in candidates:
            categories = set(candidate.categories)
            if selected and categories and categories <= represented_categories:
                continue
            selected.append(candidate)
            represented_categories.update(categories)
            if len(selected) == 3:
                break
        if not selected:
            return CompositionPlan((), "diagnostic", diagnostic_fallback=True)
        return CompositionPlan(
            tuple(item.asset_id for item in selected),
            self._relation(query, selected),
            tuple(sorted({item.license_id for item in selected})),
        )

    def generate(self, query: AssetQuery, output_path: Path) -> Path:
        """Create a stable editable SVG from retrieved local primitives."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        composition = self.plan(query)
        root = ElementTree.Element(
            "svg",
            {
                "xmlns": "http://www.w3.org/2000/svg",
                "width": str(self.SIZE),
                "height": str(self.SIZE),
                "viewBox": f"0 0 {self.SIZE} {self.SIZE}",
            },
        )
        metadata = ElementTree.SubElement(root, "metadata")
        metadata.text = (
            f"Huse offline composition; relation={composition.relation}; "
            f"components={','.join(composition.asset_ids) or 'diagnostic'}; "
            f"licenses={','.join(composition.license_ids) or 'generated-internal'}"
        )
        if composition.diagnostic_fallback:
            self._draw_diagnostic(root)
        else:
            records = {
                item.asset_id: item for item in self._catalog.records()
            }
            components = self._ordered_components(
                [records[item] for item in composition.asset_ids],
                composition.relation,
            )
            self._draw_composition(root, components, composition.relation)
        ElementTree.SubElement(
            root,
            "text",
            {
                "x": "256",
                "y": "462",
                "text-anchor": "middle",
                "font-size": "25",
                "font-family": "sans-serif",
                "fill": "#202124",
            },
        ).text = query.concept[:48]
        ElementTree.ElementTree(root).write(
            output_path,
            encoding="utf-8",
            xml_declaration=True,
        )
        return output_path

    @staticmethod
    def _ordered_components(
        components: list[CatalogAsset],
        relation: str,
    ) -> list[CatalogAsset]:
        """Use catalog roles to put containers, centers, sources, and targets correctly."""

        priorities = {
            "inside": ("container", "content"),
            "orbits": ("center", "satellite"),
            "flows-to": ("source", "target"),
            "transforms-into": ("source", "target"),
            "blocks": ("source", "barrier", "target"),
        }.get(relation, ())
        if not priorities:
            return components
        rank = {role: index for index, role in enumerate(priorities)}
        return sorted(
            components,
            key=lambda item: min(
                (rank[role] for role in item.composition_roles if role in rank),
                default=len(rank),
            ),
        )

    def _draw_composition(
        self,
        root: ElementTree.Element,
        components: list[CatalogAsset],
        relation: str,
    ) -> None:
        """Place retrieved icons and draw the selected relation explicitly."""

        if relation == "inside" and len(components) >= 2:
            placements = [(126.0, 54.0, 260.0), (206.0, 134.0, 100.0)]
        elif relation == "orbits" and len(components) >= 2:
            placements = [(176.0, 104.0, 160.0), (356.0, 84.0, 82.0)]
            ElementTree.SubElement(
                root,
                "ellipse",
                self._stroke_attrs(cx="256", cy="190", rx="192", ry="106", width="4"),
            )
        else:
            placements_by_count = {
                1: [(136.0, 62.0, 240.0)],
                2: [(62.0, 100.0, 148.0), (302.0, 100.0, 148.0)],
                3: [(34.0, 122.0, 112.0), (200.0, 122.0, 112.0), (366.0, 122.0, 112.0)],
            }
            placements = placements_by_count[len(components)]
        for component, placement in zip(components, placements, strict=False):
            self._embed(component, root, *placement)
        if len(components) >= 2 and relation not in {"inside", "orbits"}:
            self._draw_relation(root, relation, len(components))

    def _embed(
        self,
        asset: CatalogAsset,
        destination: ElementTree.Element,
        x: float,
        y: float,
        size: float,
    ) -> None:
        """Embed the complete local SVG tree under a deterministic transform."""

        source_path = asset.path if asset.path.is_absolute() else PROJECT_ROOT / asset.path
        source_root = ElementTree.parse(source_path).getroot()
        view_box = [float(item) for item in source_root.get("viewBox", "0 0 24 24").split()]
        source_width, source_height = view_box[2], view_box[3]
        scale = min(size / source_width, size / source_height)
        group_attributes = {
            "transform": (
                f"translate({x:.3f} {y:.3f}) scale({scale:.6f}) "
                f"translate({-view_box[0]:.3f} {-view_box[1]:.3f})"
            ),
            "color": "#202124",
        }
        for attribute in (
            "fill",
            "stroke",
            "stroke-width",
            "stroke-linecap",
            "stroke-linejoin",
        ):
            if source_root.get(attribute) is not None:
                group_attributes[attribute] = str(source_root.get(attribute))
        group = ElementTree.SubElement(destination, "g", group_attributes)
        for child in source_root:
            group.append(deepcopy(child))

    def _draw_relation(self, root: ElementTree.Element, relation: str, count: int) -> None:
        """Draw relation grammar between horizontally placed components."""

        centers = [136, 376] if count == 2 else [90, 256, 422]
        if relation == "compares-with":
            for left, right in zip(centers, centers[1:], strict=False):
                self._arrow(root, left + 60, right - 60, 190, bidirectional=True)
        elif relation == "blocks":
            for left, right in zip(centers, centers[1:], strict=False):
                midpoint = (left + right) / 2
                self._arrow(root, left + 60, midpoint - 15, 190)
                ElementTree.SubElement(root, "line", self._stroke_attrs(x1=str(midpoint), y1="142", x2=str(midpoint), y2="238", width="9"))
        else:
            for left, right in zip(centers, centers[1:], strict=False):
                self._arrow(root, left + 60, right - 60, 190)
        ElementTree.SubElement(
            root,
            "text",
            {
                "x": "256",
                "y": "382",
                "text-anchor": "middle",
                "font-size": "20",
                "font-family": "sans-serif",
                "fill": "#2a6ccd",
            },
        ).text = relation.replace("-", " ")

    def _arrow(
        self,
        root: ElementTree.Element,
        start: float,
        end: float,
        y: float,
        *,
        bidirectional: bool = False,
    ) -> None:
        ElementTree.SubElement(root, "line", self._stroke_attrs(x1=str(start), y1=str(y), x2=str(end), y2=str(y), width="5"))
        ElementTree.SubElement(root, "polygon", {"points": f"{end},{y} {end - 16},{y - 10} {end - 16},{y + 10}", "fill": "#202124", "stroke": "none", "stroke-width": "1"})
        if bidirectional:
            ElementTree.SubElement(root, "polygon", {"points": f"{start},{y} {start + 16},{y - 10} {start + 16},{y + 10}", "fill": "#202124", "stroke": "none", "stroke-width": "1"})

    def _draw_diagnostic(self, root: ElementTree.Element) -> None:
        """Draw the lightbulb only when the catalog supplies no components."""

        ElementTree.SubElement(root, "circle", self._stroke_attrs(cx="256", cy="178", r="102", width="10"))
        ElementTree.SubElement(root, "rect", self._stroke_attrs(x="218", y="278", width_value="76", height="54", width="10"))
        ElementTree.SubElement(root, "line", self._stroke_attrs(x1="226", y1="368", x2="286", y2="368", width="10"))
        ElementTree.SubElement(
            root,
            "text",
            {"x": "256", "y": "404", "text-anchor": "middle", "font-size": "18", "font-family": "sans-serif", "fill": "#b3261e"},
        ).text = "unresolved asset"

    @classmethod
    def _relation(cls, query: AssetQuery, assets: list[CatalogAsset]) -> str:
        """Prefer explicit language, then metadata relation hints."""

        text = f" {query.concept} {' '.join(query.required_semantics)} ".lower()
        for relation, markers in cls._RELATION_PATTERNS:
            if any(marker in text for marker in markers):
                return relation
        votes: dict[str, int] = {}
        for asset in assets:
            for relation in asset.relations:
                votes[relation] = votes.get(relation, 0) + 1
        if votes:
            return sorted(votes, key=lambda item: (-votes[item], item))[0]
        return "associates-with"

    @staticmethod
    def _stroke_attrs(
        *,
        width: str,
        width_value: str | None = None,
        **coordinates: str,
    ) -> dict[str, str]:
        attrs = {
            **coordinates,
            "fill": "none",
            "stroke": "#202124",
            "stroke-width": width,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
        }
        if width_value is not None:
            attrs["width"] = width_value
        return attrs
