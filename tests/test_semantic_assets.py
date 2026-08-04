"""Regression coverage for the offline asset vocabulary and composition fallback."""

from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, ImageChops

from app.config.settings import PROJECT_ROOT
from app.domain.assets import AssetKind, AssetQuery, AssetSource
from app.domain.storyboard import VisualObjectSpec
from app.semantic_assets import (
    AssetCatalog,
    CatalogSemanticAssetResolver,
    DeterministicLineArtGenerator,
)
from app.models.render import RenderableObject
from app.renderers.svg_renderer import SVGRenderer
import app.renderers.svg_renderer as svg_renderer_module
from tests.test_domain_v2 import storyboard


def query(concept: str, *semantics: str) -> AssetQuery:
    """Build one whiteboard vector query."""

    return AssetQuery(
        concept=concept,
        asset_kind=AssetKind.LINE_ART,
        style_id="whiteboard.default",
        required_semantics=list(semantics),
    )


def semantic_object(object_id: str, concept: str) -> VisualObjectSpec:
    """Build one resolvable semantic asset object."""

    return VisualObjectSpec(
        object_id=object_id,
        kind="semantic_asset",
        semantic_role="evidence",
        content={"label": concept},
        asset_query=query(concept),
        accessibility_label=concept,
    )


def test_default_catalog_ships_two_licensed_offline_collections() -> None:
    """The production default is populated, local, and license-addressable."""

    records = AssetCatalog().records()

    assert len(records) >= 5900
    assert {item.source_collection for item in records} >= {
        "Lucide",
        "Material Symbols",
    }
    assert all(item.license_id for item in records)
    assert all((PROJECT_ROOT / item.path).is_file() for item in records)
    assert (PROJECT_ROOT / "app/semantic_assets/offline/LUCIDE_LICENSE.txt").is_file()
    assert (PROJECT_ROOT / "app/semantic_assets/offline/MATERIAL_SYMBOLS_LICENSE.txt").is_file()


def test_catalog_normalizes_aliases_and_vector_kinds() -> None:
    """Aliases and morphology retrieve a line SVG for icon-style requests."""

    match = AssetCatalog().find(
        AssetQuery(
            concept="SQL storage records",
            asset_kind=AssetKind.ICON,
            style_id="whiteboard.default",
        )
    )

    assert match is not None
    assert match.asset_id == "lucide-database"


def test_catalog_resolution_uses_storyboard_object_identity() -> None:
    """A catalog hit is addressable by the renderer's asset_<object> contract."""

    board = storyboard().model_copy(
        update={"initial_objects": [semantic_object("agent", "robot learning")]},
        deep=True,
    )

    result = CatalogSemanticAssetResolver().resolve(board)

    assert len(result.assets) == 1
    assert result.assets[0].asset_id == "asset_agent"
    assert result.assets[0].source is AssetSource.CATALOG
    assert result.assets[0].license_id == "Lucide-ISC"


def test_full_catalog_path_svg_is_renderable_and_cached_without_cairo() -> None:
    """General upstream SVG paths render through the Windows-safe resvg backend."""

    match = AssetCatalog().find(query("nephrology"))
    assert match is not None
    canvas = Image.new("RGB", (256, 256), "white")
    renderer = SVGRenderer()
    rendered = renderer.render(
        canvas,
        RenderableObject(
            object_id="nephrology",
            type="svg",
            content=match.path.as_posix(),
            x=128,
            y=128,
            width=220,
            height=220,
            animation="none",
            start_time=0,
            end_time=1,
        ),
    )

    assert rendered
    assert ImageChops.difference(canvas, Image.new("RGB", canvas.size, "white")).getbbox()
    assert len(renderer._raster_cache) == 1
    renderer.render(
        canvas,
        RenderableObject(
            object_id="nephrology",
            type="svg",
            content=match.path.as_posix(),
            x=128,
            y=128,
            width=220,
            height=220,
            animation="none",
            start_time=0,
            end_time=1,
        ),
    )
    assert len(renderer._raster_cache) == 1


def test_fallback_svg_renderer_preserves_inherited_outline_styles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Parent fill/stroke styles must not turn outline icons into black blocks."""

    source = tmp_path / "outline.svg"
    source.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'color="#202124" stroke-width="2">'
        '<path d="M4 4h16v16H4z"/></svg>',
        encoding="utf-8",
    )
    monkeypatch.setattr(svg_renderer_module, "resvg_py", None)
    canvas = Image.new("RGB", (120, 120), "white")

    assert SVGRenderer().render(
        canvas,
        RenderableObject(
            object_id="outline",
            type="svg",
            content=source.as_posix(),
            x=60,
            y=60,
            width=96,
            height=96,
            animation="none",
            start_time=0,
            end_time=1,
        ),
    )
    assert canvas.getpixel((60, 60)) == (255, 255, 255)


def test_exact_full_pack_name_beats_a_related_alias() -> None:
    """Broad vocabularies do not let weak aliases outrank an exact symbol name."""

    match = AssetCatalog().find(query("home"))

    assert match is not None
    assert match.asset_id in {"full-lucide-home", "full-material-symbols-home"}


def test_identical_queries_for_multiple_objects_are_not_deduplicated() -> None:
    """Each visual object gets the renderer identity even when content is shared."""

    board = storyboard().model_copy(
        update={
            "initial_objects": [
                semantic_object("client_a", "user"),
                semantic_object("client_b", "user"),
            ]
        },
        deep=True,
    )

    result = CatalogSemanticAssetResolver().resolve(board)

    assert {item.asset_id for item in result.assets} == {
        "asset_client_a",
        "asset_client_b",
    }
    assert len({item.content_hash for item in result.assets}) == 1


def test_unknown_domain_concepts_use_multi_icon_relation_grammar() -> None:
    """Weak metadata matches compose locally instead of changing one badge label."""

    generator = DeterministicLineArtGenerator(AssetCatalog())
    expected = {
        "mitochondria": "inside",
        "inflation": "flows-to",
        "gravity": "orbits",
        "recursion": "repeats",
    }

    for concept, relation in expected.items():
        plan = generator.plan(query(concept))
        assert len(plan.asset_ids) >= 2
        assert plan.relation == relation
        assert not plan.diagnostic_fallback


def test_explicit_relation_language_controls_generic_composition(
    tmp_path: Path,
) -> None:
    """Relation grammar comes from query semantics rather than topic branches."""

    catalog = AssetCatalog()
    generator = DeterministicLineArtGenerator(catalog)
    relation_query = query(
        "user request flows to server", "person", "backend"
    )
    plan = generator.plan(relation_query)

    assert catalog.find(relation_query) is None
    assert plan.relation == "flows-to"
    assert len(plan.asset_ids) >= 2
    output = generator.generate(relation_query, tmp_path / "request-flow.svg")
    assert "path" in output.read_text(encoding="utf-8")
    canvas = Image.new("RGB", (512, 512), "white")
    assert SVGRenderer().render(
        canvas,
        RenderableObject(
            object_id="request_flow",
            type="svg",
            content=output.as_posix(),
            x=256,
            y=256,
            width=512,
            height=512,
            animation="none",
            start_time=0,
            end_time=1,
        ),
    )


def test_lightbulb_is_only_a_labeled_diagnostic_fallback(tmp_path: Path) -> None:
    """A true vocabulary miss is explicit and cannot masquerade as illustration."""

    output = tmp_path / "diagnostic.svg"
    generator = DeterministicLineArtGenerator(AssetCatalog())
    plan = generator.plan(query("purple quasar luminosity"))
    generator.generate(query("purple quasar luminosity"), output)

    assert plan.diagnostic_fallback
    root = ElementTree.parse(output).getroot()
    texts = [
        (element.text or "").strip()
        for element in root.iter()
        if element.tag.rsplit("}", maxsplit=1)[-1] == "text"
    ]
    assert "unresolved asset" in texts
    assert "purple quasar luminosity" in texts


def test_generated_composition_is_cached_by_query_digest(tmp_path: Path) -> None:
    """Repeated resolution reuses the local SVG instead of regenerating it."""

    board = storyboard().model_copy(
        update={
            "initial_objects": [
                semantic_object("mitochondria", "mitochondria")
            ]
        },
        deep=True,
    )
    resolver = CatalogSemanticAssetResolver(generated_dir=tmp_path)
    first = resolver.resolve(board).assets[0]
    output = Path(first.path)
    before = output.stat().st_mtime_ns
    second = resolver.resolve(board).assets[0]

    assert first.source is AssetSource.GENERATED
    assert first.license_id == "composed:Material-Symbols-Apache-2.0"
    assert second.content_hash == first.content_hash
    assert output.stat().st_mtime_ns == before
