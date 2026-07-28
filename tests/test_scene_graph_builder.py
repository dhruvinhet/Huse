"""Unit tests for building renderer-ready scene graphs."""

from pathlib import Path

from app.core.scene_graph_builder import SceneGraphBuilder
from app.models.assets import AssetType
from app.models.render import RenderScene
from app.models.resolved_assets import (
    ResolvedAsset,
    ResolvedAssetPlan,
    SceneResolvedAssets,
)
from app.models.scene import Scene, VisualInstruction
from app.models.script import Script


def visual(
    visual_type: str,
    content: str,
    position: str = "center",
) -> VisualInstruction:
    """Create a visual instruction for scene graph tests."""

    return VisualInstruction(
        type=visual_type,
        content=content,
        position=position,
        animation="ignored-by-defaults",
    )


def resolved(
    asset_id: str,
    asset_type: AssetType,
    path: str,
    ready: bool = True,
) -> ResolvedAsset:
    """Create a resolved asset for scene graph tests."""

    return ResolvedAsset(
        asset_id=asset_id,
        asset_type=asset_type,
        resolved_path=path,
        ready=ready,
        generated=asset_type is AssetType.SVG,
    )


def build_inputs() -> tuple[Script, ResolvedAssetPlan]:
    """Create matching script and asset inputs covering every asset type."""

    scene = Scene(
        scene_number=1,
        title="Data Flow",
        narration="Data moves through a simple system.",
        estimated_duration=12.0,
        visuals=[
            visual("title", "Data Flow", "title"),
            visual("text", "Information moves between components."),
            visual("icon", "computer", "left"),
            visual("arrow", "arrow"),
            visual("illustration", "network diagram"),
        ],
    )
    script = Script(
        title="Data Flow Explained",
        topic="Data flow",
        total_duration=12,
        scenes=[scene],
    )
    assets = ResolvedAssetPlan(
        scenes=[
            SceneResolvedAssets(
                scene_number=1,
                assets=[
                    resolved("title-1", AssetType.TEXT, ""),
                    resolved("body-1", AssetType.TEXT, ""),
                    resolved(
                        "icon-1",
                        AssetType.ICON,
                        "assets/icons/computer.svg",
                    ),
                    resolved(
                        "svg-1",
                        AssetType.SVG,
                        "temp/svg/test-missing-arrow.svg",
                    ),
                    resolved(
                        "image-1",
                        AssetType.IMAGE,
                        "temp/images/image-1.png",
                        ready=False,
                    ),
                ],
            )
        ]
    )
    return script, assets


def test_correct_object_count_is_generated() -> None:
    """Every visual instruction produces exactly one renderable object."""

    script, assets = build_inputs()

    render_scenes = SceneGraphBuilder().build(script, assets)

    assert len(render_scenes) == 1
    assert isinstance(render_scenes[0], RenderScene)
    assert len(render_scenes[0].objects) == 5


def test_coordinates_are_assigned() -> None:
    """Titles, body content, icons, SVGs, and images receive coordinates."""

    script, assets = build_inputs()
    objects = SceneGraphBuilder().build(script, assets)[0].objects

    assert (objects[0].x, objects[0].y) == (960, 160)
    assert (objects[1].x, objects[1].y) == (520, 360)
    assert (objects[2].x, objects[2].y) == (480, 540)
    assert (objects[3].x, objects[3].y) == (1400, 360)
    assert (objects[4].x, objects[4].y) == (960, 780)


def test_default_animations_are_assigned() -> None:
    """Each asset category receives its specified animation default."""

    script, assets = build_inputs()
    objects = SceneGraphBuilder().build(script, assets)[0].objects

    assert [item.animation for item in objects] == [
        "write",
        "write",
        "fade",
        "draw",
        "fade",
    ]


def test_scene_duration_is_assigned_to_every_object() -> None:
    """All objects span the complete duration of their source scene."""

    script, assets = build_inputs()
    objects = SceneGraphBuilder().build(script, assets)[0].objects

    assert all(item.start_time == 0.0 for item in objects)
    assert all(item.end_time == 12.0 for item in objects)


def test_sizes_are_assigned_by_asset_type() -> None:
    """Text is estimated while media uses deterministic category sizes."""

    script, assets = build_inputs()
    objects = SceneGraphBuilder().build(script, assets)[0].objects

    assert objects[0].width >= 320
    assert objects[0].height == 64
    assert (objects[2].width, objects[2].height) == (128, 128)
    assert (objects[3].width, objects[3].height) == (256, 256)
    assert (objects[4].width, objects[4].height) == (520, 280)
    assert objects[4].content == "description:network diagram"


def test_svg_size_is_read_from_resolved_metadata(tmp_path: Path) -> None:
    """Available SVG dimensions are carried into the renderable object."""

    svg_path = tmp_path / "custom.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="320px" height="180px"></svg>',
        encoding="utf-8",
    )
    script, assets = build_inputs()
    assets.scenes[0].assets[3].resolved_path = str(svg_path)

    svg_object = SceneGraphBuilder().build(script, assets)[0].objects[3]

    assert (svg_object.width, svg_object.height) == (320, 180)
