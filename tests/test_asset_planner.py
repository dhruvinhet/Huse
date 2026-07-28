"""Unit tests for deterministic scene asset planning."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.core.asset_planner import AssetPlanner
from app.models.assets import AssetPlan, AssetType
from app.models.scene import VisualInstruction
from app.models.script import Script


def mock_script(visuals: list[VisualInstruction]) -> Script:
    """Create a Script mock containing one scene with supplied visuals."""

    script = MagicMock(spec=Script)
    script.scenes = [SimpleNamespace(scene_number=1, visuals=visuals)]
    return script


def visual(visual_type: str, content: str) -> VisualInstruction:
    """Create a valid visual instruction for planner tests."""

    return VisualInstruction(
        type=visual_type,
        content=content,
        position="center",
        animation="draw",
    )


def test_visual_types_are_classified_correctly() -> None:
    """Text and primitive shape instructions use their specified asset types."""

    script = mock_script(
        [
            visual("text", "Hello world"),
            visual("arrow", "points right"),
            visual("box", "highlight"),
            visual("circle", "focus area"),
        ]
    )

    plan = AssetPlanner().plan(script)

    assert isinstance(plan, AssetPlan)
    assert [asset.asset_type for asset in plan.scenes[0].assets] == [
        AssetType.TEXT,
        AssetType.SVG,
        AssetType.SVG,
        AssetType.SVG,
    ]
    assert [asset.asset_id for asset in plan.scenes[0].assets] == [
        "scene-1-asset-1",
        "scene-1-asset-2",
        "scene-1-asset-3",
        "scene-1-asset-4",
    ]


def test_known_icon_is_reused_from_local_assets() -> None:
    """Known icon content resolves to its stable local SVG path."""

    script = mock_script([visual("illustration", "Computer")])

    asset = AssetPlanner().plan(script).scenes[0].assets[0]

    assert asset.asset_type is AssetType.ICON
    assert asset.exists_locally is True
    assert asset.local_path == "assets/icons/computer.svg"
    assert asset.needs_generation is False


def test_unknown_visual_requires_image_generation() -> None:
    """Unknown nonprimitive content is planned as a generated image."""

    script = mock_script([visual("illustration", "mountain landscape")])

    asset = AssetPlanner().plan(script).scenes[0].assets[0]

    assert asset.asset_type is AssetType.IMAGE
    assert asset.exists_locally is False
    assert asset.local_path is None
    assert asset.needs_generation is True
