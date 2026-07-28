"""Unit tests for resolving and preparing planned visual assets."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core import asset_manager
from app.core.asset_manager import AssetManager
from app.models.assets import (
    AssetPlan,
    AssetRequirement,
    AssetType,
    SceneAssets,
)
from app.models.resolved_assets import ResolvedAssetPlan


def plan_with(asset: AssetRequirement) -> AssetPlan:
    """Create a one-scene plan containing one asset requirement."""

    return AssetPlan(scenes=[SceneAssets(scene_number=1, assets=[asset])])


def requirement(
    asset_id: str,
    asset_type: AssetType,
    content: str,
    local_path: str | None = None,
) -> AssetRequirement:
    """Create a consistent asset requirement for manager tests."""

    return AssetRequirement(
        asset_id=asset_id,
        asset_type=asset_type,
        content=content,
        exists_locally=asset_type is AssetType.ICON,
        local_path=local_path,
        needs_generation=asset_type is AssetType.IMAGE,
    )


def test_existing_icon_is_resolved(tmp_path: Path) -> None:
    """An existing icon is marked ready without being generated."""

    icon_path = tmp_path / "assets" / "icons" / "computer.svg"
    icon_path.parent.mkdir(parents=True)
    icon_path.write_text("<svg></svg>", encoding="utf-8")
    plan = plan_with(
        requirement(
            "icon-1",
            AssetType.ICON,
            "computer",
            str(icon_path),
        )
    )

    resolved_plan = AssetManager().resolve(plan)
    resolved = resolved_plan.scenes[0].assets[0]

    assert isinstance(resolved_plan, ResolvedAssetPlan)
    assert resolved.ready is True
    assert resolved.generated is False
    assert resolved.resolved_path == icon_path.as_posix()


def test_missing_icon_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """A missing local icon raises a meaningful FileNotFoundError."""

    missing_path = tmp_path / "assets" / "icons" / "missing.svg"
    plan = plan_with(
        requirement(
            "icon-missing",
            AssetType.ICON,
            "missing",
            str(missing_path),
        )
    )

    with pytest.raises(FileNotFoundError, match="icon-missing"):
        AssetManager().resolve(plan)


def test_supported_svg_assets_are_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrow, box, and circle requirements create predictable SVG files."""

    temp_dir = tmp_path / "temp"
    monkeypatch.setattr(
        asset_manager,
        "settings",
        SimpleNamespace(TEMP_DIR=temp_dir),
    )
    plan = AssetPlan(
        scenes=[
            SceneAssets(
                scene_number=1,
                assets=[
                    requirement("svg-arrow", AssetType.SVG, "arrow"),
                    requirement("svg-box", AssetType.SVG, "box"),
                    requirement("svg-circle", AssetType.SVG, "circle"),
                ],
            )
        ]
    )

    resolved_assets = AssetManager().resolve(plan).scenes[0].assets

    assert [item.generated for item in resolved_assets] == [True, True, True]
    assert [item.ready for item in resolved_assets] == [True, True, True]
    for svg_name in ("arrow", "box", "circle"):
        svg_path = temp_dir / "svg" / f"{svg_name}.svg"
        assert svg_path.is_file()
        assert "<svg" in svg_path.read_text(encoding="utf-8")


def test_image_receives_pending_placeholder_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image requirements receive a path without creating an image file."""

    temp_dir = tmp_path / "temp"
    monkeypatch.setattr(
        asset_manager,
        "settings",
        SimpleNamespace(TEMP_DIR=temp_dir),
    )
    plan = plan_with(
        requirement(
            "image-1",
            AssetType.IMAGE,
            "mountain landscape",
        )
    )

    resolved = AssetManager().resolve(plan).scenes[0].assets[0]
    expected_path = temp_dir / "images" / "image-1.png"

    assert resolved.resolved_path == expected_path.as_posix()
    assert resolved.ready is False
    assert resolved.generated is False
    assert expected_path.parent.is_dir()
    assert expected_path.exists() is False
