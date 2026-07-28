"""Deterministic planning of visual assets required by a script."""

from pathlib import PurePosixPath

from loguru import logger

from app.models.assets import (
    AssetPlan,
    AssetRequirement,
    AssetType,
    SceneAssets,
)
from app.models.scene import VisualInstruction
from app.models.script import Script


class AssetPlanner:
    """Classify script visuals and build an asset execution plan."""

    ICON_NAMES = frozenset(
        {
            "computer",
            "database",
            "cloud",
            "server",
            "user",
            "ai",
            "robot",
            "folder",
            "document",
            "book",
            "network",
        }
    )
    SVG_VISUAL_TYPES = frozenset({"arrow", "box", "circle"})
    TEXT_VISUAL_TYPES = frozenset({"text", "title", "body", "body-text"})

    def plan(self, script: Script) -> AssetPlan:
        """Build an asset plan for every visual in every script scene."""

        logger.info("Planning assets for {} scenes.", len(script.scenes))
        planned_scenes: list[SceneAssets] = []
        total_assets = 0
        icons_reused = 0
        images_required = 0

        for scene in script.scenes:
            scene_assets: list[AssetRequirement] = []
            for asset_index, visual in enumerate(scene.visuals, start=1):
                requirement = self._build_requirement(
                    scene_number=scene.scene_number,
                    asset_index=asset_index,
                    visual=visual,
                )
                scene_assets.append(requirement)
                total_assets += 1
                icons_reused += requirement.asset_type is AssetType.ICON
                images_required += requirement.asset_type is AssetType.IMAGE

            planned_scenes.append(
                SceneAssets(
                    scene_number=scene.scene_number,
                    assets=scene_assets,
                )
            )

        logger.info(
            "Asset plan complete (scenes={}, assets={}, icons_reused={}, "
            "images_required={}).",
            len(planned_scenes),
            total_assets,
            icons_reused,
            images_required,
        )
        return AssetPlan(scenes=planned_scenes)

    def _build_requirement(
        self,
        scene_number: int,
        asset_index: int,
        visual: VisualInstruction,
    ) -> AssetRequirement:
        """Convert one visual instruction into an asset requirement."""

        normalized_type = visual.type.strip().lower()
        normalized_content = visual.content.strip().lower()
        asset_type = self._classify(normalized_type, normalized_content)
        exists_locally = asset_type is AssetType.ICON
        needs_generation = asset_type is AssetType.IMAGE
        requirement_content = (
            normalized_type
            if asset_type is AssetType.SVG
            else visual.content
        )
        local_path = None

        if exists_locally:
            local_path = str(
                PurePosixPath(
                    "assets",
                    "icons",
                    f"{normalized_content}.svg",
                )
            )

        return AssetRequirement(
            asset_id=f"scene-{scene_number}-asset-{asset_index}",
            asset_type=asset_type,
            content=requirement_content,
            exists_locally=exists_locally,
            local_path=local_path,
            needs_generation=needs_generation,
        )

    def _classify(
        self,
        visual_type: str,
        content: str,
    ) -> AssetType:
        """Classify a normalized visual type and content value."""

        normalized_visual_type = visual_type.replace("_", "-")
        if normalized_visual_type in self.TEXT_VISUAL_TYPES:
            return AssetType.TEXT
        if normalized_visual_type in self.SVG_VISUAL_TYPES:
            return AssetType.SVG
        if content in self.ICON_NAMES:
            return AssetType.ICON
        return AssetType.IMAGE
