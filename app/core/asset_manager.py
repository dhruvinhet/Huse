"""Resolution and preparation of files referenced by an asset plan."""

from pathlib import Path

import svgwrite
from loguru import logger

from app.config.settings import PROJECT_ROOT, settings
from app.models.assets import AssetPlan, AssetRequirement, AssetType
from app.models.resolved_assets import (
    ResolvedAsset,
    ResolvedAssetPlan,
    SceneResolvedAssets,
)


class AssetManager:
    """Resolve local assets and create supported static SVG primitives."""

    SVG_NAMES = frozenset({"arrow", "box", "circle"})
    SVG_CANVAS_SIZE = 512

    def resolve(self, plan: AssetPlan) -> ResolvedAssetPlan:
        """Resolve every requirement while preserving scenes and order."""

        resolved_scenes: list[SceneResolvedAssets] = []
        icons_resolved = 0
        svgs_generated = 0
        images_pending = 0

        for scene in plan.scenes:
            resolved_assets: list[ResolvedAsset] = []
            for asset in scene.assets:
                resolved = self._resolve_asset(asset)
                resolved_assets.append(resolved)
                icons_resolved += asset.asset_type is AssetType.ICON
                svgs_generated += asset.asset_type is AssetType.SVG
                images_pending += asset.asset_type is AssetType.IMAGE

            resolved_scenes.append(
                SceneResolvedAssets(
                    scene_number=scene.scene_number,
                    assets=resolved_assets,
                )
            )

        logger.info(
            "Asset resolution complete (icons_resolved={}, svgs_generated={}, "
            "images_pending={}).",
            icons_resolved,
            svgs_generated,
            images_pending,
        )
        return ResolvedAssetPlan(scenes=resolved_scenes)

    def _resolve_asset(self, asset: AssetRequirement) -> ResolvedAsset:
        """Resolve one requirement according to its asset type."""

        if asset.asset_type is AssetType.TEXT:
            return self._resolved(asset, "", ready=True, generated=False)
        if asset.asset_type is AssetType.ICON:
            return self._resolve_icon(asset)
        if asset.asset_type is AssetType.SVG:
            return self._resolve_svg(asset)
        return self._resolve_image(asset)

    def _resolve_icon(self, asset: AssetRequirement) -> ResolvedAsset:
        """Verify and return the path of a required local icon."""

        if not asset.local_path:
            raise FileNotFoundError(
                f"Icon asset {asset.asset_id!r} has no local path."
            )

        configured_path = Path(asset.local_path)
        working_path = self._working_path(configured_path)
        if not working_path.is_file():
            logger.error(
                "Icon asset {} was not found at {}.",
                asset.asset_id,
                working_path,
            )
            raise FileNotFoundError(
                f"Icon asset {asset.asset_id!r} was not found at "
                f"{str(working_path)!r}."
            )

        logger.debug("Resolved icon {} at {}.", asset.asset_id, working_path)
        return self._resolved(
            asset,
            configured_path.as_posix(),
            ready=True,
            generated=False,
        )

    def _resolve_svg(self, asset: AssetRequirement) -> ResolvedAsset:
        """Generate a supported static SVG primitive in the temp directory."""

        svg_name = asset.content.strip().lower()
        if svg_name not in self.SVG_NAMES:
            raise ValueError(f"Unsupported SVG asset: {asset.content!r}")

        configured_path = Path(settings.TEMP_DIR) / "svg" / f"{svg_name}.svg"
        working_path = self._working_path(configured_path)
        working_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_svg(svg_name, working_path)

        logger.debug("Generated SVG {} at {}.", asset.asset_id, working_path)
        return self._resolved(
            asset,
            configured_path.as_posix(),
            ready=True,
            generated=True,
        )

    def _resolve_image(self, asset: AssetRequirement) -> ResolvedAsset:
        """Prepare a pending path without creating an image file."""

        configured_path = (
            Path(settings.TEMP_DIR) / "images" / f"{asset.asset_id}.png"
        )
        working_path = self._working_path(configured_path)
        working_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug("Image {} remains pending at {}.", asset.asset_id, working_path)
        return self._resolved(
            asset,
            configured_path.as_posix(),
            ready=False,
            generated=False,
        )

    def _write_svg(self, svg_name: str, output_path: Path) -> None:
        """Write one supported primitive as a simple static SVG file."""

        canvas = self.SVG_CANVAS_SIZE
        drawing = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{canvas}px", f"{canvas}px"),
            profile="tiny",
        )
        drawing.viewbox(0, 0, canvas, canvas)

        if svg_name == "arrow":
            drawing.add(
                drawing.line(
                    start=(64, 256),
                    end=(400, 256),
                    stroke="black",
                    stroke_width=24,
                )
            )
            drawing.add(
                drawing.polygon(
                    points=[(400, 176), (480, 256), (400, 336)],
                    fill="black",
                )
            )
        elif svg_name == "box":
            drawing.add(
                drawing.rect(
                    insert=(64, 96),
                    size=(384, 320),
                    fill="none",
                    stroke="black",
                    stroke_width=20,
                )
            )
        else:
            drawing.add(
                drawing.circle(
                    center=(256, 256),
                    r=176,
                    fill="none",
                    stroke="black",
                    stroke_width=20,
                )
            )

        drawing.save(pretty=True)

    @staticmethod
    def _working_path(configured_path: Path) -> Path:
        """Resolve configured relative paths against the project root."""

        if configured_path.is_absolute():
            return configured_path
        return PROJECT_ROOT / configured_path

    @staticmethod
    def _resolved(
        asset: AssetRequirement,
        resolved_path: str,
        ready: bool,
        generated: bool,
    ) -> ResolvedAsset:
        """Build a resolved asset while preserving identity and type."""

        return ResolvedAsset(
            asset_id=asset.asset_id,
            asset_type=asset.asset_type,
            resolved_path=resolved_path,
            ready=ready,
            generated=generated,
        )
