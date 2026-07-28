"""Conversion of scripts and resolved assets into renderable scene graphs."""

from math import ceil
from pathlib import Path
from xml.etree import ElementTree

from loguru import logger

from app.config.settings import PROJECT_ROOT
from app.models.assets import AssetType
from app.models.render import RenderableObject, RenderScene
from app.models.resolved_assets import ResolvedAsset, ResolvedAssetPlan
from app.models.scene import VisualInstruction
from app.models.script import Script


class SceneGraphBuilder:
    """Build fully positioned and timed render scenes without rendering them."""

    CANVAS_WIDTH = 1920
    CANVAS_HEIGHT = 1080
    TITLE_Y = 160
    ICON_SIZE = 128
    SVG_SIZE = 256
    IMAGE_SIZE = 512
    IMAGE_FALLBACK_SIZE = (520, 280)
    TEXT_CHARACTER_WIDTH = 32
    TEXT_HORIZONTAL_PADDING = 64
    BODY_LINE_HEIGHT = 48
    TITLE_LINE_HEIGHT = 64
    MIN_TEXT_WIDTH = 160
    MAX_TEXT_WIDTH = 1200

    ANIMATION_DEFAULTS = {
        AssetType.TEXT: "write",
        AssetType.ICON: "fade",
        AssetType.SVG: "draw",
        AssetType.IMAGE: "fade",
    }

    def build(
        self,
        script: Script,
        assets: ResolvedAssetPlan,
    ) -> list[RenderScene]:
        """Convert script visuals and resolved assets into render scenes."""

        assets_by_scene = {
            scene.scene_number: scene
            for scene in assets.scenes
        }
        render_scenes: list[RenderScene] = []
        object_count = 0

        for scene in script.scenes:
            resolved_scene = assets_by_scene.get(scene.scene_number)
            if resolved_scene is None:
                raise ValueError(
                    f"Resolved assets are missing for scene {scene.scene_number}."
                )
            if len(scene.visuals) != len(resolved_scene.assets):
                raise ValueError(
                    "Visual and resolved asset counts differ for scene "
                    f"{scene.scene_number}."
                )

            center_positions = self._automatic_center_positions(scene.visuals)
            objects = [
                self._build_object(
                    visual=visual,
                    asset=asset,
                    scene_duration=scene.estimated_duration,
                    automatic_position=center_positions.get(visual_index),
                )
                for visual_index, (visual, asset) in enumerate(
                    zip(
                    scene.visuals,
                    resolved_scene.assets,
                    strict=True,
                    )
                )
            ]
            object_count += len(objects)
            render_scenes.append(
                RenderScene(
                    scene_number=scene.scene_number,
                    objects=objects,
                )
            )

        logger.info(
            "Scene graph built (scenes_processed={}, objects_generated={}).",
            len(render_scenes),
            object_count,
        )
        return render_scenes

    def _build_object(
        self,
        visual: VisualInstruction,
        asset: ResolvedAsset,
        scene_duration: float,
        automatic_position: tuple[int, int] | None = None,
    ) -> RenderableObject:
        """Build one renderable object from matching visual and asset data."""

        is_title = self._is_title(visual)
        x, y = self._position(
            visual,
            asset.asset_type,
            is_title,
            automatic_position,
        )
        width, height = self._size(visual, asset, is_title)
        if asset.asset_type is AssetType.TEXT:
            content = visual.content
        elif asset.asset_type is AssetType.IMAGE and not asset.ready:
            content = f"description:{visual.content}"
        else:
            content = asset.resolved_path

        return RenderableObject(
            object_id=asset.asset_id,
            type=asset.asset_type.value,
            content=content,
            x=x,
            y=y,
            width=width,
            height=height,
            animation=self.ANIMATION_DEFAULTS[asset.asset_type],
            start_time=0.0,
            end_time=scene_duration,
        )

    def _position(
        self,
        visual: VisualInstruction,
        asset_type: AssetType,
        is_title: bool,
        automatic_position: tuple[int, int] | None,
    ) -> tuple[int, int]:
        """Assign deterministic canvas coordinates for a visual asset."""

        center_x = self.CANVAS_WIDTH // 2
        center_y = self.CANVAS_HEIGHT // 2

        if is_title:
            return center_x, self.TITLE_Y
        normalized_position = visual.position.strip().lower().replace("_", "-")
        if normalized_position == "center" and automatic_position is not None:
            return automatic_position
        return self._named_position(normalized_position)

    def _named_position(self, position: str) -> tuple[int, int]:
        """Map a requested position for any visual asset type."""

        center_x = self.CANVAS_WIDTH // 2
        center_y = self.CANVAS_HEIGHT // 2
        positions = {
            "left": (self.CANVAS_WIDTH // 4, center_y),
            "right": (self.CANVAS_WIDTH * 3 // 4, center_y),
            "top": (center_x, self.CANVAS_HEIGHT // 4),
            "bottom": (center_x, self.CANVAS_HEIGHT * 3 // 4),
            "top-left": (self.CANVAS_WIDTH // 4, self.CANVAS_HEIGHT // 4),
            "top-right": (
                self.CANVAS_WIDTH * 3 // 4,
                self.CANVAS_HEIGHT // 4,
            ),
            "bottom-left": (
                self.CANVAS_WIDTH // 4,
                self.CANVAS_HEIGHT * 3 // 4,
            ),
            "bottom-right": (
                self.CANVAS_WIDTH * 3 // 4,
                self.CANVAS_HEIGHT * 3 // 4,
            ),
        }
        return positions.get(position, (center_x, center_y))

    def _automatic_center_positions(
        self,
        visuals: list[VisualInstruction],
    ) -> dict[int, tuple[int, int]]:
        """Spread center-positioned scene content across non-overlapping slots."""

        visual_indexes = [
            index
            for index, visual in enumerate(visuals)
            if not self._is_title(visual)
            and visual.position.strip().lower().replace("_", "-") == "center"
        ]
        count = len(visual_indexes)
        if count == 0:
            return {}

        layouts = {
            1: [(960, 540)],
            2: [(520, 540), (1400, 540)],
            3: [(520, 360), (1400, 360), (960, 780)],
            4: [(520, 360), (1400, 360), (520, 780), (1400, 780)],
        }
        positions = layouts.get(count)
        if positions is None:
            columns = (400, 960, 1520)
            rows = (320, 600, 840)
            positions = [
                (columns[index % len(columns)], rows[index // len(columns)])
                for index in range(min(count, len(columns) * len(rows)))
            ]
            if count > len(positions):
                positions.extend([(960, 840)] * (count - len(positions)))
        return dict(zip(visual_indexes, positions, strict=True))

    def _size(
        self,
        visual: VisualInstruction,
        asset: ResolvedAsset,
        is_title: bool,
    ) -> tuple[int, int]:
        """Assign fixed media sizes or estimate text dimensions."""

        if asset.asset_type is AssetType.TEXT:
            return self._estimate_text_size(visual.content, is_title)
        if asset.asset_type is AssetType.ICON:
            return self.ICON_SIZE, self.ICON_SIZE
        if asset.asset_type is AssetType.SVG:
            return self._read_svg_size(asset.resolved_path)
        if not asset.ready:
            return self.IMAGE_FALLBACK_SIZE
        return self.IMAGE_SIZE, self.IMAGE_SIZE

    def _read_svg_size(self, resolved_path: str) -> tuple[int, int]:
        """Read SVG metadata for automatic sizing, with a safe fallback."""

        configured_path = Path(resolved_path)
        working_path = (
            configured_path
            if configured_path.is_absolute()
            else PROJECT_ROOT / configured_path
        )
        if not working_path.is_file():
            return self.SVG_SIZE, self.SVG_SIZE

        try:
            root = ElementTree.parse(working_path).getroot()
        except (ElementTree.ParseError, OSError):
            logger.warning(
                "Could not read SVG dimensions from {}; using defaults.",
                working_path,
            )
            return self.SVG_SIZE, self.SVG_SIZE

        width = self._parse_svg_dimension(root.get("width"))
        height = self._parse_svg_dimension(root.get("height"))
        if width is not None and height is not None:
            return width, height

        view_box = root.get("viewBox")
        if view_box:
            values = view_box.replace(",", " ").split()
            if len(values) == 4:
                try:
                    return max(1, round(float(values[2]))), max(
                        1,
                        round(float(values[3])),
                    )
                except ValueError:
                    pass

        return self.SVG_SIZE, self.SVG_SIZE

    @staticmethod
    def _parse_svg_dimension(value: str | None) -> int | None:
        """Parse a positive pixel dimension from an SVG attribute."""

        if value is None:
            return None
        normalized_value = value.strip().lower().removesuffix("px")
        try:
            parsed_value = round(float(normalized_value))
        except ValueError:
            return None
        return parsed_value if parsed_value > 0 else None

    def _estimate_text_size(
        self,
        content: str,
        is_title: bool,
    ) -> tuple[int, int]:
        """Estimate a bounded text box from character and line counts."""

        source_lines = content.splitlines() or [content]
        maximum_characters = max(len(line) for line in source_lines)
        maximum_characters_per_line = (
            self.MAX_TEXT_WIDTH // self.TEXT_CHARACTER_WIDTH
        )
        rendered_line_count = sum(
            max(1, ceil(len(line) / maximum_characters_per_line))
            for line in source_lines
        )
        width = min(
            self.MAX_TEXT_WIDTH,
            max(
                self.MIN_TEXT_WIDTH,
                maximum_characters * self.TEXT_CHARACTER_WIDTH
                + self.TEXT_HORIZONTAL_PADDING,
            ),
        )
        line_height = (
            self.TITLE_LINE_HEIGHT if is_title else self.BODY_LINE_HEIGHT
        )
        return width, rendered_line_count * line_height

    @staticmethod
    def _is_title(visual: VisualInstruction) -> bool:
        """Identify visuals explicitly designated as scene titles."""

        visual_type = visual.type.strip().lower().replace("_", "-")
        visual_position = visual.position.strip().lower().replace("_", "-")
        return visual_type == "title" or visual_position == "title"
