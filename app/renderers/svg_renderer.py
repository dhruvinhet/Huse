"""Pillow rendering for the limited generated SVG primitive set."""

import re
from pathlib import Path
from xml.etree import ElementTree

from loguru import logger
from PIL import Image, ImageDraw

from app.config.settings import PROJECT_ROOT
from app.models.render import RenderableObject


class SVGRenderer:
    """Draw line, polygon, rectangle, and circle elements from local SVGs."""

    def render(self, canvas: Image.Image, obj: RenderableObject) -> bool:
        """Render supported primitive elements from a generated SVG file."""

        configured_path = Path(obj.content)
        svg_path = (
            configured_path
            if configured_path.is_absolute()
            else PROJECT_ROOT / configured_path
        )
        if not svg_path.is_file():
            logger.warning(
                "Skipping SVG {} because {} does not exist.",
                obj.object_id,
                svg_path,
            )
            return False

        try:
            root = ElementTree.parse(svg_path).getroot()
            view_width, view_height = self._view_size(root)
            rendered_elements = self._draw_elements(
                canvas,
                obj,
                root,
                view_width,
                view_height,
            )
        except (ElementTree.ParseError, OSError, ValueError) as exc:
            logger.warning("Skipping SVG {}: {}", obj.object_id, exc)
            return False

        if rendered_elements == 0:
            logger.warning(
                "Skipping SVG {} because it has no supported primitives.",
                obj.object_id,
            )
            return False
        return True

    def _draw_elements(
        self,
        canvas: Image.Image,
        obj: RenderableObject,
        root: ElementTree.Element,
        view_width: float,
        view_height: float,
    ) -> int:
        """Scale and draw supported SVG child elements into an object box."""

        drawing = ImageDraw.Draw(canvas)
        scale_x = obj.width / view_width
        scale_y = obj.height / view_height
        left = obj.x - obj.width / 2
        top = obj.y - obj.height / 2
        rendered_elements = 0

        for element in root.iter():
            tag = element.tag.rsplit("}", maxsplit=1)[-1]
            if tag == "line":
                points = [
                    self._point(
                        element.get("x1"),
                        element.get("y1"),
                        left,
                        top,
                        scale_x,
                        scale_y,
                    ),
                    self._point(
                        element.get("x2"),
                        element.get("y2"),
                        left,
                        top,
                        scale_x,
                        scale_y,
                    ),
                ]
                drawing.line(
                    points,
                    fill=element.get("stroke", "black"),
                    width=self._stroke_width(element, scale_x, scale_y),
                )
                rendered_elements += 1
            elif tag == "polygon":
                points = self._polygon_points(
                    element.get("points", ""),
                    left,
                    top,
                    scale_x,
                    scale_y,
                )
                if points:
                    drawing.polygon(points, fill=element.get("fill", "black"))
                    rendered_elements += 1
            elif tag == "rect":
                x, y = self._point(
                    element.get("x"),
                    element.get("y"),
                    left,
                    top,
                    scale_x,
                    scale_y,
                )
                width = self._number(element.get("width")) * scale_x
                height = self._number(element.get("height")) * scale_y
                drawing.rectangle(
                    (x, y, x + width, y + height),
                    fill=self._paint(element.get("fill")),
                    outline=self._paint(element.get("stroke", "black")),
                    width=self._stroke_width(element, scale_x, scale_y),
                )
                rendered_elements += 1
            elif tag == "circle":
                center_x, center_y = self._point(
                    element.get("cx"),
                    element.get("cy"),
                    left,
                    top,
                    scale_x,
                    scale_y,
                )
                radius_x = self._number(element.get("r")) * scale_x
                radius_y = self._number(element.get("r")) * scale_y
                drawing.ellipse(
                    (
                        center_x - radius_x,
                        center_y - radius_y,
                        center_x + radius_x,
                        center_y + radius_y,
                    ),
                    fill=self._paint(element.get("fill")),
                    outline=self._paint(element.get("stroke", "black")),
                    width=self._stroke_width(element, scale_x, scale_y),
                )
                rendered_elements += 1

        return rendered_elements

    def _polygon_points(
        self,
        value: str,
        left: float,
        top: float,
        scale_x: float,
        scale_y: float,
    ) -> list[tuple[float, float]]:
        """Parse and scale an SVG polygon points attribute."""

        values = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
        return [
            (left + values[index] * scale_x, top + values[index + 1] * scale_y)
            for index in range(0, len(values) - 1, 2)
        ]

    def _point(
        self,
        x: str | None,
        y: str | None,
        left: float,
        top: float,
        scale_x: float,
        scale_y: float,
    ) -> tuple[float, float]:
        """Parse and scale one SVG coordinate pair."""

        return (
            left + self._number(x) * scale_x,
            top + self._number(y) * scale_y,
        )

    def _view_size(self, root: ElementTree.Element) -> tuple[float, float]:
        """Read positive dimensions from viewBox or width and height."""

        view_box = root.get("viewBox")
        if view_box:
            values = [float(item) for item in view_box.replace(",", " ").split()]
            if len(values) == 4 and values[2] > 0 and values[3] > 0:
                return values[2], values[3]

        width = self._number(root.get("width"))
        height = self._number(root.get("height"))
        if width <= 0 or height <= 0:
            raise ValueError("SVG dimensions must be positive")
        return width, height

    @staticmethod
    def _number(value: str | None) -> float:
        """Parse a numeric SVG value with an optional pixel suffix."""

        if value is None:
            return 0.0
        return float(value.strip().lower().removesuffix("px"))

    def _stroke_width(
        self,
        element: ElementTree.Element,
        scale_x: float,
        scale_y: float,
    ) -> int:
        """Scale an SVG stroke width to the target object box."""

        source_width = self._number(element.get("stroke-width") or "1")
        return max(1, round(source_width * (scale_x + scale_y) / 2))

    @staticmethod
    def _paint(value: str | None) -> str | None:
        """Convert the SVG none keyword to Pillow's transparent paint."""

        if value is None or value.lower() == "none":
            return None
        return value
