"""SVG rendering with cached resvg rasterization and primitive fallback."""

from io import BytesIO
from math import atan2, cos, degrees, pi, sin, sqrt
import re
from pathlib import Path
from xml.etree import ElementTree

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from app.config.settings import PROJECT_ROOT
from app.models.render import RenderableObject

try:
    import resvg_py
except ImportError:  # pragma: no cover - dependency diagnostics are tested indirectly
    resvg_py = None


def _sample_cubic(
    start: tuple[float, float],
    control_1: tuple[float, float],
    control_2: tuple[float, float],
    end: tuple[float, float],
    steps: int = 8,
) -> list[tuple[float, float]]:
    """Approximate a cubic SVG curve with a short deterministic polyline."""

    result: list[tuple[float, float]] = []
    for index in range(1, steps + 1):
        t = index / steps
        inverse = 1.0 - t
        result.append((
            inverse ** 3 * start[0]
            + 3 * inverse ** 2 * t * control_1[0]
            + 3 * inverse * t ** 2 * control_2[0]
            + t ** 3 * end[0],
            inverse ** 3 * start[1]
            + 3 * inverse ** 2 * t * control_1[1]
            + 3 * inverse * t ** 2 * control_2[1]
            + t ** 3 * end[1],
        ))
    return result


def _sample_quadratic(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    steps: int = 8,
) -> list[tuple[float, float]]:
    """Approximate a quadratic SVG curve with a deterministic polyline."""

    result: list[tuple[float, float]] = []
    for index in range(1, steps + 1):
        t = index / steps
        inverse = 1.0 - t
        result.append((
            inverse ** 2 * start[0]
            + 2 * inverse * t * control[0]
            + t ** 2 * end[0],
            inverse ** 2 * start[1]
            + 2 * inverse * t * control[1]
            + t ** 2 * end[1],
        ))
    return result


def _sample_arc(
    start: tuple[float, float],
    values: list[float],
    steps: int = 12,
) -> list[tuple[float, float]]:
    """Approximate one SVG elliptical arc using the endpoint parameterization."""

    rx, ry, rotation, large_arc, sweep, end_x, end_y = values
    rx, ry = abs(rx), abs(ry)
    end = (end_x, end_y)
    if rx == 0 or ry == 0 or start == end:
        return [end]
    phi = rotation * pi / 180.0
    cos_phi, sin_phi = cos(phi), sin(phi)
    dx, dy = (start[0] - end[0]) / 2, (start[1] - end[1]) / 2
    x_prime = cos_phi * dx + sin_phi * dy
    y_prime = -sin_phi * dx + cos_phi * dy
    radius_scale = (x_prime * x_prime) / (rx * rx) + (y_prime * y_prime) / (ry * ry)
    if radius_scale > 1:
        scale = sqrt(radius_scale)
        rx *= scale
        ry *= scale
    denominator = rx * rx * y_prime * y_prime + ry * ry * x_prime * x_prime
    numerator = max(0.0, rx * rx * ry * ry - denominator)
    coefficient = 0.0 if denominator == 0 else sqrt(numerator / denominator)
    if bool(large_arc) == bool(sweep):
        coefficient = -coefficient
    center_x_prime = coefficient * rx * y_prime / ry
    center_y_prime = -coefficient * ry * x_prime / rx
    center = (
        cos_phi * center_x_prime - sin_phi * center_y_prime + (start[0] + end[0]) / 2,
        sin_phi * center_x_prime + cos_phi * center_y_prime + (start[1] + end[1]) / 2,
    )

    def angle(vector_x: float, vector_y: float) -> float:
        return atan2(vector_y, vector_x)

    start_angle = angle(
        (x_prime - center_x_prime) / rx,
        (y_prime - center_y_prime) / ry,
    )
    delta = angle(
        (-x_prime - center_x_prime) / rx,
        (-y_prime - center_y_prime) / ry,
    ) - start_angle
    if not sweep and delta > 0:
        delta -= 2 * pi
    if sweep and delta < 0:
        delta += 2 * pi
    result: list[tuple[float, float]] = []
    for index in range(1, steps + 1):
        theta = start_angle + delta * index / steps
        result.append((
            center[0] + rx * cos_phi * cos(theta) - ry * sin_phi * sin(theta),
            center[1] + rx * sin_phi * cos(theta) + ry * cos_phi * sin(theta),
        ))
    return result


class SVGRenderer:
    """Draw line, polygon, rectangle, and circle elements from local SVGs."""

    def __init__(self) -> None:
        """Initialize a bounded per-renderer raster cache."""

        self._raster_cache: dict[tuple[str, int, int, int], Image.Image] = {}
        self._raster_cache_limit = 256

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

        if resvg_py is not None:
            try:
                rendered_icon = self._rasterized(
                    svg_path,
                    max(1, int(obj.width)),
                    max(1, int(obj.height)),
                )
                position = (
                    round(obj.x - obj.width / 2),
                    round(obj.y - obj.height / 2),
                )
                canvas.paste(rendered_icon, position, rendered_icon)
                return True
            except (OSError, ValueError) as exc:
                logger.warning(
                    "resvg could not rasterize SVG {} ({}); trying the "
                    "primitive renderer.",
                    obj.object_id,
                    exc,
                )

        try:
            root = ElementTree.parse(svg_path).getroot()
            view_min_x, view_min_y, view_width, view_height = self._view_bounds(root)
            target_width = max(1, int(obj.width))
            target_height = max(1, int(obj.height))
            cache_key = (str(svg_path.resolve()), svg_path.stat().st_mtime_ns, target_width, target_height)
            cached_fallback = self._raster_cache.get(cache_key)
            if cached_fallback is not None:
                canvas.paste(
                    cached_fallback,
                    (round(obj.x - obj.width / 2), round(obj.y - obj.height / 2)),
                    cached_fallback,
                )
                return True
            fallback_canvas = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
            fallback_obj = obj.model_copy(
                update={
                    "x": target_width / 2,
                    "y": target_height / 2,
                    "width": target_width,
                    "height": target_height,
                }
            )
            rendered_elements = self._draw_elements(
                fallback_canvas,
                fallback_obj,
                root,
                view_min_x,
                view_min_y,
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
        self._raster_cache[cache_key] = fallback_canvas
        if len(self._raster_cache) > self._raster_cache_limit:
            self._raster_cache.pop(next(iter(self._raster_cache)))
        canvas.paste(
            fallback_canvas,
            (round(obj.x - obj.width / 2), round(obj.y - obj.height / 2)),
            fallback_canvas,
        )
        return True

    def _rasterized(self, path: Path, width: int, height: int) -> Image.Image:
        """Rasterize a general SVG path once for a stable file and target size."""

        key = (str(path.resolve()), path.stat().st_mtime_ns, width, height)
        cached = self._raster_cache.get(key)
        if cached is not None:
            return cached
        png_bytes = resvg_py.svg_to_bytes(
            svg_path=str(path),
            width=width,
            height=height,
            shape_rendering="geometric_precision",
            text_rendering="optimize_legibility",
            image_rendering="optimize_quality",
        )
        with Image.open(BytesIO(png_bytes)) as image:
            rendered = image.convert("RGBA")
        if len(self._raster_cache) >= self._raster_cache_limit:
            oldest = next(iter(self._raster_cache))
            self._raster_cache.pop(oldest)
        self._raster_cache[key] = rendered
        return rendered

    def _draw_elements(
        self,
        canvas: Image.Image,
        obj: RenderableObject,
        root: ElementTree.Element,
        view_min_x: float,
        view_min_y: float,
        view_width: float,
        view_height: float,
    ) -> int:
        """Scale and draw supported SVG child elements into an object box."""

        drawing = ImageDraw.Draw(canvas)
        parents = {
            child: parent
            for parent in root.iter()
            for child in list(parent)
        }
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
                    fill=self._paint(
                        self._inherited(element, parents, "stroke", "none"),
                        self._inherited(element, parents, "color", "black"),
                    ),
                    width=self._stroke_width(
                        element,
                        scale_x,
                        scale_y,
                        self._inherited(element, parents, "stroke-width", "1"),
                    ),
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
                    drawing.polygon(
                        points,
                        fill=self._paint(
                            self._inherited(element, parents, "fill", "black"),
                            self._inherited(element, parents, "color", "black"),
                        ),
                    )
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
                    fill=self._paint(
                        self._inherited(element, parents, "fill", "black"),
                        self._inherited(element, parents, "color", "black"),
                    ),
                    outline=self._paint(
                        self._inherited(element, parents, "stroke", "none"),
                        self._inherited(element, parents, "color", "black"),
                    ),
                    width=self._stroke_width(
                        element,
                        scale_x,
                        scale_y,
                        self._inherited(element, parents, "stroke-width", "1"),
                    ),
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
                    fill=self._paint(
                        self._inherited(element, parents, "fill", "black"),
                        self._inherited(element, parents, "color", "black"),
                    ),
                    outline=self._paint(
                        self._inherited(element, parents, "stroke", "none"),
                        self._inherited(element, parents, "color", "black"),
                    ),
                    width=self._stroke_width(
                        element,
                        scale_x,
                        scale_y,
                        self._inherited(element, parents, "stroke-width", "1"),
                    ),
                )
                rendered_elements += 1
            elif tag == "ellipse":
                center_x, center_y = self._point(
                    element.get("cx"),
                    element.get("cy"),
                    left,
                    top,
                    scale_x,
                    scale_y,
                )
                radius_x = self._number(element.get("rx")) * scale_x
                radius_y = self._number(element.get("ry")) * scale_y
                drawing.ellipse(
                    (
                        center_x - radius_x,
                        center_y - radius_y,
                        center_x + radius_x,
                        center_y + radius_y,
                    ),
                    fill=self._paint(
                        self._inherited(element, parents, "fill", "black"),
                        self._inherited(element, parents, "color", "black"),
                    ),
                    outline=self._paint(
                        self._inherited(element, parents, "stroke", "none"),
                        self._inherited(element, parents, "color", "black"),
                    ),
                    width=self._stroke_width(
                        element,
                        scale_x,
                        scale_y,
                        self._inherited(element, parents, "stroke-width", "1"),
                    ),
                )
                rendered_elements += 1
            elif tag == "path":
                rendered_elements += self._draw_path(
                    drawing,
                    element,
                    left,
                    top,
                    scale_x,
                    scale_y,
                    view_min_x,
                    view_min_y,
                    fill=self._paint(
                        self._inherited(element, parents, "fill", "black"),
                        self._inherited(element, parents, "color", "black"),
                    ),
                    stroke=self._paint(
                        self._inherited(element, parents, "stroke", "none"),
                        self._inherited(element, parents, "color", "black"),
                    ),
                    stroke_width=self._stroke_width(
                        element,
                        scale_x,
                        scale_y,
                        self._inherited(element, parents, "stroke-width", "1"),
                    ),
                )
            elif tag == "text" and (element.text or "").strip():
                x, y = self._point(
                    element.get("x"),
                    element.get("y"),
                    left,
                    top,
                    scale_x,
                    scale_y,
                )
                font_size = max(
                    8,
                    round(
                        self._number(element.get("font-size") or "16")
                        * (scale_x + scale_y)
                        / 2
                    ),
                )
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except OSError:
                    font = ImageFont.load_default(size=font_size)
                anchor = "ma" if element.get("text-anchor") == "middle" else "la"
                drawing.text(
                    (x, y),
                    element.text.strip(),
                    fill=self._paint(
                        self._inherited(element, parents, "fill", "black"),
                        self._inherited(element, parents, "color", "black"),
                    ),
                    font=font,
                    anchor=anchor,
                )
                rendered_elements += 1

        return rendered_elements

    def _draw_path(
        self,
        drawing: ImageDraw.ImageDraw,
        element: ElementTree.Element,
        left: float,
        top: float,
        scale_x: float,
        scale_y: float,
        view_min_x: float,
        view_min_y: float,
        *,
        fill: str | None,
        stroke: str | None,
        stroke_width: int,
    ) -> int:
        """Render common icon paths without requiring a native SVG library.

        The fallback understands the path commands used by the bundled icon
        catalogs. Curves are represented by their endpoints, which preserves
        the asset silhouette and keeps rendering deterministic when resvg is
        unavailable. Native resvg remains the preferred high-fidelity path.
        """

        tokens = re.findall(
            r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            element.get("d", ""),
        )
        arity = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}
        index = 0
        command = ""
        current = (0.0, 0.0)
        start = current
        previous_cubic_control: tuple[float, float] | None = None
        previous_quadratic_control: tuple[float, float] | None = None
        subpaths: list[tuple[list[tuple[float, float]], bool]] = []
        points: list[tuple[float, float]] = []
        closed = False

        def finish() -> None:
            nonlocal points, closed
            if points:
                subpaths.append((points, closed))
            points = []
            closed = False

        while index < len(tokens):
            if tokens[index].isalpha():
                command = tokens[index]
                index += 1
                if command in "Zz":
                    current = start
                    if points and points[-1] != start:
                        points.append(start)
                    closed = True
                    finish()
                    command = ""
                    previous_cubic_control = None
                    previous_quadratic_control = None
                    continue
            if not command or command.upper() not in arity:
                index += 1
                continue
            upper = command.upper()
            count = arity[upper]
            if index + count > len(tokens) or any(
                token.isalpha() for token in tokens[index:index + count]
            ):
                command = ""
                continue
            values = [float(token) for token in tokens[index:index + count]]
            index += count
            relative = command.islower()
            old = current
            if upper in {"M", "L"}:
                target = (values[0], values[1])
            elif upper == "H":
                target = (values[0], old[1])
            elif upper == "V":
                target = (old[0], values[0])
            elif upper == "T":
                target = (values[0], values[1])
            elif upper in {"C", "S", "Q"}:
                target = (values[-2], values[-1])
            elif upper == "A":
                # A/a: the final pair is the arc endpoint.
                target = (values[5], values[6])
            else:
                command = ""
                continue
            if relative:
                target = (old[0] + target[0], old[1] + target[1])
            current = target
            if upper == "M":
                if points:
                    finish()
                start = target
                points = [target]
                command = "l" if relative else "L"
                previous_cubic_control = None
                previous_quadratic_control = None
            elif upper == "C":
                controls = [
                    (values[0], values[1]),
                    (values[2], values[3]),
                ]
                if relative:
                    controls = [
                        (old[0] + x, old[1] + y) for x, y in controls
                    ]
                points.extend(_sample_cubic(old, controls[0], controls[1], target))
                previous_cubic_control = controls[1]
                previous_quadratic_control = None
            elif upper == "S":
                control = (
                    (
                        2 * old[0] - previous_cubic_control[0],
                        2 * old[1] - previous_cubic_control[1],
                    )
                    if previous_cubic_control is not None
                    else old
                )
                second = (values[0], values[1])
                if relative:
                    second = (old[0] + second[0], old[1] + second[1])
                points.extend(_sample_cubic(old, control, second, target))
                previous_cubic_control = second
                previous_quadratic_control = None
            elif upper == "Q":
                control = (values[0], values[1])
                if relative:
                    control = (old[0] + control[0], old[1] + control[1])
                points.extend(_sample_quadratic(old, control, target))
                previous_quadratic_control = control
                previous_cubic_control = None
            elif upper == "T":
                control = (
                    (
                        2 * old[0] - previous_quadratic_control[0],
                        2 * old[1] - previous_quadratic_control[1],
                    )
                    if previous_quadratic_control is not None
                    else old
                )
                points.extend(_sample_quadratic(old, control, target))
                previous_quadratic_control = control
                previous_cubic_control = None
            elif upper == "A":
                arc_values = values
                if relative:
                    arc_values = [*values]
                    arc_values[5] += old[0]
                    arc_values[6] += old[1]
                points.extend(_sample_arc(old, arc_values))
                previous_cubic_control = None
                previous_quadratic_control = None
            else:
                points.append(target)
                previous_cubic_control = None
                previous_quadratic_control = None
        finish()

        rendered = 0
        for path_points, is_closed in subpaths:
            if len(path_points) < 2:
                continue
            scaled = [
                (
                    left + (x - view_min_x) * scale_x,
                    top + (y - view_min_y) * scale_y,
                )
                for x, y in path_points
            ]
            if fill is not None and len(scaled) >= 3:
                drawing.polygon(scaled, fill=fill)
            if stroke is not None:
                drawing.line(
                    scaled + ([scaled[0]] if is_closed else []),
                    fill=stroke,
                    width=stroke_width,
                )
            rendered += 1
        return rendered

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

    def _view_bounds(self, root: ElementTree.Element) -> tuple[float, float, float, float]:
        """Read positive dimensions from viewBox or width and height."""

        view_box = root.get("viewBox")
        if view_box:
            values = [float(item) for item in view_box.replace(",", " ").split()]
            if len(values) == 4 and values[2] > 0 and values[3] > 0:
                return values[0], values[1], values[2], values[3]

        width = self._number(root.get("width"))
        height = self._number(root.get("height"))
        if width <= 0 or height <= 0:
            raise ValueError("SVG dimensions must be positive")
        return 0.0, 0.0, width, height

    def _view_size(self, root: ElementTree.Element) -> tuple[float, float]:
        """Return dimensions for callers that do not need the viewBox offset."""

        _, _, width, height = self._view_bounds(root)
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
        value: str | None = None,
    ) -> int:
        """Scale an SVG stroke width to the target object box."""

        source_width = self._number(value or element.get("stroke-width") or "1")
        # Catalog icons are commonly authored at 24x24 but laid out in a
        # several-hundred-pixel semantic box.  Scaling a 2px source stroke
        # linearly makes the fallback paint giant black bars.  Native SVG
        # rasterizers handle this correctly; keep the dependency-free fallback
        # visually legible and bounded instead.
        return min(4, max(1, round(source_width * (scale_x + scale_y) / 2)))

    @staticmethod
    def _paint(value: str | None, current_color: str = "black") -> str | None:
        """Convert the SVG none keyword to Pillow's transparent paint."""

        if value is None or value.lower() == "none":
            return None
        return current_color if value.casefold() == "currentcolor" else value

    @staticmethod
    def _inherited(
        element: ElementTree.Element,
        parents: dict[ElementTree.Element, ElementTree.Element],
        attribute: str,
        default: str,
    ) -> str:
        """Resolve presentation attributes through SVG parent groups/styles."""

        current: ElementTree.Element | None = element
        while current is not None:
            value = current.get(attribute)
            if value is not None and value.strip():
                return value.strip()
            style = current.get("style", "")
            for declaration in style.split(";"):
                key, separator, declared = declaration.partition(":")
                if separator and key.strip() == attribute and declared.strip():
                    return declared.strip()
            current = parents.get(current)
        return default
