"""Optional CairoSVG-backed rendering for local SVG icons."""

from io import BytesIO
from pathlib import Path

from loguru import logger
from PIL import Image

from app.config.settings import PROJECT_ROOT
from app.models.render import RenderableObject

try:
    import cairosvg
except ImportError:
    cairosvg = None


class IconRenderer:
    """Rasterize a local SVG icon when CairoSVG is available."""

    def render(self, canvas: Image.Image, obj: RenderableObject) -> bool:
        """Render an SVG icon, or log and skip when support is unavailable."""

        if cairosvg is None:
            logger.warning(
                "Skipping icon {} because CairoSVG is unavailable.",
                obj.object_id,
            )
            return False

        configured_path = Path(obj.content)
        icon_path = (
            configured_path
            if configured_path.is_absolute()
            else PROJECT_ROOT / configured_path
        )
        if not icon_path.is_file():
            logger.warning(
                "Skipping icon {} because {} does not exist.",
                obj.object_id,
                icon_path,
            )
            return False

        try:
            png_bytes = cairosvg.svg2png(
                url=str(icon_path),
                output_width=obj.width,
                output_height=obj.height,
            )
            with Image.open(BytesIO(png_bytes)) as icon_image:
                rendered_icon = icon_image.convert("RGBA")
            self._paste_centered(canvas, rendered_icon, obj)
        except Exception as exc:
            logger.warning("Skipping icon {}: {}", obj.object_id, exc)
            return False
        return True

    @staticmethod
    def _paste_centered(
        canvas: Image.Image,
        icon: Image.Image,
        obj: RenderableObject,
    ) -> None:
        """Paste an RGBA icon using object coordinates as its center."""

        position = (obj.x - obj.width // 2, obj.y - obj.height // 2)
        canvas.paste(icon, position, icon)
