"""Pillow rendering for images and unresolved image placeholders."""

from pathlib import Path

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from app.config.settings import PROJECT_ROOT
from app.models.render import RenderableObject


class ImageRenderer:
    """Draw a local image or a clean descriptive whiteboard fallback."""

    DESCRIPTION_PREFIX = "description:"

    def render(self, canvas: Image.Image, obj: RenderableObject) -> bool:
        """Render a local image, falling back to a light gray placeholder."""

        if obj.content.startswith(self.DESCRIPTION_PREFIX):
            description = obj.content.removeprefix(self.DESCRIPTION_PREFIX)
            self._draw_fallback(canvas, obj, description)
            return True

        configured_path = Path(obj.content)
        image_path = (
            configured_path
            if configured_path.is_absolute()
            else PROJECT_ROOT / configured_path
        )
        if image_path.is_file():
            try:
                self._paste_image(canvas, obj, image_path)
                return True
            except OSError as exc:
                logger.warning(
                    "Could not load image {} at {}: {}. Drawing placeholder.",
                    obj.object_id,
                    image_path,
                    exc,
                )

        fallback_text = configured_path.stem.replace("_", " ").strip()
        self._draw_fallback(
            canvas,
            obj,
            fallback_text or "Supporting visual",
        )
        return True

    @staticmethod
    def _paste_image(
        canvas: Image.Image,
        obj: RenderableObject,
        image_path: Path,
    ) -> None:
        """Resize and center a local image in its renderable object box."""

        with Image.open(image_path) as source_image:
            rendered_image = source_image.convert("RGBA").resize(
                (obj.width, obj.height),
                Image.Resampling.LANCZOS,
            )
        position = (obj.x - obj.width // 2, obj.y - obj.height // 2)
        canvas.paste(rendered_image, position, rendered_image)

    def _draw_fallback(
        self,
        canvas: Image.Image,
        obj: RenderableObject,
        description: str,
    ) -> None:
        """Draw a compact whiteboard card containing the visual description."""

        left = obj.x - obj.width // 2
        top = obj.y - obj.height // 2
        right = left + obj.width
        bottom = top + obj.height
        drawing = ImageDraw.Draw(canvas)
        drawing.rounded_rectangle(
            (left, top, right, bottom),
            radius=24,
            fill="white",
            outline="black",
            width=4,
        )
        try:
            font = ImageFont.load_default(size=26)
        except TypeError:
            font = ImageFont.load_default()

        icon_center_x = left + 70
        icon_center_y = obj.y
        drawing.ellipse(
            (
                icon_center_x - 30,
                icon_center_y - 30,
                icon_center_x + 30,
                icon_center_y + 30,
            ),
            outline="black",
            width=4,
        )
        drawing.line(
            (
                icon_center_x - 42,
                icon_center_y + 48,
                icon_center_x,
                icon_center_y + 14,
                icon_center_x + 42,
                icon_center_y + 48,
            ),
            fill="black",
            width=4,
            joint="curve",
        )

        text_left = left + 130
        wrapped_text = self._wrap_text(
            drawing,
            description,
            font,
            max(80, right - text_left - 28),
        )
        drawing.multiline_text(
            (text_left, obj.y),
            wrapped_text,
            fill="black",
            font=font,
            anchor="lm",
            spacing=8,
        )

    @staticmethod
    def _wrap_text(
        drawing: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
    ) -> str:
        """Wrap words to a pixel width suitable for the fallback card."""

        lines: list[str] = []
        current_line = ""
        for word in text.split():
            candidate = f"{current_line} {word}".strip()
            bounds = drawing.textbbox((0, 0), candidate, font=font)
            if current_line and bounds[2] - bounds[0] > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = candidate
        if current_line:
            lines.append(current_line)
        return "\n".join(lines) or "Supporting visual"
