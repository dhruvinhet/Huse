"""Pillow-based rendering for text scene objects."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.config.settings import PROJECT_ROOT
from app.models.render import RenderableObject


class TextRenderer:
    """Draw centered text with a system font and a safe default fallback."""

    FONT_CANDIDATES = (
        PROJECT_ROOT / "assets" / "fonts" / "DejaVuSans.ttf",
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )

    def render(self, canvas: Image.Image, obj: RenderableObject) -> bool:
        """Draw a text object onto a Pillow canvas."""

        font_size = max(12, min(obj.height, 72))
        font = self._load_font(font_size)
        drawing = ImageDraw.Draw(canvas)
        wrapped_text = self._wrap_text(
            drawing,
            obj.content,
            font,
            obj.width,
        )
        drawing.multiline_text(
            (obj.x, obj.y),
            wrapped_text,
            fill="black",
            font=font,
            anchor="mm",
            align="center",
            spacing=8,
        )
        return True

    def _load_font(
        self,
        font_size: int,
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load the first available custom/system font or Pillow default."""

        for font_path in self.FONT_CANDIDATES:
            if not font_path.is_file():
                continue
            try:
                return ImageFont.truetype(str(font_path), size=font_size)
            except OSError:
                continue

        try:
            return ImageFont.load_default(size=font_size)
        except TypeError:
            return ImageFont.load_default()

    @staticmethod
    def _wrap_text(
        drawing: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
    ) -> str:
        """Wrap display text to the renderable object's pixel width."""

        wrapped_lines: list[str] = []
        for paragraph in text.splitlines() or [text]:
            current_line = ""
            for word in paragraph.split():
                candidate = f"{current_line} {word}".strip()
                bounds = drawing.textbbox((0, 0), candidate, font=font)
                if current_line and bounds[2] - bounds[0] > max_width:
                    wrapped_lines.append(current_line)
                    current_line = word
                else:
                    current_line = candidate
            wrapped_lines.append(current_line)
        return "\n".join(wrapped_lines)
