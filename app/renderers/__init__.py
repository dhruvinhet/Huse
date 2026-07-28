"""Specialized static object renderers."""

from app.renderers.icon_renderer import IconRenderer
from app.renderers.image_renderer import ImageRenderer
from app.renderers.svg_renderer import SVGRenderer
from app.renderers.text_renderer import TextRenderer

__all__ = [
    "IconRenderer",
    "ImageRenderer",
    "SVGRenderer",
    "TextRenderer",
]
