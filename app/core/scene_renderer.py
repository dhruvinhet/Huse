"""Static Pillow rendering orchestration for renderable scenes."""

from pathlib import Path

from loguru import logger
from PIL import Image

from app.models.render import RenderScene, RenderableObject
from app.renderers.icon_renderer import IconRenderer
from app.renderers.image_renderer import ImageRenderer
from app.renderers.svg_renderer import SVGRenderer
from app.renderers.text_renderer import TextRenderer


class SceneRenderer:
    """Delegate scene objects to specialized static renderers."""

    CANVAS_SIZE = (1920, 1080)
    BACKGROUND_COLOR = "white"
    RENDER_ORDER = {
        "text": 0,
        "svg": 1,
        "icon": 2,
        "image": 3,
    }

    def __init__(self) -> None:
        """Initialize one instance of each specialized renderer."""

        self._text_renderer = TextRenderer()
        self._svg_renderer = SVGRenderer()
        self._icon_renderer = IconRenderer()
        self._image_renderer = ImageRenderer()

    def render(self, scene: RenderScene, output_path: str) -> None:
        """Render one scene to a white 1920x1080 PNG image."""

        canvas = Image.new("RGB", self.CANVAS_SIZE, self.BACKGROUND_COLOR)
        objects_rendered = 0
        objects_skipped = 0

        ordered_objects = sorted(
            scene.objects,
            key=lambda item: self.RENDER_ORDER.get(item.type.lower(), 99),
        )
        for obj in ordered_objects:
            renderer = self._renderer_for(obj)
            if renderer is None:
                logger.warning(
                    "Skipping object {} with unsupported type {}.",
                    obj.object_id,
                    obj.type,
                )
                objects_skipped += 1
                continue
            try:
                was_rendered = renderer.render(canvas, obj)
            except Exception as exc:
                logger.warning("Skipping object {}: {}", obj.object_id, exc)
                objects_skipped += 1
                continue

            if was_rendered:
                objects_rendered += 1
            else:
                objects_skipped += 1

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="PNG")
        logger.info(
            "Static scene rendered (objects_rendered={}, objects_skipped={}, "
            "output_path={}).",
            objects_rendered,
            objects_skipped,
            destination,
        )

    def _renderer_for(
        self,
        obj: RenderableObject,
    ) -> TextRenderer | SVGRenderer | IconRenderer | ImageRenderer | None:
        """Return the specialized renderer for an object type, if supported."""

        renderers = {
            "text": self._text_renderer,
            "svg": self._svg_renderer,
            "icon": self._icon_renderer,
            "image": self._image_renderer,
        }
        return renderers.get(obj.type.strip().lower())
