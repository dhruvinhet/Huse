"""Tests for static Pillow scene rendering and delegation."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from app.core.scene_renderer import SceneRenderer
from app.models.render import RenderableObject, RenderScene


def renderable(
    object_id: str,
    object_type: str,
    content: str,
    x: int = 960,
    y: int = 540,
    width: int = 256,
    height: int = 128,
) -> RenderableObject:
    """Create a renderable object for static renderer tests."""

    return RenderableObject(
        object_id=object_id,
        type=object_type,
        content=content,
        x=x,
        y=y,
        width=width,
        height=height,
        animation="none",
        start_time=0.0,
        end_time=1.0,
    )


def test_png_is_created(tmp_path: Path) -> None:
    """Rendering a scene creates a valid PNG file."""

    output_path = tmp_path / "scene-1.png"
    scene = RenderScene(
        scene_number=1,
        objects=[renderable("text-1", "text", "Hello whiteboard")],
    )

    SceneRenderer().render(scene, str(output_path))

    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_canvas_size_is_1920_by_1080(tmp_path: Path) -> None:
    """Rendered scene images use the required full-HD canvas dimensions."""

    output_path = tmp_path / "scene-size.png"
    SceneRenderer().render(
        RenderScene(scene_number=1, objects=[]),
        str(output_path),
    )

    with Image.open(output_path) as rendered_image:
        assert rendered_image.size == (1920, 1080)
        assert rendered_image.getpixel((0, 0)) == (255, 255, 255)


def test_objects_are_delegated_in_required_order(tmp_path: Path) -> None:
    """SceneRenderer delegates text, SVG, icon, and image objects in order."""

    renderer = SceneRenderer()
    call_order: list[str] = []

    def record(name: str) -> Callable[..., bool]:
        def render_object(*_args) -> bool:
            call_order.append(name)
            return True

        return render_object

    renderer._text_renderer.render = MagicMock(side_effect=record("text"))
    renderer._svg_renderer.render = MagicMock(side_effect=record("svg"))
    renderer._icon_renderer.render = MagicMock(side_effect=record("icon"))
    renderer._image_renderer.render = MagicMock(side_effect=record("image"))
    scene = RenderScene(
        scene_number=1,
        objects=[
            renderable("image-1", "image", "missing.png"),
            renderable("icon-1", "icon", "missing.svg"),
            renderable("svg-1", "svg", "missing.svg"),
            renderable("text-1", "text", "Text"),
        ],
    )

    renderer.render(scene, str(tmp_path / "delegation.png"))

    assert call_order == ["text", "svg", "icon", "image"]
    renderer._text_renderer.render.assert_called_once()
    renderer._svg_renderer.render.assert_called_once()
    renderer._icon_renderer.render.assert_called_once()
    renderer._image_renderer.render.assert_called_once()


def test_missing_image_draws_whiteboard_fallback(tmp_path: Path) -> None:
    """A pending image renders a clean card instead of a gray placeholder."""

    output_path = tmp_path / "placeholder.png"
    scene = RenderScene(
        scene_number=1,
        objects=[
            renderable(
                "image-1",
                "image",
                "description:A clear supporting diagram",
                width=512,
                height=512,
            )
        ],
    )

    SceneRenderer().render(scene, str(output_path))

    with Image.open(output_path) as rendered_image:
        fallback_region = rendered_image.crop((704, 284, 1217, 797))
        pixels = fallback_region.get_flattened_data()
        assert sum(pixel == (211, 211, 211) for pixel in pixels) < 1000
        assert (0, 0, 0) in pixels
        assert rendered_image.getpixel((100, 100)) == (255, 255, 255)
