"""Production visual-design tokens for consistent educational diagrams."""

from dataclasses import dataclass


Color = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ObjectStyle:
    """Resolved renderer styling for one semantic object."""

    ink: Color
    fill: Color
    accent: Color
    stroke_width: int
    corner_radius: int
    font_size: int


class WhiteboardDesignSystem:
    """Resolve a restrained, accessible whiteboard visual language."""

    background: Color = (255, 255, 253, 255)
    ink: Color = (29, 31, 35, 255)
    muted: Color = (137, 142, 150, 255)
    blue: Color = (42, 108, 205, 255)
    green: Color = (29, 142, 90, 255)
    orange: Color = (220, 111, 35, 255)
    purple: Color = (123, 83, 178, 255)
    red: Color = (197, 65, 65, 255)
    yellow: Color = (255, 224, 102, 170)

    _TOKEN_ACCENTS: dict[str, Color] = {
        "concept.primary": blue,
        "concept.secondary": purple,
        "process.active": orange,
        "data.input": blue,
        "data.output": green,
        "success": green,
        "warning": orange,
        "error": red,
        "annotation": purple,
    }

    _KIND_FONT_SIZES: dict[str, int] = {
        "equation": 42,
        "text": 34,
        "label": 32,
        "annotation": 30,
        "component": 30,
        "graph_node": 28,
        "tree_node": 28,
        "array_cell": 28,
        "matrix_cell": 26,
    }

    def resolve(
        self,
        kind: str,
        style_token: str,
        *,
        dimmed: bool = False,
    ) -> ObjectStyle:
        """Return stable tokens for a semantic kind and style token."""

        accent = self._TOKEN_ACCENTS.get(style_token, self.blue)
        ink = self.muted if dimmed else self.ink
        fill = self._tint(accent, 0.94)
        if kind in {"text", "label", "annotation", "equation", "connector"}:
            fill = (255, 255, 255, 0)
        return ObjectStyle(
            ink=ink,
            fill=fill,
            accent=accent,
            stroke_width=5 if kind != "annotation" else 3,
            corner_radius=38 if kind in {"graph_node", "tree_node"} else 18,
            font_size=self._KIND_FONT_SIZES.get(kind, 30),
        )

    @staticmethod
    def _tint(color: Color, amount: float) -> Color:
        """Blend a color toward white while retaining opaque output."""

        return tuple(
            round(channel + (255 - channel) * amount)
            for channel in color[:3]
        ) + (245,)
