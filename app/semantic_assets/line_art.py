"""Deterministic whiteboard line-art fallback for unresolved concepts."""

from pathlib import Path

import svgwrite

from app.domain.assets import AssetQuery


class DeterministicLineArtGenerator:
    """Generate editable semantic pictograms in one coherent line-art style."""

    SIZE = 512

    def generate(self, query: AssetQuery, output_path: Path) -> Path:
        """Generate a stable whiteboard-style semantic badge."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        drawing = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{self.SIZE}px", f"{self.SIZE}px"),
            profile="tiny",
        )
        drawing.viewbox(0, 0, self.SIZE, self.SIZE)
        group = drawing.g(
            fill="white",
            stroke="#202124",
            stroke_width=10,
            stroke_linecap="round",
            stroke_linejoin="round",
        )
        concept = self._category(query.concept, query.required_semantics)
        self._draw_icon(drawing, group, concept)
        drawing.add(group)
        label = query.concept[:36]
        drawing.add(
            drawing.text(
                label,
                insert=(256, 450),
                text_anchor="middle",
                font_size=28,
                font_family="sans-serif",
                fill="black",
            )
        )
        drawing.save(pretty=True)
        return output_path

    @staticmethod
    def _category(concept: str, semantics: list[str]) -> str:
        """Map free-form semantic queries to a reviewed icon category."""

        terms = f"{concept} {' '.join(semantics)}".lower()
        categories = {
            "database": ("database", "storage", "data store", "sql"),
            "cloud": ("cloud", "internet"),
            "server": ("server", "compute", "backend"),
            "person": ("person", "user", "student", "human", "customer"),
            "team": ("team", "people", "group"),
            "document": ("document", "file", "paper", "report", "book"),
            "security": ("security", "lock", "shield", "auth", "privacy"),
            "money": ("money", "finance", "price", "cost", "revenue"),
            "network": ("network", "nodes", "graph", "connection"),
            "ai": ("ai", "robot", "model", "neural", "llm"),
            "device": ("phone", "mobile", "browser", "computer", "client"),
            "education": ("education", "learn", "school", "course"),
            "health": ("health", "medical", "hospital", "patient"),
        }
        for category, keywords in categories.items():
            if any(keyword in terms for keyword in keywords):
                return category
        return "idea"

    def _draw_icon(self, drawing: svgwrite.Drawing, group: object, category: str) -> None:
        """Draw one recognizable icon using editable SVG primitives."""

        add = group.add
        if category == "database":
            add(drawing.ellipse(center=(256, 128), r=(118, 48)))
            add(drawing.path(d="M138 128 V330 C138 394 374 394 374 330 V128"))
            add(drawing.path(d="M138 226 C138 290 374 290 374 226"))
            add(drawing.path(d="M138 318 C138 382 374 382 374 318"))
        elif category == "cloud":
            add(drawing.path(d="M132 342 C65 342 68 240 146 230 C160 126 303 105 344 199 C438 191 460 338 365 342 Z"))
        elif category == "server":
            for y in (92, 205, 318):
                add(drawing.rect(insert=(112, y), size=(288, 82), rx=14))
                add(drawing.circle(center=(356, y + 41), r=9, fill="#2a6ccd", stroke="none"))
        elif category in {"person", "team"}:
            centers = (256,) if category == "person" else (170, 256, 342)
            for x in centers:
                add(drawing.circle(center=(x, 164), r=48 if category == "person" else 36))
                add(drawing.path(d=f"M{x-74} 350 Q{x} 246 {x+74} 350"))
        elif category == "document":
            add(drawing.path(d="M142 72 H322 L390 140 V400 H142 Z"))
            add(drawing.path(d="M322 72 V140 H390"))
            for y in (206, 262, 318):
                add(drawing.line(start=(190, y), end=(340, y)))
        elif category == "security":
            add(drawing.path(d="M256 68 L390 118 V224 C390 324 334 386 256 420 C178 386 122 324 122 224 V118 Z"))
            add(drawing.rect(insert=(202, 220), size=(108, 96), rx=14))
            add(drawing.path(d="M220 220 V188 C220 138 292 138 292 188 V220"))
        elif category == "money":
            add(drawing.circle(center=(256, 244), r=148))
            add(drawing.path(d="M310 170 C285 136 205 144 205 190 C205 238 309 218 309 276 C309 330 221 338 190 298"))
            add(drawing.line(start=(256, 126), end=(256, 354)))
        elif category == "network":
            nodes = ((256, 100), (128, 240), (384, 240), (200, 380), (330, 380))
            for first, second in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4)):
                add(drawing.line(start=nodes[first], end=nodes[second]))
            for point in nodes:
                add(drawing.circle(center=point, r=26, fill="white"))
        elif category == "ai":
            add(drawing.rect(insert=(130, 118), size=(252, 244), rx=54))
            add(drawing.line(start=(256, 74), end=(256, 118)))
            add(drawing.circle(center=(256, 62), r=13, fill="#2a6ccd"))
            add(drawing.circle(center=(205, 218), r=16, fill="#2a6ccd"))
            add(drawing.circle(center=(307, 218), r=16, fill="#2a6ccd"))
            add(drawing.path(d="M194 292 Q256 332 318 292"))
        elif category == "device":
            add(drawing.rect(insert=(105, 105), size=(302, 210), rx=18))
            add(drawing.line(start=(176, 372), end=(336, 372)))
            add(drawing.path(d="M220 315 L202 372 M292 315 L310 372"))
        elif category == "education":
            add(drawing.path(d="M86 190 L256 96 L426 190 L256 284 Z"))
            add(drawing.path(d="M150 228 V326 Q256 392 362 326 V228"))
            add(drawing.path(d="M426 190 V320"))
        elif category == "health":
            add(drawing.path(d="M256 404 C122 328 94 246 130 176 C170 96 246 130 256 180 C266 130 342 96 382 176 C418 246 390 328 256 404 Z"))
            add(drawing.path(d="M256 214 V326 M200 270 H312"))
        else:
            add(drawing.path(d="M256 74 C158 74 128 180 176 246 C198 276 210 292 214 326 H298 C302 292 314 276 336 246 C384 180 354 74 256 74 Z"))
            add(drawing.line(start=(216, 362), end=(296, 362)))
            add(drawing.line(start=(226, 396), end=(286, 396)))
