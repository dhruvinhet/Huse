"""Export versioned V2 JSON Schemas into the documentation tree."""

from pathlib import Path

from app.domain.schemas import export_schemas


def main() -> None:
    """Generate all registered JSON Schema documents."""

    paths = export_schemas(Path("docs/schemas"))
    print(f"Exported {len(paths)} schemas to docs/schemas")


if __name__ == "__main__":
    main()
