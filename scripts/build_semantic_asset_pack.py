"""Vendor and index complete Lucide and Material Symbols SVG distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil


CATEGORY_TERMS = {
    "people": {"user", "person", "people", "face", "group", "child", "man", "woman"},
    "communication": {"mail", "message", "chat", "phone", "call", "send", "inbox"},
    "data": {"database", "table", "chart", "analytics", "data", "storage", "folder", "file"},
    "computing": {"code", "terminal", "server", "computer", "cpu", "memory", "network", "api"},
    "security": {"lock", "key", "shield", "security", "verified", "password"},
    "commerce": {"money", "coin", "bank", "cart", "store", "payment", "price", "wallet"},
    "science": {"science", "atom", "dna", "experiment", "lab", "planet", "orbit", "calculate"},
    "health": {"health", "heart", "medical", "hospital", "pill", "stethoscope", "body"},
    "navigation": {"arrow", "direction", "map", "route", "navigation", "compass"},
    "media": {"image", "video", "music", "audio", "camera", "mic", "play"},
    "nature": {"tree", "flower", "leaf", "animal", "water", "sun", "cloud", "earth"},
    "transport": {"car", "bus", "train", "flight", "bike", "ship", "traffic"},
}


def tokens(value: str) -> set[str]:
    """Split icon metadata into stable search terms."""

    return set(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ").replace("-", " ")))


def categories(values: set[str]) -> list[str]:
    """Infer broad compositional categories from icon vocabulary."""

    result = [
        category
        for category, keywords in CATEGORY_TERMS.items()
        if values.intersection(keywords)
    ]
    return sorted(result or ["general"])


def relations(values: set[str]) -> list[str]:
    """Infer reusable relation hints from names and upstream tags."""

    mapping = {
        "flows-to": {"arrow", "send", "forward", "upload", "download", "route", "transfer"},
        "compares-with": {"compare", "swap", "exchange", "difference", "contrast"},
        "orbits": {"orbit", "satellite", "planet", "gravity"},
        "blocks": {"block", "shield", "lock", "security", "guard", "cancel"},
        "repeats": {"repeat", "cycle", "refresh", "sync", "loop", "rotate"},
        "transforms-into": {"transform", "convert", "change", "wand", "sparkles"},
        "inside": {"inside", "container", "box", "inbox", "frame"},
    }
    return sorted(
        relation for relation, keywords in mapping.items() if values.intersection(keywords)
    )


def roles(values: set[str]) -> list[str]:
    """Infer ordering roles used by the deterministic compositor."""

    mapping = {
        "container": {"container", "box", "frame", "folder", "inbox", "database", "world", "planet"},
        "content": {"file", "document", "item", "bolt", "energy", "person"},
        "center": {"world", "planet", "sun", "target", "hub"},
        "satellite": {"satellite", "moon", "arrow", "orbit"},
        "source": {"user", "input", "upload", "sender", "start"},
        "target": {"output", "download", "receiver", "result", "chart"},
        "barrier": {"shield", "lock", "block", "wall", "security"},
    }
    return sorted(role for role, keywords in mapping.items() if values.intersection(keywords))


def record(
    *,
    collection: str,
    icon_name: str,
    path: str,
    license_id: str,
    upstream_tags: list[str] | None = None,
) -> dict[str, object]:
    """Create one normalized catalog record from an upstream filename."""

    display_name = icon_name.replace("_", " ").replace("-", " ")
    tag_values = [item.strip().lower() for item in upstream_tags or [] if item.strip()]
    vocabulary = tokens(display_name + " " + " ".join(tag_values))
    return {
        "asset_id": f"full-{collection.lower().replace(' ', '-')}-{icon_name}",
        "source_collection": collection,
        "concepts": [display_name],
        "aliases": sorted(set(tag_values)),
        "categories": categories(vocabulary),
        "tags": sorted(vocabulary),
        "relations": relations(vocabulary),
        "composition_roles": roles(vocabulary),
        "path": path,
        "license_id": license_id,
    }


def build(lucide_root: Path, material_root: Path, destination: Path) -> int:
    """Copy approved variants and write a deterministic combined metadata index."""

    lucide_output = destination / "full" / "lucide"
    material_output = destination / "full" / "material-symbols-outlined"
    lucide_output.mkdir(parents=True, exist_ok=True)
    material_output.mkdir(parents=True, exist_ok=True)
    lucide_tags = json.loads((lucide_root / "tags.json").read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for source in sorted((lucide_root / "icons").glob("*.svg")):
        target = lucide_output / source.name
        shutil.copyfile(source, target)
        records.append(
            record(
                collection="Lucide",
                icon_name=source.stem,
                path=target.relative_to(destination.parents[2]).as_posix(),
                license_id="Lucide-ISC",
                upstream_tags=lucide_tags.get(source.stem, []),
            )
        )
    for source in sorted(material_root.glob("*.svg")):
        if source.stem.endswith("-fill"):
            continue
        target = material_output / source.name
        shutil.copyfile(source, target)
        records.append(
            record(
                collection="Material Symbols",
                icon_name=source.stem,
                path=target.relative_to(destination.parents[2]).as_posix(),
                license_id="Material-Symbols-Apache-2.0",
            )
        )
    index = destination / "full_index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "description": "Complete vendored Lucide and Material Symbols outlined SVG indexes.",
                "assets": sorted(records, key=lambda item: str(item["asset_id"])),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(lucide_root / "LICENSE", destination / "LUCIDE_LICENSE.txt")
    shutil.copyfile(material_root.parent / "LICENSE", destination / "MATERIAL_SYMBOLS_LICENSE.txt")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lucide-root", type=Path, required=True)
    parser.add_argument("--material-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    count = build(args.lucide_root, args.material_root, args.destination)
    print(f"Indexed {count} offline SVG assets.")


if __name__ == "__main__":
    main()
