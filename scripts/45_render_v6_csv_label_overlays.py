#!/usr/bin/env python
"""Render v6 CSV annotation labels over source images for review."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_LABELS = V6_DIR / "pseudo_labels" / "labels_pseudo_rgb.csv"
DEFAULT_OUTPUT_DIR = V6_DIR / "overlays" / "pseudo_rgb_csv"
CLASS_COLORS = {
    "single_cell": (0, 255, 255),
    "mother_bud_pair": (255, 210, 0),
    "early_bud_pair": (255, 80, 220),
}


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def read_labels(path: Path) -> dict[str, list[dict[str, str]]]:
    labels: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("image_id"):
                labels[row["image_id"]].append(row)
    return labels


def draw_overlay(image_path: Path, rows: list[dict[str, str]], output_path: Path, hide_text: bool) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    count = 0
    for row in rows:
        class_name = row["class_name"]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        if hide_text:
            count += 1
            continue
        label = class_name
        if row.get("review_status"):
            label = f"{label} [{row['review_status']}]"
        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0))
        count += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return count


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only-with-labels", action="store_true")
    parser.add_argument("--hide-text", action="store_true", help="Draw box outlines only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = read_manifest(args.manifest)
    labels = read_labels(args.labels)
    rendered = 0
    total_boxes = 0
    for image_id, meta in manifest.items():
        rows = labels.get(image_id, [])
        if args.only_with_labels and not rows:
            continue
        image_path = PROJECT_ROOT / meta["image_path"]
        output_path = args.output_dir / f"{Path(meta['image_path']).stem}_csv_overlay.png"
        total_boxes += draw_overlay(image_path, rows, output_path, hide_text=bool(args.hide_text))
        rendered += 1
    print(f"Rendered {rendered} CSV-label overlays with {total_boxes} boxes to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
