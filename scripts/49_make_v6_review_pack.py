#!/usr/bin/env python
"""Create per-class review crops and contact sheets from a v6 label CSV."""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_LABELS = V6_DIR / "pseudo_labels" / "labels_pseudo_rgb.csv"
DEFAULT_OUTPUT_DIR = V6_DIR / "review" / "pseudo_rgb"
CLASS_COLORS = {
    "single_cell": (0, 255, 255),
    "mother_bud_pair": (255, 210, 0),
    "early_bud_pair": (255, 80, 220),
}


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("image_id") and row.get("class_name")]


def crop_with_context(image: Image.Image, row: dict[str, str], context: int) -> Image.Image:
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])
    left = max(0, int(math.floor(x1 - context)))
    top = max(0, int(math.floor(y1 - context)))
    right = min(image.size[0], int(math.ceil(x2 + context)))
    bottom = min(image.size[1], int(math.ceil(y2 + context)))
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    color = CLASS_COLORS.get(row["class_name"], (255, 255, 255))
    draw.rectangle((x1 - left, y1 - top, x2 - left, y2 - top), outline=color, width=2)
    return crop


def save_contact_sheet(paths: list[Path], output_path: Path, columns: int, thumb_size: int) -> None:
    if not paths:
        return
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(image, (thumb_size, thumb_size))
        canvas = Image.new("RGB", (thumb_size, thumb_size), (8, 8, 8))
        x = (thumb_size - thumb.size[0]) // 2
        y = (thumb_size - thumb.size[1]) // 2
        canvas.paste(thumb, (x, y))
        thumbs.append(canvas)
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * thumb_size, rows * thumb_size), (20, 20, 20))
    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        sheet.paste(thumb, (col * thumb_size, row * thumb_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_html_index(rows: list[dict[str, str]], crop_paths: list[Path], output_path: Path) -> None:
    items = []
    for row, crop_path in zip(rows, crop_paths):
        rel = crop_path.relative_to(output_path.parent).as_posix()
        label = html.escape(f"{row['image_id']} | {row['class_name']} | {row.get('review_status', '')}")
        notes = html.escape(row.get("notes", ""))
        items.append(f"<figure><img src='{rel}'><figcaption>{label}<br>{notes}</figcaption></figure>")
    style = """
    body { font-family: Arial, sans-serif; background: #111; color: #eee; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
    figure { margin: 0; background: #1d1d1d; padding: 8px; }
    img { width: 100%; image-rendering: pixelated; }
    figcaption { font-size: 12px; overflow-wrap: anywhere; }
    """
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        + style
        + "</style></head><body><h1>v6 Label Review Pack</h1><div class='grid'>"
        + "\n".join(items)
        + "</div></body></html>",
        encoding="utf-8",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-per-class", type=int, default=80)
    parser.add_argument("--context", type=int, default=18)
    parser.add_argument("--thumb-size", type=int, default=160)
    parser.add_argument("--columns", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = read_manifest(args.manifest)
    labels = read_labels(args.labels)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in labels:
        by_class[row["class_name"]].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, str]] = []
    crop_paths: list[Path] = []
    for class_name in sorted(by_class):
        class_rows = by_class[class_name][: args.max_per_class]
        class_crop_paths: list[Path] = []
        for idx, row in enumerate(class_rows, start=1):
            meta = manifest.get(row["image_id"])
            if meta is None:
                continue
            image = Image.open(PROJECT_ROOT / meta["image_path"]).convert("RGB")
            crop = crop_with_context(image, row, args.context)
            crop_path = args.output_dir / "crops" / class_name / f"{idx:04d}_{row['image_id']}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path)
            selected_rows.append(row)
            crop_paths.append(crop_path)
            class_crop_paths.append(crop_path)
        save_contact_sheet(class_crop_paths, args.output_dir / "contact_sheets" / f"{class_name}.png", args.columns, args.thumb_size)

    write_html_index(selected_rows, crop_paths, args.output_dir / "index.html")
    print(f"Wrote {len(crop_paths)} review crops to {args.output_dir / 'crops'}")
    print(f"Contact sheets: {args.output_dir / 'contact_sheets'}")
    print(f"HTML index: {args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
