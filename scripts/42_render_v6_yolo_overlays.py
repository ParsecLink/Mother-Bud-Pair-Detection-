#!/usr/bin/env python
"""Render YOLO labels or predictions as box overlays for visual review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_CLASSES = V6_DIR / "classes.txt"
DEFAULT_IMAGES_DIR = V6_DIR / "datasets" / "yolo_v6" / "images" / "val"
DEFAULT_LABELS_DIR = V6_DIR / "datasets" / "yolo_v6" / "labels" / "val"
DEFAULT_OUTPUT_DIR = V6_DIR / "overlays" / "yolo_labels"
CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 210, 0),
    2: (255, 80, 220),
}


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = images_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def draw_yolo_file(image_path: Path, label_path: Path, output_path: Path, classes: list[str]) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    count = 0
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, box_w, box_h = [float(value) for value in parts[1:5]]
        score = float(parts[5]) if len(parts) >= 6 else None
        x1 = (cx - box_w / 2.0) * width
        y1 = (cy - box_h / 2.0) * height
        x2 = (cx + box_w / 2.0) * width
        y2 = (cy + box_h / 2.0) * height
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        class_name = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
        label = class_name if score is None else f"{class_name} {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0))
        count += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return count


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    classes = read_classes(args.classes)
    rendered = 0
    boxes = 0
    for label_path in sorted(args.labels_dir.glob("*.txt")):
        image_path = find_image(args.images_dir, label_path.stem)
        if image_path is None:
            print(f"Skipping {label_path}; no matching image found", file=sys.stderr)
            continue
        boxes += draw_yolo_file(image_path, label_path, args.output_dir / f"{label_path.stem}_overlay.png", classes)
        rendered += 1
    print(f"Rendered {rendered} overlays with {boxes} boxes to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
