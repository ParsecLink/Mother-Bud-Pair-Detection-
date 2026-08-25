#!/usr/bin/env python
"""Export reviewed v6 labels to a YOLO object-detection dataset."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_LABELS = V6_DIR / "annotations" / "labels.csv"
DEFAULT_CLASSES = V6_DIR / "classes.txt"
DEFAULT_OUTPUT_DIR = V6_DIR / "datasets" / "yolo_v6"
REQUIRED_LABEL_COLUMNS = {"image_id", "class_name", "x1", "y1", "x2", "y2"}


def read_classes(path: Path) -> list[str]:
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {path}")
    return classes


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def read_labels(path: Path, reviewed_only: bool) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_LABEL_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if reviewed_only and "review_status" in (rows[0].keys() if rows else []):
        rows = [row for row in rows if row.get("review_status", "").strip().lower() == "reviewed"]
    return [row for row in rows if row.get("class_name", "").strip()]


def clamp_box(row: dict[str, str], width: float, height: float) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(width - 1.0, float(row["x1"])))
    y1 = max(0.0, min(height - 1.0, float(row["y1"])))
    x2 = max(0.0, min(width - 1.0, float(row["x2"])))
    y2 = max(0.0, min(height - 1.0, float(row["y2"])))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box for image_id={row.get('image_id')}: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def yolo_line(row: dict[str, str], class_to_id: dict[str, int], width: float, height: float) -> str:
    class_name = row["class_name"].strip()
    if class_name not in class_to_id:
        raise ValueError(f"Unknown class {class_name!r}; expected one of {sorted(class_to_id)}")
    x1, y1, x2, y2 = clamp_box(row, width, height)
    box_width = x2 - x1
    box_height = y2 - y1
    cx = x1 + box_width / 2.0
    cy = y1 + box_height / 2.0
    return f"{class_to_id[class_name]} {cx / width:.6f} {cy / height:.6f} {box_width / width:.6f} {box_height / height:.6f}"


def copy_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_data_yaml(output_dir: Path, classes: list[str]) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    content = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-unreviewed", action="store_true", help="Export labels even when review_status is not reviewed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    classes = read_classes(args.classes)
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    manifest = read_manifest(args.manifest)
    labels = read_labels(args.labels, reviewed_only=not args.include_unreviewed)
    labels_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in labels:
        labels_by_image[row["image_id"]].append(row)

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    exported_images = 0
    exported_boxes = 0
    for image_id, meta in manifest.items():
        split = meta.get("split", "train").strip() or "train"
        image_path = PROJECT_ROOT / meta["image_path"]
        target_image = output_dir / "images" / split / image_path.name
        target_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
        copy_image(image_path, target_image)
        width = float(meta["width"])
        height = float(meta["height"])
        lines = [yolo_line(row, class_to_id, width, height) for row in labels_by_image.get(image_id, [])]
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        exported_images += 1
        exported_boxes += len(lines)

    write_data_yaml(output_dir, classes)
    print(f"Exported {exported_images} images and {exported_boxes} boxes to {output_dir}")
    print(f"YOLO data config: {output_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
