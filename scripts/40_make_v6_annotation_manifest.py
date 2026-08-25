#!/usr/bin/env python
"""Create the v6 image manifest and empty label CSV for ML detection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "result_pic" / "RGB"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "v6_ml_detection" / "annotations"
LABEL_COLUMNS = ["image_id", "class_name", "x1", "y1", "x2", "y2", "source", "review_status", "notes"]


def split_for_image(image_id: str, train_percent: int = 70, val_percent: int = 15) -> str:
    """Assign a deterministic split from the image ID."""

    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + val_percent:
        return "val"
    return "test"


def image_id_from_path(path: Path) -> str:
    """Return a stable image ID from the filename stem."""

    return path.stem.replace(" ", "_")


def iter_images(image_dir: Path) -> list[Path]:
    """Return supported images in a deterministic order."""

    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def write_manifest(image_dir: Path, output_path: Path) -> int:
    """Write image metadata used by the YOLO exporter."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in iter_images(image_dir):
        with Image.open(path) as image:
            width, height = image.size
        image_id = image_id_from_path(path)
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(path.relative_to(PROJECT_ROOT)),
                "width": width,
                "height": height,
                "split": split_for_image(image_id),
                "annotation_status": "unreviewed",
                "notes": "",
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "width", "height", "split", "annotation_status", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def ensure_label_template(output_path: Path, overwrite: bool = False) -> bool:
    """Create an empty object-label CSV unless a human-edited file exists."""

    if output_path.exists() and not overwrite:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite-labels", action="store_true", help="Replace labels.csv even if it already exists.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    image_dir = args.image_dir.resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "image_manifest.csv"
    labels_path = output_dir / "labels.csv"
    image_count = write_manifest(image_dir, manifest_path)
    labels_created = ensure_label_template(labels_path, overwrite=bool(args.overwrite_labels))

    print(f"Wrote {image_count} images to {manifest_path}")
    print(f"{'Created' if labels_created else 'Kept existing'} label file: {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
