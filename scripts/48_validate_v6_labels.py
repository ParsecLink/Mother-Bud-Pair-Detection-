#!/usr/bin/env python
"""Validate v6 label CSVs and write a compact summary report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_LABELS = V6_DIR / "annotations" / "labels.csv"
DEFAULT_CLASSES = V6_DIR / "classes.txt"
DEFAULT_REPORT = V6_DIR / "reports" / "label_validation_summary.json"
REQUIRED_COLUMNS = {"image_id", "class_name", "x1", "y1", "x2", "y2"}


def read_classes(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing {key}")
    return float(value)


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def validate_labels(labels_path: Path, manifest_path: Path, classes_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    classes = read_classes(classes_path)
    manifest = read_manifest(manifest_path)
    errors: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()

    with labels_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            errors.append({"row_number": 0, "error": f"missing_columns: {sorted(missing)}"})
            rows: list[dict[str, str]] = []
        else:
            rows = list(reader)

    for index, row in enumerate(rows, start=2):
        image_id = row.get("image_id", "")
        class_name = row.get("class_name", "")
        if image_id not in manifest:
            errors.append({"row_number": index, "image_id": image_id, "error": "unknown_image_id"})
            continue
        if class_name not in classes:
            errors.append({"row_number": index, "image_id": image_id, "error": f"unknown_class: {class_name}"})
            continue
        try:
            x1 = parse_float(row, "x1")
            y1 = parse_float(row, "y1")
            x2 = parse_float(row, "x2")
            y2 = parse_float(row, "y2")
        except ValueError as exc:
            errors.append({"row_number": index, "image_id": image_id, "error": str(exc)})
            continue
        width = float(manifest[image_id]["width"])
        height = float(manifest[image_id]["height"])
        if x2 <= x1 or y2 <= y1:
            errors.append({"row_number": index, "image_id": image_id, "error": "non_positive_box_area"})
            continue
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            errors.append(
                {
                    "row_number": index,
                    "image_id": image_id,
                    "error": "box_outside_image",
                    "box": [x1, y1, x2, y2],
                    "image_size": [width, height],
                }
            )
            continue
        class_counts[class_name] += 1
        split_counts[manifest[image_id].get("split", "train")] += 1
        review_counts[row.get("review_status", "") or "blank"] += 1
        image_counts[image_id] += 1

    images_with_labels = len(image_counts)
    empty_images = len(manifest) - images_with_labels
    summary: dict[str, object] = {
        "labels_path": portable_path(labels_path),
        "manifest_path": portable_path(manifest_path),
        "total_rows": len(rows),
        "valid_rows": sum(class_counts.values()),
        "error_count": len(errors),
        "images_in_manifest": len(manifest),
        "images_with_labels": images_with_labels,
        "images_without_labels": empty_images,
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "review_status_counts": dict(review_counts),
        "top_images_by_box_count": image_counts.most_common(12),
    }
    return summary, errors


def write_report(summary: dict[str, object], errors: list[dict[str, object]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "errors": errors[:200]}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary, errors = validate_labels(args.labels, args.manifest, args.classes)
    write_report(summary, errors, args.report)
    print(json.dumps(summary, indent=2))
    print(f"Wrote validation report to {args.report}")
    if errors:
        print(f"First error: {errors[0]}")
    if errors and args.fail_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
