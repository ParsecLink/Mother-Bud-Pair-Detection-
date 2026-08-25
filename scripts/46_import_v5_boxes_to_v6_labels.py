#!/usr/bin/env python
"""Convert v5 rule/SAM box tables into the v6 ML label CSV schema."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_OUTPUT = V6_DIR / "annotations" / "labels_from_v5.csv"
DEFAULT_V5_CANDIDATES = [
    PROJECT_ROOT / "rule_discovery" / "sam_v5_all_objects_trial" / "tables" / "draft_boxes_v5_all_objects_sam.csv",
    PROJECT_ROOT / "rule_discovery" / "sam_v5_all_objects_trial" / "boxes" / "draft_boxes_v5_all_objects_sam.csv",
    PROJECT_ROOT / "rule_discovery" / "rules_v5_all_objects_matching" / "tables" / "draft_boxes_v5_all_objects.csv",
    PROJECT_ROOT / "rule_discovery" / "rules_v5_all_objects_matching" / "boxes" / "draft_boxes_v5_all_objects.csv",
]
LABEL_COLUMNS = ["image_id", "class_name", "x1", "y1", "x2", "y2", "source", "review_status", "notes"]
ACCEPTED_CLASSES = {"single_cell", "mother_bud_pair", "early_bud_pair"}


def read_manifest(path: Path) -> dict[tuple[str, int], str]:
    """Map v5 condition/frame keys to v6 image IDs."""

    lookup: dict[tuple[str, int], str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            stem = Path(row["image_path"]).stem
            if "_frame_" not in stem:
                continue
            condition, frame_text = stem.rsplit("_frame_", 1)
            try:
                frame = int(frame_text)
            except ValueError:
                continue
            lookup[(condition, frame)] = image_id
            lookup[(condition.replace("_", " "), frame)] = image_id
    return lookup


def resolve_source_path(path_arg: Path | None) -> Path | None:
    if path_arg is not None:
        return path_arg if path_arg.exists() else None
    for path in DEFAULT_V5_CANDIDATES:
        if path.exists():
            return path
    return None


def parse_corners(value: str) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    try:
        corners = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not corners:
        return None
    ys = [float(point[0]) for point in corners]
    xs = [float(point[1]) for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def box_from_row(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    axis_cols = ("axis_aligned_x1", "axis_aligned_y1", "axis_aligned_x2", "axis_aligned_y2")
    if all(row.get(col, "") != "" for col in axis_cols):
        return tuple(float(row[col]) for col in axis_cols)  # type: ignore[return-value]
    if row.get("corner_yx_json"):
        return parse_corners(row["corner_yx_json"])
    if all(row.get(col, "") != "" for col in ("x1", "y1", "x2", "y2")):
        return float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
    return None


def class_from_row(row: dict[str, str]) -> str:
    return (row.get("source_class") or row.get("clean_class") or row.get("class_name") or "").strip()


def image_id_from_row(row: dict[str, str], manifest_lookup: dict[tuple[str, int], str]) -> str | None:
    if row.get("image_id"):
        return row["image_id"]
    if not row.get("condition") or not row.get("frame"):
        return None
    condition = row["condition"]
    try:
        frame = int(float(row["frame"]))
    except ValueError:
        return None
    return manifest_lookup.get((condition, frame)) or manifest_lookup.get((condition.replace("/", "_"), frame))


def convert_rows(source_path: Path, manifest_path: Path, review_status: str) -> list[dict[str, object]]:
    manifest_lookup = read_manifest(manifest_path)
    rows: list[dict[str, object]] = []
    skipped = 0
    with source_path.open("r", newline="", encoding="utf-8") as handle:
        for source_row in csv.DictReader(handle):
            class_name = class_from_row(source_row)
            if class_name not in ACCEPTED_CLASSES:
                skipped += 1
                continue
            image_id = image_id_from_row(source_row, manifest_lookup)
            box = box_from_row(source_row)
            if image_id is None or box is None:
                skipped += 1
                continue
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue
            rows.append(
                {
                    "image_id": image_id,
                    "class_name": class_name,
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "source": "v5_box_import",
                    "review_status": review_status,
                    "notes": f"source={source_path.relative_to(PROJECT_ROOT)}; candidate_id={source_row.get('candidate_id', '')}; skipped_before_write={skipped}",
                }
            )
    return rows


def write_labels(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None, help="Specific v5 box CSV to import.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-status", default="needs_review", choices=["needs_review", "reviewed"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    source_path = resolve_source_path(args.source)
    if source_path is None:
        print("No v5 box CSV found. Checked:")
        for path in DEFAULT_V5_CANDIDATES:
            print(f"  {path}")
        print("Run v5 box generation first, or pass --source path\\to\\draft_boxes.csv.")
        return 0
    rows = convert_rows(source_path, args.manifest, args.review_status)
    write_labels(rows, args.output)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["class_name"])] = counts.get(str(row["class_name"]), 0) + 1
    print(f"Imported {len(rows)} v5 boxes from {source_path}")
    print(f"Wrote v6 labels to {args.output}")
    for class_name in ("single_cell", "mother_bud_pair", "early_bud_pair"):
        print(f"  {class_name}: {counts.get(class_name, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
