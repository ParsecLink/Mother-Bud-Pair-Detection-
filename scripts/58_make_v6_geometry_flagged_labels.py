"""Add mother-bud line geometry flags to the current v6 pseudo-label CSV."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_LABELS = (
    V6_DIR / "pseudo_labels" / "labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv"
)
DEFAULT_GEOMETRY = V6_DIR / "reports" / "rule_geometry_mother_bud_pairs.csv"
DEFAULT_OUTPUT = (
    V6_DIR
    / "pseudo_labels"
    / "labels_pseudo_mobile_sam_from_source_tifs_mask_aware_geometry_flagged.csv"
)

LABEL_COLUMNS = ["image_id", "class_name", "x1", "y1", "x2", "y2", "source", "review_status", "notes"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_budneck_idx(notes: str) -> str | None:
    match = re.search(r"budneck_idx=(\d+)", notes)
    if not match:
        return None
    return match.group(1)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def geometry_quality(row: dict[str, str], max_tilt_deg: float, max_perp_px: float) -> str:
    between = str(row.get("between_nuclei", "")).lower() == "true"
    tilt = as_float(row, "tilt_angle_deg")
    perp = as_float(row, "perpendicular_distance")
    if between and tilt <= max_tilt_deg and perp <= max_perp_px:
        return "good_line_rule"
    return "review_line_rule"


def augment_labels(
    labels_path: Path,
    geometry_path: Path,
    output_path: Path,
    max_tilt_deg: float,
    max_perp_px: float,
) -> dict[str, int]:
    labels = read_csv(labels_path)
    geometry_rows = read_csv(geometry_path)
    geometry_by_key = {
        (row["image_id"], str(row["budneck_idx"])): row
        for row in geometry_rows
        if row.get("image_id") and row.get("budneck_idx")
    }

    counts: Counter[str] = Counter()
    output_rows: list[dict[str, str]] = []
    for row in labels:
        updated = {key: row.get(key, "") for key in LABEL_COLUMNS}
        if updated["class_name"] != "mother_bud_pair":
            output_rows.append(updated)
            continue

        budneck_idx = parse_budneck_idx(updated.get("notes", ""))
        geom = geometry_by_key.get((updated["image_id"], str(budneck_idx))) if budneck_idx is not None else None
        if not geom:
            updated["notes"] = f"{updated['notes']}; geometry_quality=missing_geometry"
            counts["missing_geometry"] += 1
            output_rows.append(updated)
            continue

        quality = geometry_quality(geom, max_tilt_deg=max_tilt_deg, max_perp_px=max_perp_px)
        counts[quality] += 1
        counts["between_nuclei"] += int(str(geom.get("between_nuclei", "")).lower() == "true")

        updated["notes"] = (
            f"{updated['notes']}; "
            f"geometry_quality={quality}; "
            f"between_nuclei={geom.get('between_nuclei', '')}; "
            f"projection_t={geom.get('projection_t', '')}; "
            f"perp_px={geom.get('perpendicular_distance', '')}; "
            f"tilt_deg={geom.get('tilt_angle_deg', '')}; "
            f"nucleus_distance_px={geom.get('nucleus_distance', '')}"
        )
        output_rows.append(updated)

    write_csv(output_path, output_rows)
    counts["total_rows"] = len(output_rows)
    counts["mother_bud_pair"] = sum(1 for row in output_rows if row["class_name"] == "mother_bud_pair")
    return dict(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tilt-deg", type=float, default=30.0)
    parser.add_argument("--max-perp-px", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = augment_labels(
        labels_path=args.labels,
        geometry_path=args.geometry,
        output_path=args.output,
        max_tilt_deg=args.max_tilt_deg,
        max_perp_px=args.max_perp_px,
    )
    print(f"Wrote geometry-flagged labels to {args.output}")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
