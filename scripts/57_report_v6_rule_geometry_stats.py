"""Report geometry statistics for the current v6 green/red biological rules."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_REPORT = V6_DIR / "reports" / "rule_geometry_stats.json"
DEFAULT_MOTHER_CSV = V6_DIR / "reports" / "rule_geometry_mother_bud_pairs.csv"
DEFAULT_IMAGE_CSV = V6_DIR / "reports" / "rule_geometry_image_counts.csv"

PAIR_RADIUS = 38.0
EARLY_BUD_RADIUS = 30.0
PAIR_MAX_NUCLEUS_DISTANCE = 56.0


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def channel_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = rgb.astype(np.float32)
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    intensity = arr.mean(axis=2)
    green_score = green - np.maximum(red, blue)
    magenta_score = np.minimum(red, blue) - green

    green_cut = max(35.0, float(np.percentile(green_score, 99.2)))
    magenta_cut = max(28.0, float(np.percentile(magenta_score, 98.8)))
    green_mask = (green_score >= green_cut) & (green >= 80.0) & (intensity >= 35.0)
    magenta_mask = (magenta_score >= magenta_cut) & (red >= 95.0) & (blue >= 80.0)
    return green_mask, magenta_mask


def connected_components(mask: np.ndarray, min_area: int, max_area: int) -> list[dict[str, float]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    objects: list[dict[str, float]] = []
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        coords: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            coords.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        area = len(coords)
        if area < min_area or area > max_area:
            continue
        ys = np.asarray([point[0] for point in coords], dtype=np.float32)
        xs = np.asarray([point[1] for point in coords], dtype=np.float32)
        objects.append(
            {
                "centroid_y": float(ys.mean()),
                "centroid_x": float(xs.mean()),
                "area": float(area),
                "x1": float(xs.min()),
                "y1": float(ys.min()),
                "x2": float(xs.max() + 1.0),
                "y2": float(ys.max() + 1.0),
            }
        )
    return objects


def distance(a: dict[str, float], b: dict[str, float]) -> float:
    return float(math.hypot(a["centroid_y"] - b["centroid_y"], a["centroid_x"] - b["centroid_x"]))


def condition_from_image_id(image_id: str) -> str:
    if "_frame_" not in image_id:
        return image_id
    return image_id.rsplit("_frame_", 1)[0]


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p05": None, "p10": None, "median": None, "p90": None, "p95": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(arr.min()), 3),
        "p05": round(float(np.percentile(arr, 5)), 3),
        "p10": round(float(np.percentile(arr, 10)), 3),
        "median": round(float(np.percentile(arr, 50)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "max": round(float(arr.max()), 3),
        "mean": round(float(arr.mean()), 3),
    }


def line_geometry(n1: dict[str, float], n2: dict[str, float], bud: dict[str, float]) -> dict[str, float | bool]:
    x1, y1 = n1["centroid_x"], n1["centroid_y"]
    x2, y2 = n2["centroid_x"], n2["centroid_y"]
    bx, by = bud["centroid_x"], bud["centroid_y"]
    vx = x2 - x1
    vy = y2 - y1
    wx = bx - x1
    wy = by - y1
    segment_sq = vx * vx + vy * vy
    segment = math.sqrt(segment_sq)
    if segment <= 1e-9:
        return {
            "projection_t": 0.0,
            "between_nuclei": False,
            "perpendicular_distance": float("nan"),
            "normalized_perpendicular_distance": float("nan"),
            "midpoint_distance": float("nan"),
            "midpoint_offset_parallel": float("nan"),
            "line_deviation_deg": float("nan"),
        }

    projection_t = (wx * vx + wy * vy) / segment_sq
    proj_x = x1 + projection_t * vx
    proj_y = y1 + projection_t * vy
    perpendicular = math.hypot(bx - proj_x, by - proj_y)
    midpoint_x = (x1 + x2) / 2.0
    midpoint_y = (y1 + y2) / 2.0
    midpoint_distance = math.hypot(bx - midpoint_x, by - midpoint_y)
    midpoint_offset_parallel = abs(projection_t - 0.5) * segment

    d1 = distance(n1, bud)
    d2 = distance(n2, bud)
    if d1 <= 1e-9 or d2 <= 1e-9:
        line_deviation_deg = 0.0
    else:
        cos_angle = ((x1 - bx) * (x2 - bx) + (y1 - by) * (y2 - by)) / (d1 * d2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_at_bud = math.degrees(math.acos(cos_angle))
        line_deviation_deg = abs(180.0 - angle_at_bud)

    return {
        "projection_t": float(projection_t),
        "between_nuclei": bool(0.0 <= projection_t <= 1.0),
        "perpendicular_distance": float(perpendicular),
        "normalized_perpendicular_distance": float(perpendicular / segment),
        "midpoint_distance": float(midpoint_distance),
        "midpoint_offset_parallel": float(midpoint_offset_parallel),
        "line_deviation_deg": float(line_deviation_deg),
    }


def assign_with_stats(image_id: str, image_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    green_mask, magenta_mask = channel_masks(rgb)
    nuclei = connected_components(green_mask, min_area=8, max_area=900)
    budnecks = connected_components(magenta_mask, min_area=4, max_area=500)

    used_nuclei: set[int] = set()
    used_budnecks: set[int] = set()
    mother_rows: list[dict[str, object]] = []
    class_counts = Counter()

    for bud_idx, bud in sorted(enumerate(budnecks), key=lambda item: item[1]["area"], reverse=True):
        nearby = sorted(
            [(idx, nucleus, distance(nucleus, bud)) for idx, nucleus in enumerate(nuclei) if idx not in used_nuclei],
            key=lambda item: item[2],
        )
        pair_options = [item for item in nearby if item[2] <= PAIR_RADIUS]
        if len(pair_options) >= 2:
            first = pair_options[0]
            second = pair_options[1]
            nucleus_distance = distance(first[1], second[1])
            if nucleus_distance <= PAIR_MAX_NUCLEUS_DISTANCE:
                geom = line_geometry(first[1], second[1], bud)
                d1 = distance(first[1], bud)
                d2 = distance(second[1], bud)
                mother_rows.append(
                    {
                        "image_id": image_id,
                        "condition": condition_from_image_id(image_id),
                        "budneck_idx": bud_idx,
                        "nucleus_1_idx": first[0],
                        "nucleus_2_idx": second[0],
                        "nucleus_distance": round(nucleus_distance, 3),
                        "bud_to_nucleus_1": round(d1, 3),
                        "bud_to_nucleus_2": round(d2, 3),
                        "bud_to_nearest_nucleus": round(min(d1, d2), 3),
                        "bud_to_farthest_nucleus": round(max(d1, d2), 3),
                        "projection_t": round(float(geom["projection_t"]), 3),
                        "between_nuclei": geom["between_nuclei"],
                        "perpendicular_distance": round(float(geom["perpendicular_distance"]), 3),
                        "normalized_perpendicular_distance": round(float(geom["normalized_perpendicular_distance"]), 3),
                        "midpoint_distance": round(float(geom["midpoint_distance"]), 3),
                        "midpoint_offset_parallel": round(float(geom["midpoint_offset_parallel"]), 3),
                        "line_deviation_deg": round(float(geom["line_deviation_deg"]), 3),
                        "nucleus_1_x": round(float(first[1]["centroid_x"]), 3),
                        "nucleus_1_y": round(float(first[1]["centroid_y"]), 3),
                        "nucleus_2_x": round(float(second[1]["centroid_x"]), 3),
                        "nucleus_2_y": round(float(second[1]["centroid_y"]), 3),
                        "bud_x": round(float(bud["centroid_x"]), 3),
                        "bud_y": round(float(bud["centroid_y"]), 3),
                    }
                )
                class_counts["mother_bud_pair"] += 1
                used_nuclei.update({first[0], second[0]})
                used_budnecks.add(bud_idx)

    for bud_idx, bud in sorted(enumerate(budnecks), key=lambda item: item[1]["area"], reverse=True):
        if bud_idx in used_budnecks:
            continue
        nearby = sorted(
            [(idx, nucleus, distance(nucleus, bud)) for idx, nucleus in enumerate(nuclei) if idx not in used_nuclei],
            key=lambda item: item[2],
        )
        if not nearby or nearby[0][2] > EARLY_BUD_RADIUS:
            continue
        idx, _nucleus, _ = nearby[0]
        used_nuclei.add(idx)
        used_budnecks.add(bud_idx)
        class_counts["early_bud_pair"] += 1

    for idx, _nucleus in enumerate(nuclei):
        if idx not in used_nuclei:
            class_counts["single_cell"] += 1

    image_stats = {
        "image_id": image_id,
        "condition": condition_from_image_id(image_id),
        "nucleus_count": len(nuclei),
        "budneck_count": len(budnecks),
        "mother_bud_pair": class_counts["mother_bud_pair"],
        "early_bud_pair": class_counts["early_bud_pair"],
        "single_cell": class_counts["single_cell"],
    }
    return mother_rows, image_stats


def summarize_by_condition(mother_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in mother_rows:
        grouped[str(row["condition"])].append(row)
    summary: dict[str, dict[str, object]] = {}
    for condition, rows in sorted(grouped.items()):
        summary[condition] = {
            "mother_bud_pair_count": len(rows),
            "nucleus_distance": quantiles([float(row["nucleus_distance"]) for row in rows]),
            "perpendicular_distance": quantiles([float(row["perpendicular_distance"]) for row in rows]),
            "projection_t": quantiles([float(row["projection_t"]) for row in rows]),
            "line_deviation_deg": quantiles([float(row["line_deviation_deg"]) for row in rows]),
            "between_nuclei_fraction": round(sum(1 for row in rows if row["between_nuclei"]) / len(rows), 4) if rows else None,
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(manifest_path: Path, report_path: Path, mother_csv_path: Path, image_csv_path: Path) -> dict[str, object]:
    manifest = read_manifest(manifest_path)
    try:
        manifest_display = str(manifest_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        manifest_display = str(manifest_path)
    all_mother_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    for meta in manifest:
        mother_rows, image_stats = assign_with_stats(meta["image_id"], PROJECT_ROOT / meta["image_path"])
        all_mother_rows.extend(mother_rows)
        image_rows.append(image_stats)

    label_counts = Counter()
    for row in image_rows:
        label_counts["mother_bud_pair"] += int(row["mother_bud_pair"])
        label_counts["early_bud_pair"] += int(row["early_bud_pair"])
        label_counts["single_cell"] += int(row["single_cell"])

    values = {
        "nucleus_distance": [float(row["nucleus_distance"]) for row in all_mother_rows],
        "bud_to_nearest_nucleus": [float(row["bud_to_nearest_nucleus"]) for row in all_mother_rows],
        "bud_to_farthest_nucleus": [float(row["bud_to_farthest_nucleus"]) for row in all_mother_rows],
        "projection_t": [float(row["projection_t"]) for row in all_mother_rows],
        "perpendicular_distance": [float(row["perpendicular_distance"]) for row in all_mother_rows],
        "normalized_perpendicular_distance": [float(row["normalized_perpendicular_distance"]) for row in all_mother_rows],
        "midpoint_distance": [float(row["midpoint_distance"]) for row in all_mother_rows],
        "midpoint_offset_parallel": [float(row["midpoint_offset_parallel"]) for row in all_mother_rows],
        "line_deviation_deg": [float(row["line_deviation_deg"]) for row in all_mother_rows],
    }

    between_count = sum(1 for row in all_mother_rows if row["between_nuclei"])
    mother_count = len(all_mother_rows)
    candidate_rule_effects = {}
    for max_perp in (6, 8, 10, 12, 15):
        for t_min, t_max in ((0.0, 1.0), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8)):
            kept = [
                row
                for row in all_mother_rows
                if t_min <= float(row["projection_t"]) <= t_max and float(row["perpendicular_distance"]) <= max_perp
            ]
            candidate_rule_effects[f"t_{t_min:.2f}_{t_max:.2f}_perp_le_{max_perp}"] = {
                "kept": len(kept),
                "kept_fraction_of_current_mother_pairs": round(len(kept) / mother_count, 4) if mother_count else None,
            }

    report = {
        "manifest_path": manifest_display,
        "image_count": len(manifest),
        "current_rule_thresholds": {
            "PAIR_RADIUS_bud_to_each_nucleus_px": PAIR_RADIUS,
            "PAIR_MAX_NUCLEUS_DISTANCE_px": PAIR_MAX_NUCLEUS_DISTANCE,
            "EARLY_BUD_RADIUS_bud_to_one_nucleus_px": EARLY_BUD_RADIUS,
        },
        "detected_signal_counts_per_image": {
            "nucleus_count": quantiles([float(row["nucleus_count"]) for row in image_rows]),
            "budneck_count": quantiles([float(row["budneck_count"]) for row in image_rows]),
        },
        "rule_label_counts": dict(label_counts),
        "mother_bud_pair_geometry": {
            key: quantiles(val) for key, val in values.items()
        },
        "mother_bud_pair_between_nuclei": {
            "count": between_count,
            "total": mother_count,
            "fraction": round(between_count / mother_count, 4) if mother_count else None,
        },
        "candidate_line_rule_effects_on_current_mother_pairs": candidate_rule_effects,
        "by_condition": summarize_by_condition(all_mother_rows),
        "notes": [
            "These are descriptive statistics for current pseudo-label candidates, not ground-truth optimized thresholds.",
            "Use human-reviewed audit labels to choose final thresholds.",
            "projection_t is bud-neck projection on the nucleus-nucleus segment: 0 means at nucleus 1, 1 means at nucleus 2, 0.5 means midpoint.",
            "line_deviation_deg is 0 when the bud-neck lies exactly on the nucleus-nucleus line segment.",
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(mother_csv_path, all_mother_rows)
    write_csv(image_csv_path, image_rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mother-csv", type=Path, default=DEFAULT_MOTHER_CSV)
    parser.add_argument("--image-csv", type=Path, default=DEFAULT_IMAGE_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.manifest, args.report, args.mother_csv, args.image_csv)
    print(json.dumps(report, indent=2))
    print(f"Wrote report to {args.report}")
    print(f"Wrote mother-bud geometry rows to {args.mother_csv}")
    print(f"Wrote image-level rows to {args.image_csv}")


if __name__ == "__main__":
    main()
