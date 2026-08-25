#!/usr/bin/env python
"""Generate review-needed v6 pseudo-labels from merged RGB images.

This is a bootstrapper, not a source of ground truth. It uses visible GFP-like
green blobs as nuclei and mCherry-like magenta blobs as bud-neck candidates,
then assigns coarse object boxes for the three v6 detector classes.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_OUTPUT_DIR = V6_DIR / "pseudo_labels"
LABEL_COLUMNS = ["image_id", "class_name", "x1", "y1", "x2", "y2", "source", "review_status", "notes"]

PAIR_RADIUS = 38.0
EARLY_BUD_RADIUS = 30.0
PAIR_MAX_NUCLEUS_DISTANCE = 56.0
SINGLE_BOX_SIZE = 34.0
PAIR_PADDING = 18.0
EARLY_PADDING = 20.0


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def channel_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return green-nucleus and magenta-budneck masks from an RGB image."""

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
    """Find 8-connected components without external image-processing deps."""

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


def expanded_box(points: list[tuple[float, float]], image_width: int, image_height: int, padding: float) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1 = max(0.0, min(xs) - padding)
    y1 = max(0.0, min(ys) - padding)
    x2 = min(float(image_width - 1), max(xs) + padding)
    y2 = min(float(image_height - 1), max(ys) + padding)
    return x1, y1, x2, y2


def single_box(nucleus: dict[str, float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    half = SINGLE_BOX_SIZE / 2.0
    x = nucleus["centroid_x"]
    y = nucleus["centroid_y"]
    return (
        max(0.0, x - half),
        max(0.0, y - half),
        min(float(image_width - 1), x + half),
        min(float(image_height - 1), y + half),
    )


def assign_pseudo_labels(image_id: str, image_path: Path) -> list[dict[str, object]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    rgb = np.asarray(image)
    green_mask, magenta_mask = channel_masks(rgb)
    nuclei = connected_components(green_mask, min_area=8, max_area=900)
    budnecks = connected_components(magenta_mask, min_area=4, max_area=500)
    rows: list[dict[str, object]] = []
    used_nuclei: set[int] = set()
    used_budnecks: set[int] = set()

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
                points = [
                    (first[1]["centroid_x"], first[1]["centroid_y"]),
                    (second[1]["centroid_x"], second[1]["centroid_y"]),
                    (bud["centroid_x"], bud["centroid_y"]),
                ]
                x1, y1, x2, y2 = expanded_box(points, width, height, PAIR_PADDING)
                rows.append(
                    {
                        "image_id": image_id,
                        "class_name": "mother_bud_pair",
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "source": "v6_rgb_pseudo_label",
                        "review_status": "needs_review",
                        "notes": f"two_nuclei_near_budneck; budneck_idx={bud_idx}",
                    }
                )
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
        idx, nucleus, _ = nearby[0]
        points = [(nucleus["centroid_x"], nucleus["centroid_y"]), (bud["centroid_x"], bud["centroid_y"])]
        x1, y1, x2, y2 = expanded_box(points, width, height, EARLY_PADDING)
        rows.append(
            {
                "image_id": image_id,
                "class_name": "early_bud_pair",
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "source": "v6_rgb_pseudo_label",
                "review_status": "needs_review",
                "notes": f"one_nucleus_near_budneck; budneck_idx={bud_idx}",
            }
        )
        used_nuclei.add(idx)
        used_budnecks.add(bud_idx)

    for idx, nucleus in enumerate(nuclei):
        if idx in used_nuclei:
            continue
        x1, y1, x2, y2 = single_box(nucleus, width, height)
        rows.append(
            {
                "image_id": image_id,
                "class_name": "single_cell",
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "source": "v6_rgb_pseudo_label",
                "review_status": "needs_review",
                "notes": f"unpaired_green_nucleus; nucleus_idx={idx}",
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
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="labels_pseudo_rgb.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = read_manifest(args.manifest)
    rows: list[dict[str, object]] = []
    for meta in manifest:
        image_path = PROJECT_ROOT / meta["image_path"]
        rows.extend(assign_pseudo_labels(meta["image_id"], image_path))
    output_path = args.output_dir / args.output_name
    write_labels(rows, output_path)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["class_name"])] = counts.get(str(row["class_name"]), 0) + 1
    print(f"Wrote {len(rows)} pseudo-label boxes to {output_path}")
    for class_name in ("single_cell", "mother_bud_pair", "early_bud_pair"):
        print(f"  {class_name}: {counts.get(class_name, 0)}")
    print("All pseudo-label rows are marked review_status=needs_review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
