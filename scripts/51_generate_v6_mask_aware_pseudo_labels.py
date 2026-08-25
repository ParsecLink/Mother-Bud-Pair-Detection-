#!/usr/bin/env python
"""Generate v6 pseudo-labels using cell masks to size boxes.

The class decisions intentionally keep the RGB fluorescence rules from
44_generate_v6_pseudo_labels_from_rgb.py. This script changes the box geometry:
it first tries to find whole-cell masks from an external mask directory, a
transmitted/DIC-like image, or an RGB-derived transmitted proxy. Each green
nucleus and magenta bud-neck candidate is matched to those masks, and label
boxes are built from the union of the matched masks. If mask matching fails for
an object, the original dot-based box is used as a fallback.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageSequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_OUTPUT_DIR = V6_DIR / "pseudo_labels"
LABEL_COLUMNS = ["image_id", "class_name", "x1", "y1", "x2", "y2", "source", "review_status", "notes"]
DIAGNOSTIC_COLUMNS = [
    "image_id",
    "mask_source",
    "raw_mask_count",
    "seeded_mask_count",
    "nucleus_count",
    "budneck_count",
    "mask_union_labels",
    "fallback_dot_labels",
]

PAIR_RADIUS = 38.0
EARLY_BUD_RADIUS = 30.0
PAIR_MAX_NUCLEUS_DISTANCE = 56.0
SINGLE_BOX_SIZE = 34.0
PAIR_PADDING = 18.0
EARLY_PADDING = 20.0
MASK_BOX_PADDING = 4.0
MASK_MATCH_RADIUS = 7
MIN_CELL_MASK_AREA = 55
MAX_CELL_MASK_AREA = 14000
MIN_SEEDED_MASK_AREA = 18


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile_normalize(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(image, low))
    hi = float(np.percentile(image, high))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def gaussian_blur01(image: np.ndarray, radius: float) -> np.ndarray:
    image_u8 = np.clip(np.asarray(image, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    blurred = Image.fromarray(image_u8).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def disk_offsets(radius: int) -> list[tuple[int, int]]:
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= radius * radius
    ]


def binary_morphology(mask: np.ndarray, radius: int, op: str) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    padded = np.pad(mask, radius, constant_values=(op == "erosion"))
    output = np.ones_like(mask, dtype=bool) if op == "erosion" else np.zeros_like(mask, dtype=bool)
    for dy, dx in disk_offsets(radius):
        view = padded[radius + dy : radius + dy + height, radius + dx : radius + dx + width]
        if op == "erosion":
            output &= view
        else:
            output |= view
    return output


def binary_closing(mask: np.ndarray, radius: int) -> np.ndarray:
    return binary_morphology(binary_morphology(mask, radius, "dilation"), radius, "erosion")


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    background = ~mask
    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        for y in (0, height - 1):
            if background[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if background[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if background[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    return mask | (background & ~visited)


def connected_components(mask: np.ndarray, min_area: int, max_area: int) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    objects: list[dict[str, Any]] = []
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
        ys = np.asarray([point[0] for point in coords], dtype=np.int32)
        xs = np.asarray([point[1] for point in coords], dtype=np.int32)
        component_mask = np.zeros(mask.shape, dtype=bool)
        component_mask[ys, xs] = True
        objects.append(
            {
                "centroid_y": float(ys.mean()),
                "centroid_x": float(xs.mean()),
                "area": float(area),
                "x1": float(xs.min()),
                "y1": float(ys.min()),
                "x2": float(xs.max() + 1.0),
                "y2": float(ys.max() + 1.0),
                "mask": component_mask,
            }
        )
    return objects


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


def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return float(math.hypot(a["centroid_y"] - b["centroid_y"], a["centroid_x"] - b["centroid_x"]))


def expanded_box(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1 = max(0.0, min(xs) - padding)
    y1 = max(0.0, min(ys) - padding)
    x2 = min(float(image_width - 1), max(xs) + padding)
    y2 = min(float(image_height - 1), max(ys) + padding)
    return x1, y1, x2, y2


def single_box(nucleus: dict[str, Any], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    half = SINGLE_BOX_SIZE / 2.0
    x = float(nucleus["centroid_x"])
    y = float(nucleus["centroid_y"])
    return (
        max(0.0, x - half),
        max(0.0, y - half),
        min(float(image_width - 1), x + half),
        min(float(image_height - 1), y + half),
    )


def clamp_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(image_width - 1), x1)),
        max(0.0, min(float(image_height - 1), y1)),
        max(0.0, min(float(image_width - 1), x2)),
        max(0.0, min(float(image_height - 1), y2)),
    )


def image_stem(meta: dict[str, str]) -> str:
    return Path(meta["image_path"]).stem


def parse_condition_frame(meta: dict[str, str]) -> tuple[str | None, int | None]:
    stem = image_stem(meta)
    if "_frame_" not in stem:
        return None, None
    condition, frame_text = stem.rsplit("_frame_", 1)
    try:
        return condition, int(frame_text)
    except ValueError:
        return condition, None


def candidate_named_paths(directory: Path, meta: dict[str, str]) -> list[Path]:
    stem = image_stem(meta)
    names = {stem, meta["image_id"], meta["image_id"].replace("_", " ")}
    suffixes = [".png", ".tif", ".tiff", ".jpg", ".jpeg"]
    return [directory / f"{name}{suffix}" for name in names for suffix in suffixes]


def read_grayscale_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def read_tiff_frame(path: Path, frame_index: int) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            for idx, frame in enumerate(ImageSequence.Iterator(image)):
                if idx == frame_index:
                    return np.asarray(frame.convert("L"), dtype=np.float32)
    except OSError:
        return None
    return None


def load_transmitted_image(meta: dict[str, str], rgb: np.ndarray, trans_dir: Path | None) -> tuple[np.ndarray, str]:
    search_dirs: list[Path] = []
    if trans_dir is not None:
        search_dirs.append(trans_dir)
    search_dirs.extend(
        [
            PROJECT_ROOT / "result_pic" / "trans_only",
            PROJECT_ROOT / "result_pic" / "Trans",
            PROJECT_ROOT / "pic" / "trans_only",
            PROJECT_ROOT / "pic" / "Trans",
        ]
    )
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in candidate_named_paths(directory, meta):
            if path.exists():
                return read_grayscale_image(path), str(path.relative_to(PROJECT_ROOT))

    condition, frame_index = parse_condition_frame(meta)
    if condition is not None and frame_index is not None:
        for directory in [PROJECT_ROOT / "result_pic" / "projected_tifs", PROJECT_ROOT / "pic" / "projected_tifs"]:
            path = directory / f"{condition}_Trans.tif"
            if path.exists():
                frame = read_tiff_frame(path, frame_index)
                if frame is not None:
                    return frame, str(path.relative_to(PROJECT_ROOT))

    proxy = np.asarray(rgb, dtype=np.float32).min(axis=2)
    return proxy, "rgb_min_channel_transmitted_proxy"


def load_external_cell_masks(meta: dict[str, str], mask_dir: Path | None, shape: tuple[int, int]) -> tuple[list[dict[str, Any]], str | None]:
    if mask_dir is None or not mask_dir.exists():
        return [], None
    mask_dir = mask_dir.resolve()
    for path in candidate_named_paths(mask_dir, meta):
        if not path.exists():
            continue
        path = path.resolve()
        mask_image = np.asarray(Image.open(path).convert("L"))
        if mask_image.shape != shape:
            mask_image = np.asarray(Image.fromarray(mask_image).resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST))
        values = [int(value) for value in np.unique(mask_image) if int(value) > 0]
        masks: list[dict[str, Any]] = []
        if 1 < len(values) < 512:
            for value in values:
                masks.extend(connected_components(mask_image == value, MIN_CELL_MASK_AREA, MAX_CELL_MASK_AREA))
        else:
            masks = connected_components(mask_image > 0, MIN_CELL_MASK_AREA, MAX_CELL_MASK_AREA)
        return masks, str(path.relative_to(PROJECT_ROOT))
    return [], None


def segment_cell_masks_from_transmitted(transmitted: np.ndarray) -> list[dict[str, Any]]:
    base = percentile_normalize(transmitted, low=1.0, high=99.0)
    local = gaussian_blur01(base, radius=1.0)
    background = gaussian_blur01(base, radius=7.0)
    feature = np.abs(local - background)
    threshold = max(float(np.percentile(feature, 90.0)), float(feature.mean() + 1.2 * feature.std()))
    edge_mask = feature >= threshold
    edge_mask = binary_closing(edge_mask, radius=2)
    body_mask = binary_morphology(edge_mask, radius=3, op="dilation")
    body_mask = fill_holes(body_mask)
    return connected_components(body_mask, MIN_CELL_MASK_AREA, MAX_CELL_MASK_AREA)


def point_near_mask(y: float, x: float, mask: np.ndarray, radius: int = MASK_MATCH_RADIUS) -> bool:
    height, width = mask.shape
    cy = int(round(y))
    cx = int(round(x))
    if 0 <= cy < height and 0 <= cx < width and mask[cy, cx]:
        return True
    y1 = max(0, cy - radius)
    y2 = min(height, cy + radius + 1)
    x1 = max(0, cx - radius)
    x2 = min(width, cx + radius + 1)
    return bool(mask[y1:y2, x1:x2].any())


def split_mask_by_seeds(mask: np.ndarray, seeds: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if len(seeds) == 1:
        return {str(seeds[0]["seed_key"]): mask}

    coords = np.argwhere(mask)
    if coords.size == 0:
        return {}
    seed_coords = np.asarray([[seed["centroid_y"], seed["centroid_x"]] for seed in seeds], dtype=np.float32)
    deltas = coords[:, None, :].astype(np.float32) - seed_coords[None, :, :]
    assignments = np.argmin(np.sum(deltas * deltas, axis=2), axis=1)
    split: dict[str, np.ndarray] = {}
    for seed_index, seed in enumerate(seeds):
        selected = coords[assignments == seed_index]
        if selected.shape[0] < MIN_SEEDED_MASK_AREA:
            continue
        submask = np.zeros_like(mask, dtype=bool)
        submask[selected[:, 0], selected[:, 1]] = True
        split[str(seed["seed_key"])] = submask
    return split


def build_seeded_mask_lookup(
    raw_masks: list[dict[str, Any]],
    nuclei: list[dict[str, Any]],
    budnecks: list[dict[str, Any]],
) -> tuple[dict[str, list[np.ndarray]], int]:
    seeds: list[dict[str, Any]] = []
    for idx, nucleus in enumerate(nuclei):
        seeds.append({"seed_key": f"n{idx}", **nucleus})
    for idx, budneck in enumerate(budnecks):
        seeds.append({"seed_key": f"b{idx}", **budneck})

    lookup: dict[str, list[np.ndarray]] = {}
    seeded_count = 0
    for raw_mask in raw_masks:
        mask = raw_mask["mask"]
        matched = [
            seed
            for seed in seeds
            if point_near_mask(float(seed["centroid_y"]), float(seed["centroid_x"]), mask)
        ]
        if not matched:
            continue
        split = split_mask_by_seeds(mask, matched)
        for seed_key, submask in split.items():
            lookup.setdefault(seed_key, []).append(submask)
            seeded_count += 1
    return lookup, seeded_count


def mask_union_box(
    seed_keys: list[str],
    seed_lookup: dict[str, list[np.ndarray]],
    signal_points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float, float, float] | None, str]:
    masks: list[np.ndarray] = []
    for key in seed_keys:
        masks.extend(seed_lookup.get(key, []))
    if not masks:
        return None, "no_matched_mask"

    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask
    if int(union.sum()) < MIN_SEEDED_MASK_AREA:
        return None, "matched_mask_too_small"

    ys, xs = np.nonzero(union)
    x1 = float(xs.min()) - MASK_BOX_PADDING
    y1 = float(ys.min()) - MASK_BOX_PADDING
    x2 = float(xs.max() + 1) + MASK_BOX_PADDING
    y2 = float(ys.max() + 1) + MASK_BOX_PADDING
    x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), image_width, image_height)
    if x2 <= x1 or y2 <= y1:
        return None, "matched_mask_non_positive_box"

    for point_x, point_y in signal_points:
        if point_x < x1 - 2.0 or point_x > x2 + 2.0 or point_y < y1 - 2.0 or point_y > y2 + 2.0:
            return None, "matched_mask_misses_signal"
    return (x1, y1, x2, y2), f"mask_union;seeds={'+'.join(seed_keys)}"


def row_from_box(
    image_id: str,
    class_name: str,
    box: tuple[float, float, float, float],
    source: str,
    notes: str,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    return {
        "image_id": image_id,
        "class_name": class_name,
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "source": source,
        "review_status": "needs_review",
        "notes": notes,
    }


def choose_mask_or_fallback(
    seed_keys: list[str],
    seed_lookup: dict[str, list[np.ndarray]],
    signal_points: list[tuple[float, float]],
    fallback_box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float, float, float], str]:
    mask_box, reason = mask_union_box(seed_keys, seed_lookup, signal_points, image_width, image_height)
    if mask_box is not None:
        return mask_box, reason
    return fallback_box, f"fallback_dot;mask_reason={reason}"


def assign_pseudo_labels(
    meta: dict[str, str],
    mask_dir: Path | None,
    trans_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_id = meta["image_id"]
    image_path = PROJECT_ROOT / meta["image_path"]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    rgb = np.asarray(image)

    green_mask, magenta_mask = channel_masks(rgb)
    nuclei = connected_components(green_mask, min_area=8, max_area=900)
    budnecks = connected_components(magenta_mask, min_area=4, max_area=500)

    raw_masks, mask_source = load_external_cell_masks(meta, mask_dir, shape=(height, width))
    if not raw_masks:
        transmitted, mask_source = load_transmitted_image(meta, rgb, trans_dir)
        raw_masks = segment_cell_masks_from_transmitted(transmitted)
    if mask_source is None:
        mask_source = "none"

    seed_lookup, seeded_mask_count = build_seeded_mask_lookup(raw_masks, nuclei, budnecks)

    rows: list[dict[str, Any]] = []
    used_nuclei: set[int] = set()
    used_budnecks: set[int] = set()
    mask_union_labels = 0
    fallback_dot_labels = 0

    for bud_idx, bud in sorted(enumerate(budnecks), key=lambda item: item[1]["area"], reverse=True):
        nearby = sorted(
            [(idx, nucleus, distance(nucleus, bud)) for idx, nucleus in enumerate(nuclei) if idx not in used_nuclei],
            key=lambda item: item[2],
        )
        pair_options = [item for item in nearby if item[2] <= PAIR_RADIUS]
        if len(pair_options) < 2:
            continue
        first = pair_options[0]
        second = pair_options[1]
        nucleus_distance = distance(first[1], second[1])
        if nucleus_distance > PAIR_MAX_NUCLEUS_DISTANCE:
            continue

        signal_points = [
            (first[1]["centroid_x"], first[1]["centroid_y"]),
            (second[1]["centroid_x"], second[1]["centroid_y"]),
            (bud["centroid_x"], bud["centroid_y"]),
        ]
        fallback = expanded_box(signal_points, width, height, PAIR_PADDING)
        seed_keys = [f"n{first[0]}", f"n{second[0]}", f"b{bud_idx}"]
        box, box_note = choose_mask_or_fallback(seed_keys, seed_lookup, signal_points, fallback, width, height)
        mask_union_labels += int(box_note.startswith("mask_union"))
        fallback_dot_labels += int(box_note.startswith("fallback_dot"))
        rows.append(
            row_from_box(
                image_id=image_id,
                class_name="mother_bud_pair",
                box=box,
                source="v6_mask_aware_pseudo_label",
                notes=(
                    f"two_nuclei_near_budneck; budneck_idx={bud_idx}; "
                    f"mask_source={mask_source}; box_source={box_note}"
                ),
            )
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
        signal_points = [(nucleus["centroid_x"], nucleus["centroid_y"]), (bud["centroid_x"], bud["centroid_y"])]
        fallback = expanded_box(signal_points, width, height, EARLY_PADDING)
        seed_keys = [f"n{idx}", f"b{bud_idx}"]
        box, box_note = choose_mask_or_fallback(seed_keys, seed_lookup, signal_points, fallback, width, height)
        mask_union_labels += int(box_note.startswith("mask_union"))
        fallback_dot_labels += int(box_note.startswith("fallback_dot"))
        rows.append(
            row_from_box(
                image_id=image_id,
                class_name="early_bud_pair",
                box=box,
                source="v6_mask_aware_pseudo_label",
                notes=(
                    f"one_nucleus_near_budneck; budneck_idx={bud_idx}; "
                    f"mask_source={mask_source}; box_source={box_note}"
                ),
            )
        )
        used_nuclei.add(idx)
        used_budnecks.add(bud_idx)

    for idx, nucleus in enumerate(nuclei):
        if idx in used_nuclei:
            continue
        signal_points = [(nucleus["centroid_x"], nucleus["centroid_y"])]
        fallback = single_box(nucleus, width, height)
        seed_keys = [f"n{idx}"]
        box, box_note = choose_mask_or_fallback(seed_keys, seed_lookup, signal_points, fallback, width, height)
        mask_union_labels += int(box_note.startswith("mask_union"))
        fallback_dot_labels += int(box_note.startswith("fallback_dot"))
        rows.append(
            row_from_box(
                image_id=image_id,
                class_name="single_cell",
                box=box,
                source="v6_mask_aware_pseudo_label",
                notes=(
                    f"unpaired_green_nucleus; nucleus_idx={idx}; "
                    f"mask_source={mask_source}; box_source={box_note}"
                ),
            )
        )

    diagnostics = {
        "image_id": image_id,
        "mask_source": mask_source,
        "raw_mask_count": len(raw_masks),
        "seeded_mask_count": seeded_mask_count,
        "nucleus_count": len(nuclei),
        "budneck_count": len(budnecks),
        "mask_union_labels": mask_union_labels,
        "fallback_dot_labels": fallback_dot_labels,
    }
    return rows, diagnostics


def write_csv(rows: list[dict[str, Any]], output_path: Path, columns: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="labels_pseudo_mask_aware.csv")
    parser.add_argument("--diagnostics-name", default="mask_aware_diagnostics.csv")
    parser.add_argument("--cell-mask-dir", type=Path, default=None, help="Optional directory of external CellSAM/YeastSAM mask images.")
    parser.add_argument("--trans-dir", type=Path, default=None, help="Optional directory of transmitted/DIC/brightfield images.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = read_manifest(args.manifest)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for meta in manifest:
        image_rows, image_diagnostics = assign_pseudo_labels(meta, args.cell_mask_dir, args.trans_dir)
        rows.extend(image_rows)
        diagnostics.append(image_diagnostics)

    output_path = args.output_dir / args.output_name
    diagnostics_path = args.output_dir / args.diagnostics_name
    write_csv(rows, output_path, LABEL_COLUMNS)
    write_csv(diagnostics, diagnostics_path, DIAGNOSTIC_COLUMNS)

    counts: dict[str, int] = {}
    mask_union_count = 0
    fallback_count = 0
    for row in rows:
        counts[str(row["class_name"])] = counts.get(str(row["class_name"]), 0) + 1
        notes = str(row["notes"])
        mask_union_count += int("box_source=mask_union" in notes)
        fallback_count += int("box_source=fallback_dot" in notes)

    print(f"Wrote {len(rows)} mask-aware pseudo-label boxes to {output_path}")
    for class_name in ("single_cell", "mother_bud_pair", "early_bud_pair"):
        print(f"  {class_name}: {counts.get(class_name, 0)}")
    print(f"  mask_union boxes: {mask_union_count}")
    print(f"  fallback_dot boxes: {fallback_count}")
    print(f"Wrote diagnostics to {diagnostics_path}")
    print("All pseudo-label rows are marked review_status=needs_review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
