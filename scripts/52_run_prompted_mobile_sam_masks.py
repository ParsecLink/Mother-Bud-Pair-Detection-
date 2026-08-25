#!/usr/bin/env python
"""Run prompted MobileSAM masks for v6 rule pseudo-label objects.

This script uses the same RGB green-nucleus / magenta-budneck assignment rules
as the v6 pseudo-label bootstrapper. For each rule object, it prompts MobileSAM
with the rule box plus positive points at the matched nucleus/bud-neck centers.
The output is one integer instance-mask PNG per image, suitable for
scripts/51_generate_v6_mask_aware_pseudo_labels.py --cell-mask-dir.
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
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_OUTPUT_DIR = V6_DIR / "sam_masks" / "prompted_mobile_sam"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "mobile_sam.pt"
LABEL_COLUMNS = ["image_id", "class_name", "x1", "y1", "x2", "y2", "source", "review_status", "notes"]
PROMPT_COLUMNS = [
    "image_id",
    "instance_id",
    "class_name",
    "prompt_box_x1",
    "prompt_box_y1",
    "prompt_box_x2",
    "prompt_box_y2",
    "sam_score",
    "mask_area",
    "mask_box_x1",
    "mask_box_y1",
    "mask_box_x2",
    "mask_box_y2",
    "prompt_points",
    "quality_flag",
    "notes",
]

PAIR_RADIUS = 38.0
EARLY_BUD_RADIUS = 30.0
PAIR_MAX_NUCLEUS_DISTANCE = 56.0
SINGLE_BOX_SIZE = 34.0
PAIR_PADDING = 18.0
EARLY_PADDING = 20.0
MASK_MIN_AREA = 24
MASK_MAX_PROMPT_AREA_RATIO = 5.5


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
    x = nucleus["centroid_x"]
    y = nucleus["centroid_y"]
    return (
        max(0.0, x - half),
        max(0.0, y - half),
        min(float(image_width - 1), x + half),
        min(float(image_height - 1), y + half),
    )


def rule_prompt_objects(meta: dict[str, str]) -> list[dict[str, Any]]:
    image = Image.open(PROJECT_ROOT / meta["image_path"]).convert("RGB")
    width, height = image.size
    rgb = np.asarray(image)
    green_mask, magenta_mask = channel_masks(rgb)
    nuclei = connected_components(green_mask, min_area=8, max_area=900)
    budnecks = connected_components(magenta_mask, min_area=4, max_area=500)

    objects: list[dict[str, Any]] = []
    used_nuclei: set[int] = set()
    used_budnecks: set[int] = set()

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
        if distance(first[1], second[1]) > PAIR_MAX_NUCLEUS_DISTANCE:
            continue
        points = [
            (first[1]["centroid_x"], first[1]["centroid_y"]),
            (second[1]["centroid_x"], second[1]["centroid_y"]),
            (bud["centroid_x"], bud["centroid_y"]),
        ]
        objects.append(
            {
                "class_name": "mother_bud_pair",
                "box": expanded_box(points, width, height, PAIR_PADDING),
                "points": points,
                "notes": f"two_nuclei_near_budneck; nuclei={first[0]}+{second[0]}; budneck_idx={bud_idx}",
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
        objects.append(
            {
                "class_name": "early_bud_pair",
                "box": expanded_box(points, width, height, EARLY_PADDING),
                "points": points,
                "notes": f"one_nucleus_near_budneck; nucleus_idx={idx}; budneck_idx={bud_idx}",
            }
        )
        used_nuclei.add(idx)
        used_budnecks.add(bud_idx)

    for idx, nucleus in enumerate(nuclei):
        if idx in used_nuclei:
            continue
        points = [(nucleus["centroid_x"], nucleus["centroid_y"])]
        objects.append(
            {
                "class_name": "single_cell",
                "box": single_box(nucleus, width, height),
                "points": points,
                "notes": f"unpaired_green_nucleus; nucleus_idx={idx}",
            }
        )

    return objects


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def prompt_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(1.0, (x2 - x1) * (y2 - y1))


def clean_mask_with_prompt(mask: np.ndarray, box: tuple[float, float, float, float], points: list[tuple[float, float]]) -> tuple[np.ndarray, str]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    x1, y1, x2, y2 = box
    padded_x1 = max(0, int(math.floor(x1 - 8)))
    padded_y1 = max(0, int(math.floor(y1 - 8)))
    padded_x2 = min(width, int(math.ceil(x2 + 8)))
    padded_y2 = min(height, int(math.ceil(y2 + 8)))
    prompt_window = np.zeros_like(mask, dtype=bool)
    prompt_window[padded_y1:padded_y2, padded_x1:padded_x2] = True
    mask = mask & prompt_window

    if not mask.any():
        return mask, "empty_after_prompt_crop"

    components = connected_components(mask, min_area=1, max_area=height * width)
    if not components:
        return np.zeros_like(mask, dtype=bool), "no_component"

    selected = np.zeros_like(mask, dtype=bool)
    for component in components:
        component_mask = np.zeros_like(mask, dtype=bool)
        cx1, cy1, cx2, cy2 = int(component["x1"]), int(component["y1"]), int(component["x2"]), int(component["y2"])
        sub = mask[cy1:cy2, cx1:cx2]
        component_mask[cy1:cy2, cx1:cx2] = sub
        keep = False
        for px, py in points:
            ix = int(round(px))
            iy = int(round(py))
            if 0 <= ix < width and 0 <= iy < height and component_mask[iy, ix]:
                keep = True
                break
            near_x1 = max(0, ix - 5)
            near_y1 = max(0, iy - 5)
            near_x2 = min(width, ix + 6)
            near_y2 = min(height, iy + 6)
            if component_mask[near_y1:near_y2, near_x1:near_x2].any():
                keep = True
                break
        if keep:
            selected |= component_mask

    if selected.any():
        return selected, "ok_mask"

    largest = max(components, key=lambda item: item["area"])
    largest_mask = np.zeros_like(mask, dtype=bool)
    lx1, ly1, lx2, ly2 = int(largest["x1"]), int(largest["y1"]), int(largest["x2"]), int(largest["y2"])
    largest_mask[ly1:ly2, lx1:lx2] = mask[ly1:ly2, lx1:lx2]
    return largest_mask, "largest_component_no_point_hit"


def draw_overlay(image: Image.Image, rows: list[dict[str, Any]], output_path: Path) -> None:
    overlay = image.convert("RGBA")
    mask_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw_mask = ImageDraw.Draw(mask_layer)
    draw = ImageDraw.Draw(overlay)
    colors = {
        "single_cell": (0, 255, 255, 96),
        "mother_bud_pair": (255, 210, 0, 96),
        "early_bud_pair": (255, 80, 220, 96),
    }
    for row in rows:
        mask = row.get("_mask")
        if mask is not None and mask.any():
            rgba = colors.get(str(row["class_name"]), (255, 255, 255, 96))
            ys, xs = np.nonzero(mask)
            for x, y in zip(xs.tolist(), ys.tolist()):
                draw_mask.point((x, y), fill=rgba)
        box = row.get("_mask_box") or row["box"]
        outline = colors.get(str(row["class_name"]), (255, 255, 255, 96))[:3] + (255,)
        draw.rectangle(box, outline=outline, width=2)
    composite = Image.alpha_composite(overlay, mask_layer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.convert("RGB").save(output_path)


def run_mobile_sam(
    manifest: list[dict[str, str]],
    checkpoint: Path,
    output_dir: Path,
    device: str,
    max_images: int | None,
) -> tuple[list[dict[str, Any]], int]:
    from mobile_sam import SamPredictor, sam_model_registry
    import torch

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    sam.to(device=resolved_device)
    sam.eval()
    predictor = SamPredictor(sam)

    instances_dir = output_dir / "instances"
    overlays_dir = output_dir / "overlays"
    instances_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    mask_count = 0
    selected_manifest = manifest if max_images is None else manifest[:max_images]

    for image_index, meta in enumerate(selected_manifest, start=1):
        image = Image.open(PROJECT_ROOT / meta["image_path"]).convert("RGB")
        rgb = np.asarray(image)
        predictor.set_image(rgb)
        objects = rule_prompt_objects(meta)
        instance_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint16)
        overlay_rows: list[dict[str, Any]] = []
        local_id = 1

        for obj in objects:
            box = tuple(float(value) for value in obj["box"])
            points = [(float(x), float(y)) for x, y in obj["points"]]
            point_coords = np.asarray(points, dtype=np.float32)
            point_labels = np.ones((len(points),), dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.asarray(box, dtype=np.float32),
                multimask_output=True,
            )
            best_index = int(np.argmax(scores))
            mask, quality = clean_mask_with_prompt(masks[best_index], box, points)
            area = int(mask.sum())
            mask_box = mask_bbox(mask)
            prompt_area_value = prompt_area(box)
            if area < MASK_MIN_AREA:
                quality = f"reject_small_mask:{quality}"
                mask = np.zeros_like(mask, dtype=bool)
                mask_box = None
            elif area > prompt_area_value * MASK_MAX_PROMPT_AREA_RATIO:
                quality = f"review_large_mask:{quality}"

            if mask.any():
                instance_mask[(mask) & (instance_mask == 0)] = local_id
                mask_count += 1

            row = {
                "image_id": meta["image_id"],
                "instance_id": local_id,
                "class_name": obj["class_name"],
                "prompt_box_x1": round(box[0], 2),
                "prompt_box_y1": round(box[1], 2),
                "prompt_box_x2": round(box[2], 2),
                "prompt_box_y2": round(box[3], 2),
                "sam_score": round(float(scores[best_index]), 5),
                "mask_area": area,
                "mask_box_x1": round(mask_box[0], 2) if mask_box else "",
                "mask_box_y1": round(mask_box[1], 2) if mask_box else "",
                "mask_box_x2": round(mask_box[2], 2) if mask_box else "",
                "mask_box_y2": round(mask_box[3], 2) if mask_box else "",
                "prompt_points": ";".join(f"{x:.2f}:{y:.2f}" for x, y in points),
                "quality_flag": quality,
                "notes": obj["notes"],
                "_mask": mask,
                "_mask_box": mask_box,
                "box": box,
            }
            rows.append({key: value for key, value in row.items() if not key.startswith("_") and key != "box"})
            overlay_rows.append(row)
            local_id += 1

        Image.fromarray(instance_mask).save(instances_dir / f"{Path(meta['image_path']).stem}.png")
        draw_overlay(image, overlay_rows, overlays_dir / f"{Path(meta['image_path']).stem}_mobile_sam_overlay.png")
        print(f"[{image_index}/{len(selected_manifest)}] {meta['image_id']}: {len(objects)} prompts, {int((instance_mask > 0).sum())} mask pixels", flush=True)

    return rows, mask_count


def write_csv(rows: list[dict[str, Any]], output_path: Path, columns: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-images", type=int, default=None, help="Debug option: process only the first N manifest images.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"MobileSAM checkpoint not found: {args.checkpoint}")
    manifest = read_manifest(args.manifest)
    rows, mask_count = run_mobile_sam(
        manifest=manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        max_images=args.max_images,
    )
    write_csv(rows, args.output_dir / "prompted_mobile_sam_results.csv", PROMPT_COLUMNS)
    print(f"Wrote {len(rows)} prompt rows to {args.output_dir / 'prompted_mobile_sam_results.csv'}")
    print(f"Wrote instance masks to {args.output_dir / 'instances'}")
    print(f"Wrote overlays to {args.output_dir / 'overlays'}")
    print(f"Generated {mask_count} non-empty prompted masks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
