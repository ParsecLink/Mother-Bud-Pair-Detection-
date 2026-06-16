#!/usr/bin/env python
"""Build v5 all-object trial boxes and run SAM when a checkpoint is available.

This is a review/pilot step. It uses the clean v5 biological classes:
single_cell, mother_bud_pair, and early_bud_pair. Rejected and uncertain pair
candidates are not boxed. The overlay style is intentionally plain: box
outlines plus SAM mask contours only, with no IDs or text on the image.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage import measure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    MERGED_RGB_ENHANCED_DIR,
    PROJECTED_DIR,
    SAM_TRIAL_CHECKPOINT_PATH,
    SAM_TRIAL_MODEL_TYPE,
    SAM_TRIAL_MULTIMASK_OUTPUT,
)
from my_sam_pipeline.draft_boxes_v2_3 import (  # noqa: E402
    coords_extent_along_axis,
    oriented_box_corners,
)
from my_sam_pipeline.io_utils import ensure_dir, read_tiff_stack  # noqa: E402
from my_sam_pipeline.rule_features import axis_angle_deg  # noqa: E402
from my_sam_pipeline.sam_boxes_no_cell_wall_trial import (  # noqa: E402
    build_sam_prompt_inputs,
    load_projected_stacks,
    resolve_sam_checkpoint,
    run_sam_for_prompts,
    sanitize_filename,
)


SOURCE_DIR = PROJECT_ROOT / "rule_discovery" / "rules_v5_all_objects_matching"
SOURCE_TABLES_DIR = SOURCE_DIR / "tables"
TRIAL_DIR = PROJECT_ROOT / "rule_discovery" / "sam_v5_all_objects_trial"
TABLES_DIR = TRIAL_DIR / "tables"
BOXES_DIR = TRIAL_DIR / "boxes"
MASKS_DIR = TRIAL_DIR / "masks"
OVERLAYS_DIR = TRIAL_DIR / "overlays"
COMBINED_OVERLAYS_DIR = OVERLAYS_DIR / "full_frame_combined"
OBJECT_OVERLAYS_DIR = OVERLAYS_DIR / "full_frame_by_object"
DEBUG_DIR = TRIAL_DIR / "debug"

OLD_V5_BOX_PATHS = [
    SOURCE_DIR / "boxes",
    SOURCE_DIR / "overlays" / "box_only_full_images",
    SOURCE_TABLES_DIR / "draft_boxes_v5_all_objects.csv",
    SOURCE_TABLES_DIR / "draft_boxes_v5_all_objects_summary.csv",
]

ACCEPTED_CLASSES = ("mother_bud_pair", "early_bud_pair", "single_cell")

BOX_COLOR = (0, 255, 255, 235)
SAM_CONTOUR_COLOR = (255, 230, 0, 245)
BOX_LINE_WIDTH = 2
MASK_LINE_WIDTH = 2


def clean_previous_box_results() -> list[str]:
    """Delete only previous v5 box/SAM-trial artifacts inside Model/My."""

    if os.environ.get("SAM_RESUME_EXISTING", "").lower() in {"1", "true", "yes"}:
        return []

    removed: list[str] = []
    for path in OLD_V5_BOX_PATHS:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path))
    if TRIAL_DIR.exists():
        shutil.rmtree(TRIAL_DIR)
        removed.append(str(TRIAL_DIR))
    return removed


def load_source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load v5 clean biological classes and supporting object tables."""

    clean = pd.read_csv(SOURCE_TABLES_DIR / "clean_biological_classification_all_objects.csv")
    full = pd.read_csv(SOURCE_TABLES_DIR / "rules_v5_all_objects_classification.csv")
    nuclei = pd.read_csv(SOURCE_TABLES_DIR / "all_objects_adapted_nucleus_candidates.csv")
    budnecks = pd.read_csv(SOURCE_TABLES_DIR / "all_objects_adapted_budneck_candidates.csv")
    return clean, full, nuclei, budnecks


def parse_contour(value: object) -> np.ndarray:
    """Parse contour JSON into y/x coordinates."""

    if not isinstance(value, str) or not value or value == "[]":
        return np.zeros((0, 2), dtype=np.float32)
    try:
        points = json.loads(value)
    except json.JSONDecodeError:
        return np.zeros((0, 2), dtype=np.float32)
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def point_lookup(df: pd.DataFrame, id_col: str) -> dict[tuple[str, int, int], pd.Series]:
    """Build condition/frame/id lookup for nuclei or budneck tables."""

    lookup: dict[tuple[str, int, int], pd.Series] = {}
    for row in df.itertuples(index=False):
        lookup[(str(row.condition), int(row.frame), int(getattr(row, id_col)))] = pd.Series(row._asdict())
    return lookup


def get_lookup_row(
    lookup: dict[tuple[str, int, int], pd.Series],
    condition: str,
    frame: int,
    object_id: object,
) -> pd.Series | None:
    """Return one lookup row, handling missing IDs."""

    if pd.isna(object_id):
        return None
    return lookup.get((condition, int(frame), int(object_id)))


def axis_aligned_bbox_from_corners(corners_yx: list[list[float]], image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return a clipped xyxy bounding rectangle around an oriented box."""

    corners = np.asarray(corners_yx, dtype=np.float32)
    height, width = int(image_shape[0]), int(image_shape[1])
    y1 = float(np.clip(np.floor(corners[:, 0].min()), 0, height - 1))
    x1 = float(np.clip(np.floor(corners[:, 1].min()), 0, width - 1))
    y2 = float(np.clip(np.ceil(corners[:, 0].max()), 0, height - 1))
    x2 = float(np.clip(np.ceil(corners[:, 1].max()), 0, width - 1))
    if x2 <= x1:
        x2 = min(width - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(height - 1.0, y1 + 1.0)
    return x1, y1, x2, y2


def make_box_for_row(
    row: pd.Series,
    nucleus_a: pd.Series,
    nucleus_b: pd.Series | None,
    budneck: pd.Series | None,
    image_shape: tuple[int, int],
) -> dict[str, object]:
    """Create one size-controlled oriented box for an accepted v5 class."""

    source_class = str(row["clean_class"])
    ay, ax = float(nucleus_a["centroid_y"]), float(nucleus_a["centroid_x"])
    by = float(nucleus_b["centroid_y"]) if nucleus_b is not None else np.nan
    bx = float(nucleus_b["centroid_x"]) if nucleus_b is not None else np.nan
    bud_y = float(budneck["centroid_y"]) if budneck is not None else np.nan
    bud_x = float(budneck["centroid_x"]) if budneck is not None else np.nan
    contour = parse_contour(row.get("contour_yx_json", "[]"))

    if source_class == "mother_bud_pair" and nucleus_b is not None:
        distance = float(row["distance"]) if pd.notna(row.get("distance", np.nan)) else float(math.hypot(by - ay, bx - ax))
        angle_deg = axis_angle_deg(by - ay, bx - ax)
        center_y = (ay + by) / 2.0
        center_x = (ax + bx) / 2.0
        if np.isfinite(bud_y) and np.isfinite(bud_x):
            center_y = 0.82 * center_y + 0.18 * bud_y
            center_x = 0.82 * center_x + 0.18 * bud_x
        long_extent, short_extent = coords_extent_along_axis(contour, center_y, center_x, angle_deg) if contour.size else (0.0, 0.0)
        # Pair boxes need padding around both nuclei but should not absorb a clump.
        long_size = float(np.clip(max(distance + 18.0, long_extent + 8.0, 42.0), 36.0, 94.0))
        short_size = float(np.clip(max(short_extent + 7.0, 31.0), 24.0, 56.0))
        quality = "ok_pair_box"
        reason = "mother_bud_pair: long axis follows nucleus_a_to_nucleus_b; size uses distance plus local contour extent"
    elif source_class == "early_bud_pair" and np.isfinite(bud_y) and np.isfinite(bud_x):
        distance = float(math.hypot(bud_y - ay, bud_x - ax))
        angle_deg = axis_angle_deg(bud_y - ay, bud_x - ax)
        center_y = 0.58 * ay + 0.42 * bud_y
        center_x = 0.58 * ax + 0.42 * bud_x
        long_extent, short_extent = coords_extent_along_axis(contour, center_y, center_x, angle_deg) if contour.size else (0.0, 0.0)
        long_size = float(np.clip(max(distance + 24.0, long_extent + 8.0, 40.0), 34.0, 90.0))
        short_size = float(np.clip(max(short_extent + 7.0, 30.0), 24.0, 54.0))
        quality = "ok_early_bud_box"
        reason = "early_bud_pair: long axis follows mother_nucleus_to_budneck; size covers mother plus attached bud"
    else:
        distance = np.nan
        angle_deg = 0.0
        center_y = ay
        center_x = ax
        object_area = float(nucleus_a.get("area", 0.0)) if pd.notna(nucleus_a.get("area", np.nan)) else 0.0
        # Estimate a conservative whole-cell box from fluorescence-object size.
        estimated_diameter = max(36.0, 2.6 * math.sqrt(max(object_area, 1.0)))
        long_size = float(np.clip(estimated_diameter, 34.0, 50.0))
        short_size = long_size
        quality = "ok_single_box" if long_size <= 48.0 else "review_single_box"
        reason = "single_cell: nucleus-centered conservative whole-cell box"

    corners = oriented_box_corners(center_y, center_x, long_size, short_size, angle_deg)
    x1, y1, x2, y2 = axis_aligned_bbox_from_corners(corners, image_shape)
    return {
        "center_y": float(center_y),
        "center_x": float(center_x),
        "width": float(long_size),
        "height": float(short_size),
        "angle_deg": float(angle_deg),
        "axis_aligned_x1": x1,
        "axis_aligned_y1": y1,
        "axis_aligned_x2": x2,
        "axis_aligned_y2": y2,
        "nucleus_distance": float(distance) if np.isfinite(distance) else np.nan,
        "box_quality_flag": quality,
        "box_reason": reason,
        "corner_yx_json": json.dumps(corners),
    }


def build_box_table(
    clean_df: pd.DataFrame,
    full_df: pd.DataFrame,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    trans_stacks: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build draft boxes for all accepted v5 biological classes."""

    source = clean_df[clean_df["clean_class"].isin(ACCEPTED_CLASSES)].copy()
    source = source.merge(
        full_df[
            [
                "condition",
                "frame",
                "candidate_id",
                "contour_yx_json",
                "cell_a_id",
                "cell_b_id",
                "nucleus_a_best_z",
                "nucleus_b_best_z",
                "budneck_best_z",
            ]
        ],
        on=["condition", "frame", "candidate_id"],
        how="left",
    )
    nuclei_lookup = point_lookup(nuclei_df, "nucleus_id")
    budneck_lookup = point_lookup(budneck_df, "budneck_id")
    rows: list[dict[str, object]] = []

    for row_index, row in source.reset_index(drop=True).iterrows():
        condition = str(row["condition"])
        frame = int(row["frame"])
        if condition not in trans_stacks or frame >= len(trans_stacks[condition]):
            continue
        image_shape = tuple(np.asarray(trans_stacks[condition][frame]).shape[:2])
        nucleus_a = get_lookup_row(nuclei_lookup, condition, frame, row["nucleus_a_id"])
        if nucleus_a is None:
            continue
        nucleus_b = get_lookup_row(nuclei_lookup, condition, frame, row.get("nucleus_b_id", np.nan))
        budneck = get_lookup_row(budneck_lookup, condition, frame, row.get("budneck_id", np.nan))
        box = make_box_for_row(row, nucleus_a, nucleus_b, budneck, image_shape)
        rows.append(
            {
                "condition": condition,
                "frame": frame,
                "box_id": int(row_index) + 1,
                "source_class": str(row["clean_class"]),
                "candidate_id": int(row["candidate_id"]),
                "nucleus_a_id": int(row["nucleus_a_id"]),
                "nucleus_b_id": int(row["nucleus_b_id"]) if pd.notna(row.get("nucleus_b_id", np.nan)) else np.nan,
                "budneck_id": int(row["budneck_id"]) if pd.notna(row.get("budneck_id", np.nan)) else np.nan,
                "nucleus_a_best_z": row.get("nucleus_a_best_z", np.nan),
                "nucleus_b_best_z": row.get("nucleus_b_best_z", np.nan),
                "budneck_best_z": row.get("budneck_best_z", np.nan),
                "cells_are_adjacent": bool(row.get("cells_are_adjacent", False)) if pd.notna(row.get("cells_are_adjacent", np.nan)) else False,
                "nucleus_line_hits_budneck": bool(row.get("nucleus_line_hits_budneck", False)) if pd.notna(row.get("nucleus_line_hits_budneck", np.nan)) else False,
                "budneck_between_nuclei": bool(row.get("budneck_between_nuclei", False)) if pd.notna(row.get("budneck_between_nuclei", np.nan)) else False,
                **box,
            }
        )

    boxes = pd.DataFrame(rows)
    if boxes.empty:
        return boxes
    # SAM image embeddings are the expensive step. Keep prompts grouped by frame
    # so one embedding can serve all boxes in the same Trans image.
    boxes = boxes.sort_values(["condition", "frame", "source_class", "box_id"]).reset_index(drop=True)
    boxes["box_id"] = np.arange(1, len(boxes) + 1, dtype=int)
    return boxes


def iter_all_frames() -> list[tuple[str, int]]:
    """Return all projected Trans frames."""

    keys: list[tuple[str, int]] = []
    for path in sorted(PROJECTED_DIR.glob("*_Trans.tif")):
        condition = path.name[: -len("_Trans.tif")]
        stack = read_tiff_stack(path)
        for frame in range(int(stack.shape[0])):
            keys.append((condition, frame))
    return keys


def background_image(condition: str, frame: int, trans_stacks: dict[str, np.ndarray]) -> Image.Image:
    """Load merged RGB background when available, otherwise Trans as grayscale RGB."""

    path = MERGED_RGB_ENHANCED_DIR / f"{condition}_frame_{frame:03d}.png"
    if path.exists():
        return Image.open(path).convert("RGBA")
    trans = np.asarray(trans_stacks[condition][frame], dtype=np.float32)
    lo, hi = np.percentile(trans, [1, 99.8])
    scaled = np.clip((trans - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.clip(scaled * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([rgb, rgb, rgb], axis=-1), mode="RGB").convert("RGBA")


def draw_oriented_box(draw: ImageDraw.ImageDraw, corners_json: str) -> None:
    """Draw one oriented box outline."""

    corners = json.loads(str(corners_json))
    points = [(float(x), float(y)) for y, x in corners]
    if not points:
        return
    points.append(points[0])
    draw.line(points, fill=BOX_COLOR, width=BOX_LINE_WIDTH, joint="curve")


def draw_mask_contour(draw: ImageDraw.ImageDraw, mask_path: object) -> bool:
    """Draw SAM mask contour if a mask file exists."""

    if not isinstance(mask_path, str) or not mask_path:
        return False
    path = Path(mask_path)
    if not path.exists():
        return False
    mask = imageio.imread(path) > 0
    drawn = False
    for contour in measure.find_contours(mask.astype(np.uint8), 0.5):
        points = [(float(x), float(y)) for y, x in contour]
        if len(points) >= 2:
            draw.line(points, fill=SAM_CONTOUR_COLOR, width=MASK_LINE_WIDTH)
            drawn = True
    return drawn


def save_combined_overlays(
    boxes_df: pd.DataFrame,
    masks_df: pd.DataFrame,
    trans_stacks: dict[str, np.ndarray],
) -> int:
    """Save one full-frame overlay per projected image with all accepted boxes."""

    ensure_dir(COMBINED_OVERLAYS_DIR)
    mask_lookup = {int(row.box_id): str(row.sam_mask_path) for row in masks_df.itertuples(index=False)}
    count = 0
    for condition, frame in iter_all_frames():
        boxes = boxes_df[(boxes_df["condition"] == condition) & (boxes_df["frame"] == frame)]
        image = background_image(condition, frame, trans_stacks)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for box in boxes.itertuples(index=False):
            draw_oriented_box(draw, str(box.corner_yx_json))
            draw_mask_contour(draw, mask_lookup.get(int(box.box_id), ""))
        output = COMBINED_OVERLAYS_DIR / f"{sanitize_filename(condition)}_frame_{frame:03d}_v5_boxes_sam_boundary.png"
        Image.alpha_composite(image, overlay).convert("RGB").save(output)
        count += 1
    return count


def save_object_overlays(
    boxes_df: pd.DataFrame,
    masks_df: pd.DataFrame,
    trans_stacks: dict[str, np.ndarray],
) -> int:
    """Save one full-frame overlay per object with only that object's box/mask."""

    ensure_dir(OBJECT_OVERLAYS_DIR)
    masks = {int(row.box_id): str(row.sam_mask_path) for row in masks_df.itertuples(index=False)}
    count = 0
    for box in boxes_df.itertuples(index=False):
        image = background_image(str(box.condition), int(box.frame), trans_stacks)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw_oriented_box(draw, str(box.corner_yx_json))
        draw_mask_contour(draw, masks.get(int(box.box_id), ""))
        output = (
            OBJECT_OVERLAYS_DIR
            / f"{sanitize_filename(str(box.condition))}_frame_{int(box.frame):03d}_box_{int(box.box_id):04d}_{str(box.source_class)}.png"
        )
        Image.alpha_composite(image, overlay).convert("RGB").save(output)
        count += 1
    return count


def write_box_mask_tiffs(boxes_df: pd.DataFrame, trans_stacks: dict[str, np.ndarray]) -> int:
    """Write simple binary oriented-box masks for bookkeeping/review."""

    ensure_dir(BOXES_DIR)
    count = 0
    for box in boxes_df.itertuples(index=False):
        shape = np.asarray(trans_stacks[str(box.condition)][int(box.frame)]).shape[:2]
        mask = np.zeros(shape, dtype=np.uint8)
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        corners = np.asarray(json.loads(str(box.corner_yx_json)), dtype=np.float32)
        theta = math.radians(float(box.angle_deg))
        axis = np.asarray([math.sin(theta), math.cos(theta)], dtype=np.float32)
        perp = np.asarray([math.cos(theta), -math.sin(theta)], dtype=np.float32)
        rel = np.stack([yy - float(box.center_y), xx - float(box.center_x)], axis=-1)
        long_proj = rel @ axis
        short_proj = rel @ perp
        inside = (np.abs(long_proj) <= float(box.width) / 2.0) & (np.abs(short_proj) <= float(box.height) / 2.0)
        mask[inside] = 255
        output = BOXES_DIR / f"{sanitize_filename(str(box.condition))}_frame_{int(box.frame):03d}_box_{int(box.box_id):04d}.png"
        imageio.imwrite(output, mask)
        count += 1
    return count


def build_summary(
    boxes_df: pd.DataFrame,
    masks_df: pd.DataFrame,
    removed_paths: list[str],
    combined_overlay_count: int,
    object_overlay_count: int,
    box_mask_count: int,
    checkpoint_path: Path | None,
) -> pd.DataFrame:
    """Build final run summary table."""

    counts = boxes_df["source_class"].value_counts() if not boxes_df.empty else pd.Series(dtype=int)
    mask_flags = masks_df["mask_quality_flag"].value_counts() if not masks_df.empty else pd.Series(dtype=int)
    return pd.DataFrame(
        [
            {"metric": "source_rule_table", "value": str(SOURCE_TABLES_DIR / "clean_biological_classification_all_objects.csv")},
            {"metric": "previous_v5_box_artifacts_removed", "value": len(removed_paths)},
            {"metric": "removed_paths_json", "value": json.dumps(removed_paths)},
            {"metric": "mother_bud_pair_boxes", "value": int(counts.get("mother_bud_pair", 0))},
            {"metric": "early_bud_pair_boxes", "value": int(counts.get("early_bud_pair", 0))},
            {"metric": "single_cell_boxes", "value": int(counts.get("single_cell", 0))},
            {"metric": "total_boxes", "value": int(len(boxes_df))},
            {"metric": "box_mask_png_count", "value": int(box_mask_count)},
            {"metric": "sam_checkpoint_found", "value": bool(checkpoint_path is not None)},
            {"metric": "sam_checkpoint_path", "value": str(checkpoint_path) if checkpoint_path is not None else ""},
            {"metric": "sam_masks_generated", "value": int((masks_df["sam_mask_path"].astype(str) != "").sum()) if not masks_df.empty else 0},
            {"metric": "sam_not_run_rows", "value": int(mask_flags.get("sam_not_run", 0))},
            {"metric": "review_mask_rows", "value": int(mask_flags.get("review_mask", 0))},
            {"metric": "ok_mask_rows", "value": int(mask_flags.get("ok_mask", 0))},
            {"metric": "combined_full_frame_overlay_count", "value": int(combined_overlay_count)},
            {"metric": "object_full_frame_overlay_count", "value": int(object_overlay_count)},
            {"metric": "tables_dir", "value": str(TABLES_DIR)},
            {"metric": "boxes_dir", "value": str(BOXES_DIR)},
            {"metric": "masks_dir", "value": str(MASKS_DIR)},
            {"metric": "combined_overlays_dir", "value": str(COMBINED_OVERLAYS_DIR)},
            {"metric": "object_overlays_dir", "value": str(OBJECT_OVERLAYS_DIR)},
            {"metric": "image_style", "value": "box outlines plus SAM mask boundary only; no text labels on images"},
        ]
    )


def main() -> None:
    removed_paths = clean_previous_box_results()
    for directory in [TABLES_DIR, BOXES_DIR, MASKS_DIR, COMBINED_OVERLAYS_DIR, OBJECT_OVERLAYS_DIR, DEBUG_DIR]:
        ensure_dir(directory)

    clean_df, full_df, nuclei_df, budneck_df = load_source_tables()
    stacks = load_projected_stacks(PROJECTED_DIR)
    trans_stacks = {condition: channel_stacks["Trans"] for condition, channel_stacks in stacks.items() if "Trans" in channel_stacks}

    boxes_df = build_box_table(clean_df, full_df, nuclei_df, budneck_df, trans_stacks)
    boxes_df.to_csv(TABLES_DIR / "draft_boxes_v5_all_objects_sam.csv", index=False)
    boxes_df.to_csv(BOXES_DIR / "draft_boxes_v5_all_objects_sam.csv", index=False)
    box_mask_count = write_box_mask_tiffs(boxes_df, trans_stacks)

    prompt_df = build_sam_prompt_inputs(boxes_df, nuclei_df, budneck_df)
    prompt_df.to_csv(TABLES_DIR / "sam_prompt_inputs_v5_all_objects.csv", index=False)

    checkpoint_path = resolve_sam_checkpoint(SAM_TRIAL_CHECKPOINT_PATH)
    masks_df = run_sam_for_prompts(
        boxes_df=boxes_df,
        prompt_df=prompt_df,
        trans_stacks=trans_stacks,
        masks_dir=MASKS_DIR,
        model_type=SAM_TRIAL_MODEL_TYPE,
        checkpoint_path=checkpoint_path,
        multimask_output=SAM_TRIAL_MULTIMASK_OUTPUT,
    )
    masks_df.to_csv(TABLES_DIR / "sam_mask_results_v5_all_objects.csv", index=False)

    combined_overlay_count = save_combined_overlays(boxes_df, masks_df, trans_stacks)
    object_overlay_count = save_object_overlays(boxes_df, masks_df, trans_stacks)

    summary = build_summary(
        boxes_df=boxes_df,
        masks_df=masks_df,
        removed_paths=removed_paths,
        combined_overlay_count=combined_overlay_count,
        object_overlay_count=object_overlay_count,
        box_mask_count=box_mask_count,
        checkpoint_path=checkpoint_path,
    )
    summary.to_csv(TABLES_DIR / "sam_v5_all_objects_summary.csv", index=False)

    print("v5 all-object SAM/box trial complete", flush=True)
    print(summary.to_string(index=False), flush=True)
    if checkpoint_path is None:
        print(
            "SAM checkpoint was not found, so SAM masks were not generated. "
            "Set SAM_CHECKPOINT_PATH or SAM_TRIAL_CHECKPOINT_PATH to run mask prediction.",
            flush=True,
        )


if __name__ == "__main__":
    main()
