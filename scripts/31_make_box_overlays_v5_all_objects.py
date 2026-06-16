#!/usr/bin/env python
"""Create box-only full-frame overlays from all-object rule results."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MERGED_RGB_ENHANCED_DIR, PROJECTED_DIR  # noqa: E402
from my_sam_pipeline.draft_boxes_v2_3 import coords_extent_along_axis, oriented_box_corners  # noqa: E402
from my_sam_pipeline.io_utils import ensure_dir, read_tiff_stack  # noqa: E402
from my_sam_pipeline.rule_features import axis_angle_deg  # noqa: E402


TRIAL_DIR = PROJECT_ROOT / "rule_discovery" / "rules_v5_all_objects_matching"
TABLES_DIR = TRIAL_DIR / "tables"
BOX_DIR = TRIAL_DIR / "boxes"
BOX_OVERLAY_DIR = TRIAL_DIR / "overlays" / "box_only_full_images"

BOX_COLOR = (0, 255, 255, 235)
BOX_WIDTH = 2


def iter_conditions(projected_dir: Path) -> list[str]:
    """Find conditions with projected Trans stacks."""

    return sorted(path.name[: -len("_Trans.tif")] for path in projected_dir.glob("*_Trans.tif"))


def background_png(condition: str, frame: int) -> Path:
    """Return the full-frame merged RGB background image."""

    return MERGED_RGB_ENHANCED_DIR / f"{condition}_frame_{frame:03d}.png"


def frame_keys() -> list[tuple[str, int]]:
    """Return all available condition/frame keys."""

    keys: list[tuple[str, int]] = []
    for condition in iter_conditions(PROJECTED_DIR):
        stack = read_tiff_stack(PROJECTED_DIR / f"{condition}_Trans.tif")
        for frame in range(int(stack.shape[0])):
            keys.append((condition, frame))
    return keys


def parse_contour(value: object) -> np.ndarray:
    """Parse contour JSON as y/x points."""

    if not isinstance(value, str) or not value or value == "[]":
        return np.zeros((0, 2), dtype=np.float32)
    try:
        points = json.loads(value)
    except json.JSONDecodeError:
        return np.zeros((0, 2), dtype=np.float32)
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def lookup_point(
    lookup: dict[tuple[str, int, int], tuple[float, float]],
    condition: str,
    frame: int,
    object_id: object,
) -> tuple[float, float] | None:
    """Return y/x point from a condition/frame/id lookup."""

    if pd.isna(object_id):
        return None
    return lookup.get((condition, int(frame), int(object_id)))


def make_box_for_row(
    row: pd.Series,
    nucleus_lookup: dict[tuple[str, int, int], tuple[float, float]],
    budneck_lookup: dict[tuple[str, int, int], tuple[float, float]],
) -> dict[str, object] | None:
    """Build one oriented box from one clean biological rule row."""

    condition = str(row["condition"])
    frame = int(row["frame"])
    source_class = str(row["clean_class"])
    nucleus_a = lookup_point(nucleus_lookup, condition, frame, row["nucleus_a_id"])
    if nucleus_a is None:
        return None
    nucleus_b = lookup_point(nucleus_lookup, condition, frame, row["nucleus_b_id"])
    budneck = lookup_point(budneck_lookup, condition, frame, row["budneck_id"])
    contour = parse_contour(row.get("contour_yx_json", "[]"))

    if source_class == "mother_bud_pair" and nucleus_b is not None:
        ay, ax = nucleus_a
        by, bx = nucleus_b
        center_y = (ay + by) / 2.0
        center_x = (ax + bx) / 2.0
        if budneck is not None:
            center_y = 0.78 * center_y + 0.22 * budneck[0]
            center_x = 0.78 * center_x + 0.22 * budneck[1]
        angle_deg = axis_angle_deg(by - ay, bx - ax)
        distance = float(row["distance"]) if pd.notna(row["distance"]) else float(math.hypot(by - ay, bx - ax))
        long_extent, short_extent = coords_extent_along_axis(contour, center_y, center_x, angle_deg) if contour.size else (0.0, 0.0)
        width = float(np.clip(max(distance + 18.0, long_extent + 8.0, 40.0), 34.0, 96.0))
        height = float(np.clip(max(short_extent + 8.0, 30.0), 26.0, 58.0))
        reason = "mother_bud_pair: nucleus-to-nucleus axis"
    elif source_class == "early_bud_pair" and budneck is not None:
        ay, ax = nucleus_a
        by, bx = budneck
        angle_deg = axis_angle_deg(by - ay, bx - ax)
        distance = float(row["distance"]) if pd.notna(row["distance"]) else float(math.hypot(by - ay, bx - ax))
        center_y = 0.58 * ay + 0.42 * by
        center_x = 0.58 * ax + 0.42 * bx
        long_extent, short_extent = coords_extent_along_axis(contour, center_y, center_x, angle_deg) if contour.size else (0.0, 0.0)
        width = float(np.clip(max(distance + 24.0, long_extent + 8.0, 40.0), 34.0, 92.0))
        height = float(np.clip(max(short_extent + 8.0, 30.0), 24.0, 54.0))
        reason = "early_bud_pair: nucleus-to-budneck axis"
    else:
        ay, ax = nucleus_a
        center_y = ay
        center_x = ax
        angle_deg = 0.0
        width = 36.0
        height = 36.0
        reason = "single_cell: nucleus-centered fixed cell box"

    corners = oriented_box_corners(center_y, center_x, width, height, angle_deg)
    xs = [point[1] for point in corners]
    ys = [point[0] for point in corners]
    return {
        "condition": condition,
        "frame": frame,
        "candidate_id": int(row["candidate_id"]),
        "source_class": source_class,
        "nucleus_a_id": int(row["nucleus_a_id"]) if pd.notna(row["nucleus_a_id"]) else np.nan,
        "nucleus_b_id": int(row["nucleus_b_id"]) if pd.notna(row["nucleus_b_id"]) else np.nan,
        "budneck_id": int(row["budneck_id"]) if pd.notna(row["budneck_id"]) else np.nan,
        "center_y": float(center_y),
        "center_x": float(center_x),
        "width": float(width),
        "height": float(height),
        "angle_deg": float(angle_deg),
        "axis_aligned_x1": float(max(0.0, min(xs))),
        "axis_aligned_y1": float(max(0.0, min(ys))),
        "axis_aligned_x2": float(max(xs)),
        "axis_aligned_y2": float(max(ys)),
        "corner_yx_json": json.dumps(corners),
        "box_reason": reason,
    }


def build_box_table() -> pd.DataFrame:
    """Create box rows from accepted clean biological classes."""

    clean = pd.read_csv(TABLES_DIR / "clean_biological_classification_all_objects.csv")
    full = pd.read_csv(TABLES_DIR / "rules_v5_all_objects_classification.csv")
    nuclei = pd.read_csv(TABLES_DIR / "all_objects_adapted_nucleus_candidates.csv")
    budnecks = pd.read_csv(TABLES_DIR / "all_objects_adapted_budneck_candidates.csv")

    source = clean[clean["clean_class"].isin(["mother_bud_pair", "early_bud_pair", "single_cell"])].merge(
        full[["condition", "frame", "candidate_id", "contour_yx_json"]],
        on=["condition", "frame", "candidate_id"],
        how="left",
    )
    nucleus_lookup = {
        (str(row.condition), int(row.frame), int(row.nucleus_id)): (float(row.centroid_y), float(row.centroid_x))
        for row in nuclei.itertuples(index=False)
    }
    budneck_lookup = {
        (str(row.condition), int(row.frame), int(row.budneck_id)): (float(row.centroid_y), float(row.centroid_x))
        for row in budnecks.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    for row in source.iterrows():
        box = make_box_for_row(row[1], nucleus_lookup, budneck_lookup)
        if box is not None:
            box["box_id"] = len(rows) + 1
            rows.append(box)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)[
        [
            "condition",
            "frame",
            "box_id",
            "candidate_id",
            "source_class",
            "nucleus_a_id",
            "nucleus_b_id",
            "budneck_id",
            "center_y",
            "center_x",
            "width",
            "height",
            "angle_deg",
            "axis_aligned_x1",
            "axis_aligned_y1",
            "axis_aligned_x2",
            "axis_aligned_y2",
            "corner_yx_json",
            "box_reason",
        ]
    ]


def draw_boxes_for_frame(image_path: Path, output_path: Path, boxes: pd.DataFrame) -> None:
    """Draw only box outlines on a full-frame image."""

    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for row in boxes.itertuples(index=False):
        corners = json.loads(str(row.corner_yx_json))
        points = [(float(x), float(y)) for y, x in corners]
        if not points:
            continue
        points.append(points[0])
        draw.line(points, fill=BOX_COLOR, width=BOX_WIDTH, joint="curve")
    merged = Image.alpha_composite(image, overlay).convert("RGB")
    ensure_dir(output_path.parent)
    merged.save(output_path)


def save_box_overlays(boxes_df: pd.DataFrame) -> int:
    """Save one full-frame box-only image for every projected frame."""

    count = 0
    for condition, frame in frame_keys():
        image_path = background_png(condition, frame)
        if not image_path.exists():
            continue
        frame_boxes = boxes_df[(boxes_df["condition"] == condition) & (boxes_df["frame"] == frame)].copy()
        safe_condition = condition.replace("/", "_")
        draw_boxes_for_frame(
            image_path=image_path,
            output_path=BOX_OVERLAY_DIR / f"{safe_condition}_frame_{frame:03d}_boxes_only.png",
            boxes=frame_boxes,
        )
        count += 1
    return count


def build_summary(boxes_df: pd.DataFrame, overlay_count: int) -> pd.DataFrame:
    """Summarize box-only output."""

    counts = boxes_df["source_class"].value_counts() if not boxes_df.empty else pd.Series(dtype=int)
    rows = [
        {"metric": "box_source", "value": "rules_v5_all_objects_matching clean biological classes"},
        {"metric": "mother_bud_pair_boxes", "value": int(counts.get("mother_bud_pair", 0))},
        {"metric": "early_bud_pair_boxes", "value": int(counts.get("early_bud_pair", 0))},
        {"metric": "single_cell_boxes", "value": int(counts.get("single_cell", 0))},
        {"metric": "total_boxes", "value": int(len(boxes_df))},
        {"metric": "full_frame_box_overlay_count", "value": int(overlay_count)},
        {"metric": "box_overlay_dir", "value": str(BOX_OVERLAY_DIR)},
        {"metric": "image_style", "value": "cyan box outlines only; no labels, IDs, side panel, or class text"},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    for directory in [TABLES_DIR, BOX_DIR, BOX_OVERLAY_DIR]:
        ensure_dir(directory)
    boxes_df = build_box_table()
    boxes_df.to_csv(BOX_DIR / "draft_boxes_v5_all_objects.csv", index=False)
    boxes_df.to_csv(TABLES_DIR / "draft_boxes_v5_all_objects.csv", index=False)
    overlay_count = save_box_overlays(boxes_df)
    summary = build_summary(boxes_df, overlay_count)
    summary.to_csv(TABLES_DIR / "draft_boxes_v5_all_objects_summary.csv", index=False)
    print("Box-only overlays complete", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
