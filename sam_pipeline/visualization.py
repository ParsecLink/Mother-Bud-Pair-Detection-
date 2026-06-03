"""Visualization helpers for channel PNGs, overlays, and review figures."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from PIL import ImageDraw

from .image_utils import to_uint8


def save_gray_png(path: Path, image: np.ndarray) -> None:
    """Save a grayscale float image as PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, to_uint8(image))


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    """Save an RGB float image as PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, to_uint8(image))


def green_overlay(image: np.ndarray) -> np.ndarray:
    """Render a grayscale fluorescence image as green RGB."""

    rgb = np.zeros((*image.shape, 3), dtype=np.float32)
    rgb[..., 1] = image
    return rgb


def magenta_overlay(image: np.ndarray) -> np.ndarray:
    """Render a grayscale fluorescence image as magenta RGB."""

    rgb = np.zeros((*image.shape, 3), dtype=np.float32)
    rgb[..., 0] = image
    rgb[..., 2] = image * 0.95
    return rgb


def merge_rgb(trans: np.ndarray, gfp: np.ndarray, mcherry: np.ndarray) -> np.ndarray:
    """Create a balanced merged RGB image."""

    rgb = np.stack([trans, trans, trans], axis=-1)
    rgb[..., 1] = np.clip(0.72 * trans + 0.88 * gfp, 0.0, 1.0)
    rgb[..., 0] = np.clip(0.72 * trans + 0.70 * mcherry, 0.0, 1.0)
    rgb[..., 2] = np.clip(0.72 * trans + 0.80 * mcherry, 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)


def merge_rgb_enhanced(trans: np.ndarray, gfp: np.ndarray, mcherry: np.ndarray) -> np.ndarray:
    """Create a stronger merged RGB image for fluorescence QC."""

    base = np.clip(0.55 * np.power(trans, 0.95), 0.0, 1.0)
    rgb = np.stack([base, base, base], axis=-1)
    rgb[..., 1] = np.clip(base + 1.00 * gfp, 0.0, 1.0)
    rgb[..., 0] = np.clip(base + 0.92 * mcherry, 0.0, 1.0)
    rgb[..., 2] = np.clip(base + 1.00 * mcherry, 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)


def make_contact_sheet(image_paths: list[Path], output_path: Path, columns: int = 6, thumb_size: int = 160) -> None:
    """Build a simple contact sheet from saved PNG images."""

    if not image_paths:
        return
    images = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(image, (thumb_size, thumb_size))
        canvas = Image.new("RGB", (thumb_size, thumb_size), color=(0, 0, 0))
        x = (thumb_size - thumb.size[0]) // 2
        y = (thumb_size - thumb.size[1]) // 2
        canvas.paste(thumb, (x, y))
        images.append(canvas)

    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size, rows * thumb_size), color=(8, 8, 8))
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        sheet.paste(image, (col * thumb_size, row * thumb_size))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def save_histogram(path: Path, values: np.ndarray, title: str, xlabel: str, bins: int = 40) -> None:
    """Save a simple histogram figure."""

    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, color="#4575b4", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_scatter(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    """Save a simple scatter plot."""

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x[mask], y[mask], s=18, alpha=0.45, color="#1b9e77", edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_rgb_png(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image


def _draw_nucleus_marker(draw: ImageDraw.ImageDraw, x: float, y: float, label: str) -> None:
    radius = 4
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(90, 255, 90), outline=(0, 60, 0))
    draw.text((x + 5, y - 6), label, fill=(120, 255, 120))


def _draw_nucleus_marker_style(draw: ImageDraw.ImageDraw, x: float, y: float, label: str, high_confidence: bool) -> None:
    radius = 4 if high_confidence else 5
    if high_confidence:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(90, 255, 90), outline=(0, 60, 0))
        text_color = (120, 255, 120)
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(150, 255, 150), width=2)
        text_color = (180, 255, 180)
    draw.text((x + 5, y - 6), label, fill=text_color)


def _draw_budneck_marker(draw: ImageDraw.ImageDraw, x: float, y: float, angle_deg: float, length: float, label: str) -> None:
    half = max(4.0, min(float(length) * 0.5, 12.0))
    angle_rad = np.deg2rad(float(angle_deg))
    dx = float(np.cos(angle_rad) * half)
    dy = float(np.sin(angle_rad) * half)
    draw.line((x - dx, y - dy, x + dx, y + dy), fill=(255, 80, 220), width=2)
    draw.text((x + 5, y + 2), label, fill=(255, 120, 235))


def save_rule_overlay(
    background_png: Path,
    output_path: Path,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    max_pair_lines: int = 8,
) -> None:
    """Overlay nuclei, bud-neck candidates, and high-score pair lines on an RGB background."""

    image = _load_rgb_png(background_png)
    draw = ImageDraw.Draw(image)

    for row in nuclei_df.itertuples(index=False):
        _draw_nucleus_marker(draw, float(row.centroid_x), float(row.centroid_y), f"N{int(row.nucleus_id)}")

    for row in budneck_df.itertuples(index=False):
        _draw_budneck_marker(
            draw,
            float(row.centroid_x),
            float(row.centroid_y),
            float(row.orientation_deg),
            float(row.major_axis_length),
            f"B{int(row.budneck_id)}",
        )

    if not pair_df.empty:
        ordered_pairs = pair_df.sort_values("preliminary_pair_score", ascending=False).head(max_pair_lines)
        nuclei_lookup = {
            int(row.nucleus_id): (float(row.centroid_x), float(row.centroid_y))
            for row in nuclei_df.itertuples(index=False)
        }
        for row in ordered_pairs.itertuples(index=False):
            if int(row.nucleus_a_id) not in nuclei_lookup or int(row.nucleus_b_id) not in nuclei_lookup:
                continue
            x1, y1 = nuclei_lookup[int(row.nucleus_a_id)]
            x2, y2 = nuclei_lookup[int(row.nucleus_b_id)]
            draw.line((x1, y1, x2, y2), fill=(245, 220, 100), width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_pair_patch(
    background_png: Path,
    output_path: Path,
    nucleus_points: list[tuple[float, float, int]],
    budneck_info: tuple[float, float, float, float, int] | None,
    patch_center_y: float,
    patch_center_x: float,
    patch_size: int,
    pair_line: tuple[tuple[float, float], tuple[float, float], tuple[int, int, int]] | None = None,
    class_label: str | None = None,
) -> None:
    """Crop a patch around a pair candidate and annotate it for manual review."""

    image = _load_rgb_png(background_png)
    half = patch_size // 2
    left = max(0, int(round(patch_center_x - half)))
    top = max(0, int(round(patch_center_y - half)))
    right = min(image.size[0], left + patch_size)
    bottom = min(image.size[1], top + patch_size)
    left = max(0, right - patch_size)
    top = max(0, bottom - patch_size)
    patch = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(patch)

    for x, y, nucleus_id in nucleus_points:
        _draw_nucleus_marker(draw, x - left, y - top, f"N{nucleus_id}")

    if pair_line is not None:
        start_yx, end_yx, color = pair_line
        x1, y1 = float(start_yx[1]) - left, float(start_yx[0]) - top
        x2, y2 = float(end_yx[1]) - left, float(end_yx[0]) - top
        draw.line((x1, y1, x2, y2), fill=color, width=2)

    if budneck_info is not None:
        x, y, angle_deg, length, budneck_id = budneck_info
        _draw_budneck_marker(draw, x - left, y - top, angle_deg, length, f"B{budneck_id}")

    if class_label:
        draw.text((6, 6), class_label, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(output_path)


def save_rules_v1_overlay(
    background_png: Path,
    output_path: Path,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> None:
    """Overlay rule-based pair classes without drawing final prompt boxes."""

    base = _load_rgb_png(background_png).convert("RGBA")
    line_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(line_overlay)

    nuclei_lookup = {
        int(row.nucleus_id): (float(row.centroid_x), float(row.centroid_y))
        for row in nuclei_df.itertuples(index=False)
    }
    class_order = {"reject_pair": 0, "review_pair": 1, "high_confidence_pair": 2}
    color_map = {
        "high_confidence_pair": (0, 255, 255, 235),
        "review_pair": (255, 165, 0, 210),
        "reject_pair": (170, 170, 170, 65),
    }
    width_map = {
        "high_confidence_pair": 3,
        "review_pair": 2,
        "reject_pair": 1,
    }

    if not pair_df.empty:
        ordered_pairs = pair_df.copy()
        ordered_pairs["_class_order"] = ordered_pairs["rule_class"].map(class_order).fillna(1)
        ordered_pairs = ordered_pairs.sort_values(
            ["_class_order", "preliminary_pair_score"],
            ascending=[True, True],
        )
        for row in ordered_pairs.itertuples(index=False):
            nucleus_a_id = int(row.nucleus_a_id)
            nucleus_b_id = int(row.nucleus_b_id)
            if nucleus_a_id not in nuclei_lookup or nucleus_b_id not in nuclei_lookup:
                continue
            x1, y1 = nuclei_lookup[nucleus_a_id]
            x2, y2 = nuclei_lookup[nucleus_b_id]
            color = color_map.get(str(row.rule_class), (220, 220, 220, 80))
            width = width_map.get(str(row.rule_class), 1)
            draw_lines.line((x1, y1, x2, y2), fill=color, width=width)

    merged = Image.alpha_composite(base, line_overlay)
    draw = ImageDraw.Draw(merged)
    for row in nuclei_df.itertuples(index=False):
        _draw_nucleus_marker(draw, float(row.centroid_x), float(row.centroid_y), f"N{int(row.nucleus_id)}")

    for row in budneck_df.itertuples(index=False):
        _draw_budneck_marker(
            draw,
            float(row.centroid_x),
            float(row.centroid_y),
            float(row.orientation_deg),
            float(row.major_axis_length),
            f"B{int(row.budneck_id)}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.convert("RGB").save(output_path)


def save_rules_v2_overlay(
    background_png: Path,
    output_path: Path,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> None:
    """Overlay rules_v2 classes without final prompt boxes."""

    base = _load_rgb_png(background_png).convert("RGBA")
    line_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(line_overlay)
    class_color = {
        "mother_bud_pair": (0, 255, 255, 235),
        "early_bud_pair": (255, 165, 0, 225),
        "single_cell": (220, 220, 220, 160),
    }

    nuclei_lookup = {
        int(row.nucleus_id): (float(row.centroid_x), float(row.centroid_y), bool(row.is_high_confidence))
        for row in nuclei_df.itertuples(index=False)
    }
    bud_lookup = {
        int(row.budneck_id): (float(row.centroid_x), float(row.centroid_y), float(row.orientation_deg), float(row.major_axis_length))
        for row in budneck_df.itertuples(index=False)
    }

    for row in classification_df.sort_values(["rule_class", "combined_support_score"], ascending=[True, False]).itertuples(index=False):
        class_name = str(row.rule_class)
        color = class_color.get(class_name, (220, 220, 220, 120))
        if pd.notna(row.nucleus_a_id) and int(row.nucleus_a_id) in nuclei_lookup:
            x1, y1, _ = nuclei_lookup[int(row.nucleus_a_id)]
            if pd.notna(row.nucleus_b_id) and int(row.nucleus_b_id) in nuclei_lookup:
                x2, y2, _ = nuclei_lookup[int(row.nucleus_b_id)]
                draw_lines.line((x1, y1, x2, y2), fill=color, width=3 if class_name == "mother_bud_pair" else 2)
                label_x = (x1 + x2) / 2.0
                label_y = (y1 + y2) / 2.0
            elif pd.notna(row.nearest_budneck_id) and int(row.nearest_budneck_id) in bud_lookup:
                x2, y2, _, _ = bud_lookup[int(row.nearest_budneck_id)]
                draw_lines.line((x1, y1, x2, y2), fill=color, width=2)
                label_x = (x1 + x2) / 2.0
                label_y = (y1 + y2) / 2.0
            else:
                label_x = x1 + 6
                label_y = y1 + 6
            draw_lines.text((label_x + 2, label_y + 2), class_name, fill=color)

    merged = Image.alpha_composite(base, line_overlay)
    draw = ImageDraw.Draw(merged)
    for row in nuclei_df.itertuples(index=False):
        _draw_nucleus_marker_style(
            draw,
            float(row.centroid_x),
            float(row.centroid_y),
            f"N{int(row.nucleus_id)}",
            bool(row.is_high_confidence),
        )
    for row in budneck_df.itertuples(index=False):
        _draw_budneck_marker(
            draw,
            float(row.centroid_x),
            float(row.centroid_y),
            float(row.orientation_deg),
            float(row.major_axis_length),
            f"B{int(row.budneck_id)}",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.convert("RGB").save(output_path)


def _draw_contour_from_json(draw: ImageDraw.ImageDraw, contour_json: str, color: tuple[int, int, int], width: int = 2) -> None:
    if not isinstance(contour_json, str) or not contour_json or contour_json == "[]":
        return
    try:
        points = json.loads(contour_json)
    except json.JSONDecodeError:
        return
    if not points or len(points) < 2:
        return
    flat = []
    for y, x in points:
        flat.append((float(x), float(y)))
    draw.line(flat + [flat[0]], fill=color, width=width)


def _draw_polygon(draw: ImageDraw.ImageDraw, corners: list[list[float]] | list[tuple[float, float]], color: tuple[int, int, int], width: int = 2, offset_xy: tuple[float, float] = (0.0, 0.0)) -> None:
    """Draw a polygon outline from y/x corners."""

    if not corners or len(corners) < 2:
        return
    ox, oy = float(offset_xy[0]), float(offset_xy[1])
    points = []
    for item in corners:
        y, x = item
        points.append((float(x) - ox, float(y) - oy))
    draw.line(points + [points[0]], fill=color, width=width)


def save_rules_v2_1_overlay(
    background_png: Path,
    output_path: Path,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    debug_df: pd.DataFrame,
) -> None:
    """Overlay rules_v2_1 classes, alternatives, and boundary support."""

    base = _load_rgb_png(background_png).convert("RGBA")
    line_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(line_overlay)
    draw = ImageDraw.Draw(base)

    nuclei_lookup = {
        int(row.nucleus_id): (float(row.centroid_x), float(row.centroid_y), bool(row.is_high_confidence))
        for row in nuclei_df.itertuples(index=False)
    }
    bud_lookup = {
        int(row.budneck_id): (float(row.centroid_x), float(row.centroid_y), float(row.orientation_deg), float(row.major_axis_length))
        for row in budneck_df.itertuples(index=False)
    }
    class_color = {
        "mother_bud_pair": (0, 255, 255, 235),
        "early_bud_pair": (255, 165, 0, 230),
        "uncertain_pair": (255, 235, 120, 220),
        "single_cell": (190, 190, 190, 110),
    }

    if not debug_df.empty:
        for row in debug_df.itertuples(index=False):
            alt_ids = []
            if isinstance(row.alternative_pair_ids, str) and row.alternative_pair_ids:
                alt_ids = [int(value) for value in row.alternative_pair_ids.split(";") if value.strip()]
            warning = bool(row.possible_wrong_far_pair_flag) or "conflict" in str(row.downgrade_reason) or "mutual_topk_failed" in str(row.downgrade_reason)
            if not warning:
                continue
            a_id = int(row.nucleus_a_id)
            b_id = int(row.nucleus_b_id)
            if a_id in nuclei_lookup and b_id in nuclei_lookup:
                x1, y1, _ = nuclei_lookup[a_id]
                x2, y2, _ = nuclei_lookup[b_id]
                draw_lines.line((x1, y1, x2, y2), fill=(160, 160, 160, 100), width=1)

    for row in classification_df.itertuples(index=False):
        class_name = str(row.final_class)
        color = class_color.get(class_name, (220, 220, 220, 120))
        _draw_contour_from_json(draw_lines, str(row.contour_yx_json), color[:3], width=2 if class_name != "single_cell" else 1)
        if pd.notna(row.nucleus_a_id) and int(row.nucleus_a_id) in nuclei_lookup:
            x1, y1, _ = nuclei_lookup[int(row.nucleus_a_id)]
            label_x = x1
            label_y = y1
            if pd.notna(row.nucleus_b_id) and int(row.nucleus_b_id) in nuclei_lookup:
                x2, y2, _ = nuclei_lookup[int(row.nucleus_b_id)]
                draw_lines.line((x1, y1, x2, y2), fill=color, width=3 if class_name == "mother_bud_pair" else 2)
                label_x = (x1 + x2) / 2.0
                label_y = (y1 + y2) / 2.0
            elif pd.notna(row.nearest_budneck_id) and int(row.nearest_budneck_id) in bud_lookup:
                x2, y2, _, _ = bud_lookup[int(row.nearest_budneck_id)]
                draw_lines.line((x1, y1, x2, y2), fill=color, width=2)
                label_x = (x1 + x2) / 2.0
                label_y = (y1 + y2) / 2.0
            draw_lines.text((label_x + 2, label_y + 2), class_name, fill=color)

            warnings = []
            if class_name == "mother_bud_pair":
                if not bool(row.is_mutual_top2):
                    warnings.append("not mutual top-2")
                if pd.notna(row.distance) and float(row.distance) > 35.0:
                    warnings.append("far pair")
                if float(row.cell_wall_connectedness_score) < 0.55:
                    warnings.append("weak wall support")
            if warnings:
                draw_lines.text((label_x + 2, label_y + 14), ",".join(warnings), fill=(255, 90, 90, 255))

    merged = Image.alpha_composite(base, line_overlay)
    draw_final = ImageDraw.Draw(merged)
    for row in nuclei_df.itertuples(index=False):
        _draw_nucleus_marker_style(
            draw_final,
            float(row.centroid_x),
            float(row.centroid_y),
            f"N{int(row.nucleus_id)}",
            bool(row.is_high_confidence),
        )
    for row in budneck_df.itertuples(index=False):
        _draw_budneck_marker(
            draw_final,
            float(row.centroid_x),
            float(row.centroid_y),
            float(row.orientation_deg),
            float(row.major_axis_length),
            f"B{int(row.budneck_id)}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.convert("RGB").save(output_path)


def save_rules_v2_3_overlay(
    background_png: Path,
    output_path: Path,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> None:
    """Overlay adjacency-aware v2.3 pair classes and warnings."""

    base = _load_rgb_png(background_png).convert("RGBA")
    line_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(line_overlay)

    nuclei_lookup = {
        int(row.nucleus_id): (float(row.centroid_x), float(row.centroid_y), bool(row.is_high_confidence))
        for row in nuclei_df.itertuples(index=False)
    }
    bud_lookup = {
        int(row.budneck_id): (float(row.centroid_x), float(row.centroid_y), float(row.orientation_deg), float(row.major_axis_length))
        for row in budneck_df.itertuples(index=False)
    }
    class_color = {
        "mother_bud_pair": (0, 255, 255, 235),
        "early_bud_pair": (255, 165, 0, 230),
        "uncertain_pair": (160, 160, 160, 120),
        "single_cell": (0, 0, 0, 0),
        "rejected_nonadjacent_pair": (255, 75, 75, 220),
    }
    class_width = {
        "mother_bud_pair": 3,
        "early_bud_pair": 2,
        "uncertain_pair": 1,
        "rejected_nonadjacent_pair": 2,
    }
    order_map = {
        "uncertain_pair": 0,
        "rejected_nonadjacent_pair": 1,
        "early_bud_pair": 2,
        "mother_bud_pair": 3,
    }
    ordered = classification_df.copy()
    ordered["_order"] = ordered["final_class"].map(order_map).fillna(0)
    ordered = ordered.sort_values(["_order", "pair_score"], ascending=[True, False])

    for row in ordered.itertuples(index=False):
        class_name = str(row.final_class)
        if class_name == "single_cell":
            continue
        _draw_contour_from_json(draw_lines, str(row.contour_yx_json), (255, 230, 70), width=2)
        color = class_color.get(class_name, (180, 180, 180, 120))
        width = class_width.get(class_name, 1)
        label_x = None
        label_y = None

        if pd.notna(row.nucleus_a_id) and int(row.nucleus_a_id) in nuclei_lookup:
            x1, y1, _ = nuclei_lookup[int(row.nucleus_a_id)]
            if pd.notna(row.nucleus_b_id) and int(row.nucleus_b_id) in nuclei_lookup:
                x2, y2, _ = nuclei_lookup[int(row.nucleus_b_id)]
                draw_lines.line((x1, y1, x2, y2), fill=color, width=width)
                label_x = (x1 + x2) / 2.0
                label_y = (y1 + y2) / 2.0
            elif pd.notna(row.budneck_id) and int(row.budneck_id) in bud_lookup:
                x2, y2, _, _ = bud_lookup[int(row.budneck_id)]
                draw_lines.line((x1, y1, x2, y2), fill=color, width=width)
                label_x = (x1 + x2) / 2.0
                label_y = (y1 + y2) / 2.0
            else:
                label_x = x1 + 6
                label_y = y1 + 6

        if label_x is None or label_y is None:
            continue
        draw_lines.text((label_x + 2, label_y + 2), class_name, fill=color)
        warnings = []
        if class_name == "rejected_nonadjacent_pair":
            warnings.append("not adjacent")
        if bool(row.wrong_far_assignment_warning):
            warnings.append("wrong far")
        if pd.notna(row.nucleus_line_hits_budneck) and not bool(row.nucleus_line_hits_budneck) and class_name != "single_cell":
            warnings.append("line misses neck")
        if pd.notna(row.cells_are_adjacent) and not bool(row.cells_are_adjacent) and class_name != "single_cell":
            warnings.append("no shared boundary")
        if warnings:
            draw_lines.text((label_x + 2, label_y + 14), ",".join(warnings), fill=(255, 90, 90, 255))

    merged = Image.alpha_composite(base, line_overlay)
    draw = ImageDraw.Draw(merged)
    for row in nuclei_df.itertuples(index=False):
        _draw_nucleus_marker_style(
            draw,
            float(row.centroid_x),
            float(row.centroid_y),
            f"N{int(row.nucleus_id)}",
            bool(row.is_high_confidence),
        )
    for row in budneck_df.itertuples(index=False):
        _draw_budneck_marker(
            draw,
            float(row.centroid_x),
            float(row.centroid_y),
            float(row.orientation_deg),
            float(row.major_axis_length),
            f"B{int(row.budneck_id)}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.convert("RGB").save(output_path)


def save_rules_v2_4_case_overlay(
    background_png: Path,
    output_path: Path,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    target_row: pd.Series,
    alternative_lines: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    patch_size: int = 128,
) -> None:
    """Save a patch overlay for one uncertain-case explanation target."""

    image = _load_rgb_png(background_png).convert("RGBA")
    width, height = image.size

    ys: list[float] = []
    xs: list[float] = []
    nucleus_rows = []
    for nucleus_col in ["nucleus_a_id", "nucleus_b_id"]:
        value = target_row.get(nucleus_col, np.nan)
        if pd.notna(value):
            match = nuclei_df[nuclei_df["nucleus_id"] == int(value)]
            if not match.empty:
                nucleus_rows.append(match.iloc[0])
                ys.append(float(match.iloc[0]["centroid_y"]))
                xs.append(float(match.iloc[0]["centroid_x"]))
    bud_row = None
    bud_id = target_row.get("budneck_id", np.nan)
    if pd.notna(bud_id):
        match = budneck_df[budneck_df["budneck_id"] == int(bud_id)]
        if not match.empty:
            bud_row = match.iloc[0]
            ys.append(float(bud_row["centroid_y"]))
            xs.append(float(bud_row["centroid_x"]))
    center_y = float(np.mean(ys)) if ys else height / 2.0
    center_x = float(np.mean(xs)) if xs else width / 2.0
    half = patch_size // 2
    left = max(0, int(round(center_x - half)))
    top = max(0, int(round(center_y - half)))
    right = min(width, left + patch_size)
    bottom = min(height, top + patch_size)
    left = max(0, right - patch_size)
    top = max(0, bottom - patch_size)
    patch = image.crop((left, top, right, bottom))

    line_overlay = Image.new("RGBA", patch.size, (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(line_overlay)
    draw = ImageDraw.Draw(patch)

    contour_json = str(target_row.get("contour_yx_json", "[]"))
    if contour_json and contour_json != "[]":
        try:
            points = json.loads(contour_json)
        except json.JSONDecodeError:
            points = []
        if points:
            contour = []
            for y, x in points:
                contour.append((float(x) - left, float(y) - top))
            if len(contour) >= 2:
                draw_lines.line(contour + [contour[0]], fill=(255, 230, 70, 220), width=2)

    if alternative_lines:
        for start_yx, end_yx in alternative_lines:
            draw_lines.line(
                (float(start_yx[1]) - left, float(start_yx[0]) - top, float(end_yx[1]) - left, float(end_yx[0]) - top),
                fill=(160, 160, 160, 150),
                width=1,
            )

    class_name = str(target_row.get("final_class", "uncertain_pair"))
    color = {
        "likely_reject": (255, 90, 90, 230),
        "likely_pair_review": (0, 255, 255, 235),
        "possible_early_bud_review": (255, 165, 0, 230),
        "manual_review": (255, 235, 120, 230),
    }.get(str(target_row.get("suggested_action", "manual_review")), (200, 200, 200, 200))

    if len(nucleus_rows) == 2:
        x1 = float(nucleus_rows[0]["centroid_x"]) - left
        y1 = float(nucleus_rows[0]["centroid_y"]) - top
        x2 = float(nucleus_rows[1]["centroid_x"]) - left
        y2 = float(nucleus_rows[1]["centroid_y"]) - top
        draw_lines.line((x1, y1, x2, y2), fill=color, width=3)
    elif len(nucleus_rows) == 1 and bud_row is not None:
        x1 = float(nucleus_rows[0]["centroid_x"]) - left
        y1 = float(nucleus_rows[0]["centroid_y"]) - top
        x2 = float(bud_row["centroid_x"]) - left
        y2 = float(bud_row["centroid_y"]) - top
        draw_lines.line((x1, y1, x2, y2), fill=color, width=2)

    merged = Image.alpha_composite(patch, line_overlay)
    draw_final = ImageDraw.Draw(merged)
    visible_nuclei = nuclei_df[
        (nuclei_df["centroid_x"] >= left)
        & (nuclei_df["centroid_x"] < right)
        & (nuclei_df["centroid_y"] >= top)
        & (nuclei_df["centroid_y"] < bottom)
    ]
    for row in visible_nuclei.itertuples(index=False):
        _draw_nucleus_marker_style(
            draw_final,
            float(row.centroid_x) - left,
            float(row.centroid_y) - top,
            f"N{int(row.nucleus_id)}",
            bool(row.is_high_confidence),
        )
    visible_buds = budneck_df[
        (budneck_df["centroid_x"] >= left)
        & (budneck_df["centroid_x"] < right)
        & (budneck_df["centroid_y"] >= top)
        & (budneck_df["centroid_y"] < bottom)
    ]
    for row in visible_buds.itertuples(index=False):
        _draw_budneck_marker(
            draw_final,
            float(row.centroid_x) - left,
            float(row.centroid_y) - top,
            float(row.orientation_deg),
            float(row.major_axis_length),
            f"B{int(row.budneck_id)}",
        )

    text_lines = [
        str(target_row.get("primary_uncertainty_reason", class_name)),
        str(target_row.get("suggested_action", "")),
        f"adj={bool(target_row.get('cells_are_adjacent', False))}",
        f"hit_neck={bool(target_row.get('nucleus_line_hits_budneck', False))}",
        f"between={bool(target_row.get('budneck_between_nuclei', False))}",
        f"mutual_top2={bool(target_row.get('is_mutual_top2', False))}",
    ]
    text_panel_height = max(44, 14 * len(text_lines) + 10)
    canvas = Image.new("RGBA", (merged.size[0], merged.size[1] + text_panel_height), (14, 14, 18, 255))
    canvas.paste(merged, (0, 0))
    draw_panel = ImageDraw.Draw(canvas)
    panel_top = merged.size[1]
    draw_panel.rectangle((0, panel_top, canvas.size[0], canvas.size[1]), fill=(18, 18, 24, 255))
    draw_panel.line((0, panel_top, canvas.size[0], panel_top), fill=(90, 90, 100, 255), width=1)

    y_text = panel_top + 6
    for line in text_lines:
        draw_panel.text((6, y_text), line, fill=(255, 255, 255))
        y_text += 12

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)


def save_draft_box_frame_overlay(
    background_png: Path,
    output_path: Path,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    uncertain_lines: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> None:
    """Overlay draft v2.3 review boxes on a background image."""

    base = _load_rgb_png(background_png).convert("RGBA")
    line_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(line_overlay)
    draw = ImageDraw.Draw(base)

    class_color = {
        "mother_bud_pair": (0, 255, 255, 230),
        "early_bud_pair": (255, 165, 0, 225),
        "single_cell": (90, 255, 120, 180),
    }
    if uncertain_lines:
        for start_yx, end_yx in uncertain_lines:
            draw_lines.line(
                (float(start_yx[1]), float(start_yx[0]), float(end_yx[1]), float(end_yx[0])),
                fill=(160, 160, 160, 110),
                width=1,
            )

    for row in boxes_df.itertuples(index=False):
        color = class_color.get(str(row.source_class), (220, 220, 220, 150))
        try:
            corners = json.loads(str(row.corner_yx_json))
        except json.JSONDecodeError:
            corners = []
        _draw_polygon(draw_lines, corners, color[:3], width=3 if str(row.source_class) != "single_cell" else 2)
        label = f"{str(row.source_class)} B{int(row.box_id)}"
        draw_lines.text((float(row.center_x) + 2, float(row.center_y) + 2), label, fill=color)

    merged = Image.alpha_composite(base, line_overlay)
    draw_final = ImageDraw.Draw(merged)
    for row in nuclei_df.itertuples(index=False):
        _draw_nucleus_marker_style(
            draw_final,
            float(row.centroid_x),
            float(row.centroid_y),
            f"N{int(row.nucleus_id)}",
            bool(row.is_high_confidence),
        )
    for row in budneck_df.itertuples(index=False):
        _draw_budneck_marker(
            draw_final,
            float(row.centroid_x),
            float(row.centroid_y),
            float(row.orientation_deg),
            float(row.major_axis_length),
            f"B{int(row.budneck_id)}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.convert("RGB").save(output_path)


def save_draft_box_patch(
    background_png: Path,
    output_path: Path,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    box_row: pd.Series,
    uncertain_lines: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    patch_size: int = 128,
) -> None:
    """Save a patch centered on one draft box with text in a footer panel."""

    image = _load_rgb_png(background_png).convert("RGBA")
    width, height = image.size
    center_y = float(box_row["center_y"])
    center_x = float(box_row["center_x"])
    half = patch_size // 2
    left = max(0, int(round(center_x - half)))
    top = max(0, int(round(center_y - half)))
    right = min(width, left + patch_size)
    bottom = min(height, top + patch_size)
    left = max(0, right - patch_size)
    top = max(0, bottom - patch_size)
    patch = image.crop((left, top, right, bottom))

    line_overlay = Image.new("RGBA", patch.size, (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(line_overlay)
    draw = ImageDraw.Draw(patch)

    try:
        corners = json.loads(str(box_row["corner_yx_json"]))
    except json.JSONDecodeError:
        corners = []
    color = {
        "mother_bud_pair": (0, 255, 255),
        "early_bud_pair": (255, 165, 0),
        "single_cell": (90, 255, 120),
    }.get(str(box_row["source_class"]), (200, 200, 200))
    _draw_polygon(draw_lines, corners, color, width=3 if str(box_row["source_class"]) != "single_cell" else 2, offset_xy=(left, top))

    if uncertain_lines:
        for start_yx, end_yx in uncertain_lines:
            draw_lines.line(
                (float(start_yx[1]) - left, float(start_yx[0]) - top, float(end_yx[1]) - left, float(end_yx[0]) - top),
                fill=(160, 160, 160, 120),
                width=1,
            )

    merged = Image.alpha_composite(patch, line_overlay)
    draw_final = ImageDraw.Draw(merged)
    visible_nuclei = nuclei_df[
        (nuclei_df["centroid_x"] >= left)
        & (nuclei_df["centroid_x"] < right)
        & (nuclei_df["centroid_y"] >= top)
        & (nuclei_df["centroid_y"] < bottom)
    ]
    for row in visible_nuclei.itertuples(index=False):
        _draw_nucleus_marker_style(
            draw_final,
            float(row.centroid_x) - left,
            float(row.centroid_y) - top,
            f"N{int(row.nucleus_id)}",
            bool(row.is_high_confidence),
        )
    visible_buds = budneck_df[
        (budneck_df["centroid_x"] >= left)
        & (budneck_df["centroid_x"] < right)
        & (budneck_df["centroid_y"] >= top)
        & (budneck_df["centroid_y"] < bottom)
    ]
    for row in visible_buds.itertuples(index=False):
        _draw_budneck_marker(
            draw_final,
            float(row.centroid_x) - left,
            float(row.centroid_y) - top,
            float(row.orientation_deg),
            float(row.major_axis_length),
            f"B{int(row.budneck_id)}",
        )

    text_lines = [
        f"{str(box_row['source_class'])} box {int(box_row['box_id'])}",
        f"quality={str(box_row['box_quality_flag'])}",
        f"angle={float(box_row['angle_deg']):.1f}",
        f"size=({float(box_row['width']):.1f},{float(box_row['height']):.1f})",
        str(box_row["box_reason"]),
    ]
    text_panel_height = max(44, 14 * len(text_lines) + 10)
    canvas = Image.new("RGBA", (merged.size[0], merged.size[1] + text_panel_height), (14, 14, 18, 255))
    canvas.paste(merged, (0, 0))
    draw_panel = ImageDraw.Draw(canvas)
    panel_top = merged.size[1]
    draw_panel.rectangle((0, panel_top, canvas.size[0], canvas.size[1]), fill=(18, 18, 24, 255))
    draw_panel.line((0, panel_top, canvas.size[0], panel_top), fill=(90, 90, 100, 255), width=1)
    y_text = panel_top + 6
    for line in text_lines:
        draw_panel.text((6, y_text), line, fill=(255, 255, 255))
        y_text += 12

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)
