"""Rule-discovery feature extraction for GFP, mCherry, and nucleus pairs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from skimage import filters, measure, morphology

from .image_utils import background_correct


@dataclass(frozen=True)
class ChannelParams:
    gaussian_sigma: float
    tophat_radius: int
    min_area: int
    max_area: int
    percentile_floor: float
    threshold_scale: float


CHANNEL_PARAMS = {
    "GFP": ChannelParams(
        gaussian_sigma=1.0,
        tophat_radius=9,
        min_area=3,
        max_area=220,
        percentile_floor=98.8,
        threshold_scale=0.34,
    ),
    "mCherry": ChannelParams(
        gaussian_sigma=1.0,
        tophat_radius=7,
        min_area=4,
        max_area=320,
        percentile_floor=98.5,
        threshold_scale=0.30,
    ),
}


def axis_angle_deg(dy: float, dx: float) -> float:
    """Return an orientation angle in [0, 180) for an axis-like vector."""

    return float(np.degrees(np.arctan2(dy, dx)) % 180.0)


def axis_difference_deg(angle_a: float, angle_b: float) -> float:
    """Return the minimum difference between two axis angles."""

    diff = abs((float(angle_a) - float(angle_b)) % 180.0)
    return float(min(diff, 180.0 - diff))


def region_orientation_deg(prop: measure._regionprops.RegionProperties) -> float:
    """Convert skimage orientation to an image-axis angle in degrees."""

    dy = math.cos(float(prop.orientation))
    dx = math.sin(float(prop.orientation))
    return axis_angle_deg(dy, dx)


def estimate_background(image: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 6) -> float:
    """Estimate local background intensity from a ring around the object bbox."""

    min_row, min_col, max_row, max_col = bbox
    row0 = max(0, min_row - margin)
    row1 = min(image.shape[0], max_row + margin)
    col0 = max(0, min_col - margin)
    col1 = min(image.shape[1], max_col + margin)
    patch = np.asarray(image[row0:row1, col0:col1], dtype=np.float32)
    if patch.size == 0:
        return float(np.median(image))
    core = np.zeros_like(patch, dtype=bool)
    core[(min_row - row0) : (max_row - row0), (min_col - col0) : (max_col - col0)] = True
    ring = patch[~core]
    if ring.size == 0:
        return float(np.median(patch))
    return float(np.median(ring))


def detect_channel_objects(
    image: np.ndarray,
    channel: str,
    condition: str,
    frame_index: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Detect fluorescence objects in a single frame for one channel."""

    params = CHANNEL_PARAMS[channel]
    corrected = background_correct(
        image,
        gaussian_sigma=params.gaussian_sigma,
        tophat_radius=params.tophat_radius,
    )
    positive = corrected[corrected > 0]
    if positive.size == 0:
        return [], np.zeros_like(image, dtype=np.int32)

    otsu = float(filters.threshold_otsu(corrected))
    percentile_term = float(np.percentile(positive, params.percentile_floor)) * params.threshold_scale
    threshold = max(otsu, percentile_term)
    binary = corrected > threshold
    binary = morphology.binary_opening(binary, morphology.disk(1))
    if channel == "mCherry":
        binary = morphology.binary_closing(binary, morphology.disk(1))
    binary = morphology.remove_small_objects(binary, params.min_area)
    binary = morphology.remove_small_holes(binary, area_threshold=max(params.min_area * 2, 8))

    labeled = measure.label(binary)
    rows: list[dict[str, object]] = []
    final_labels = np.zeros_like(labeled, dtype=np.int32)
    next_id = 1
    for prop in measure.regionprops(labeled, intensity_image=np.asarray(image, dtype=np.float32)):
        if prop.area < params.min_area or prop.area > params.max_area:
            continue
        minor_axis = float(prop.minor_axis_length) if prop.minor_axis_length > 0 else 1.0
        major_axis = float(prop.major_axis_length) if prop.major_axis_length > 0 else 0.0
        background = estimate_background(np.asarray(image, dtype=np.float32), prop.bbox, margin=6)
        signal_to_background = float((float(prop.max_intensity) + 1.0) / (background + 1.0))
        final_labels[labeled == prop.label] = next_id
        rows.append(
            {
                "condition": condition,
                "frame": frame_index,
                "channel": channel,
                "object_id": next_id,
                "centroid_y": float(prop.centroid[0]),
                "centroid_x": float(prop.centroid[1]),
                "area": int(prop.area),
                "mean_intensity": float(prop.mean_intensity),
                "max_intensity": float(prop.max_intensity),
                "total_intensity": float(prop.intensity_image.sum()),
                "major_axis_length": major_axis,
                "minor_axis_length": minor_axis,
                "aspect_ratio": float(major_axis / max(minor_axis, 1e-6)),
                "orientation_deg": region_orientation_deg(prop),
                "signal_to_background": signal_to_background,
            }
        )
        next_id += 1
    return rows, final_labels


def build_nucleus_candidates(gfp_objects: pd.DataFrame) -> pd.DataFrame:
    """Convert GFP objects into nucleus-like candidates."""

    if gfp_objects.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "frame",
                "nucleus_id",
                "centroid_y",
                "centroid_x",
                "area",
                "max_intensity",
                "mean_intensity",
                "signal_to_background",
                "is_high_confidence",
            ]
        )

    rows: list[dict[str, object]] = []
    for (condition, frame), group in gfp_objects.groupby(["condition", "frame"], sort=True):
        max_cutoff = float(group["max_intensity"].quantile(0.60))
        sb_cutoff = float(group["signal_to_background"].quantile(0.55))
        for nucleus_id, row in enumerate(group.sort_values("max_intensity", ascending=False).itertuples(index=False), start=1):
            is_high_confidence = bool(
                row.max_intensity >= max_cutoff
                and row.signal_to_background >= sb_cutoff
                and row.area >= 3
            )
            rows.append(
                {
                    "condition": condition,
                    "frame": int(frame),
                    "nucleus_id": nucleus_id,
                    "centroid_y": float(row.centroid_y),
                    "centroid_x": float(row.centroid_x),
                    "area": int(row.area),
                    "max_intensity": float(row.max_intensity),
                    "mean_intensity": float(row.mean_intensity),
                    "signal_to_background": float(row.signal_to_background),
                    "is_high_confidence": is_high_confidence,
                }
            )
    return pd.DataFrame(rows)


def build_budneck_candidates(mcherry_objects: pd.DataFrame) -> pd.DataFrame:
    """Convert mCherry objects into bud-neck-like candidates."""

    if mcherry_objects.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "frame",
                "budneck_id",
                "centroid_y",
                "centroid_x",
                "area",
                "max_intensity",
                "mean_intensity",
                "major_axis_length",
                "minor_axis_length",
                "aspect_ratio",
                "orientation_deg",
                "signal_to_background",
            ]
        )

    rows: list[dict[str, object]] = []
    for (condition, frame), group in mcherry_objects.groupby(["condition", "frame"], sort=True):
        ordered = group.sort_values(
            ["aspect_ratio", "signal_to_background", "max_intensity"],
            ascending=[False, False, False],
        )
        for budneck_id, row in enumerate(ordered.itertuples(index=False), start=1):
            rows.append(
                {
                    "condition": condition,
                    "frame": int(frame),
                    "budneck_id": budneck_id,
                    "centroid_y": float(row.centroid_y),
                    "centroid_x": float(row.centroid_x),
                    "area": int(row.area),
                    "max_intensity": float(row.max_intensity),
                    "mean_intensity": float(row.mean_intensity),
                    "major_axis_length": float(row.major_axis_length),
                    "minor_axis_length": float(row.minor_axis_length),
                    "aspect_ratio": float(row.aspect_ratio),
                    "orientation_deg": float(row.orientation_deg),
                    "signal_to_background": float(row.signal_to_background),
                }
            )
    return pd.DataFrame(rows)


def distance_point_to_segment(
    point_yx: tuple[float, float],
    start_yx: tuple[float, float],
    end_yx: tuple[float, float],
) -> tuple[float, float]:
    """Distance from a point to a line segment and fractional projection along it."""

    p = np.asarray(point_yx, dtype=np.float32)
    a = np.asarray(start_yx, dtype=np.float32)
    b = np.asarray(end_yx, dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 0:
        return float(np.linalg.norm(p - a)), 0.0
    t = float(np.dot(p - a, ab) / denom)
    t_clamped = min(max(t, 0.0), 1.0)
    projection = a + t_clamped * ab
    return float(np.linalg.norm(p - projection)), t


def preliminary_pair_score(
    distance: float,
    budneck_between_nuclei: bool,
    budneck_distance_to_line: float | None,
    angle_difference_from_perpendicular: float | None,
    budneck_aspect_ratio: float | None,
    budneck_signal_to_background: float | None,
) -> float:
    """Combine interpretable terms into a preliminary pair score."""

    if budneck_distance_to_line is None:
        return 0.0

    line_scale = max(4.0, 0.18 * distance)
    distance_score = max(0.0, 1.0 - (distance - 12.0) / 36.0)
    between_score = 1.0 if budneck_between_nuclei else 0.0
    line_score = math.exp(-((budneck_distance_to_line / line_scale) ** 2))
    angle_score = max(0.0, 1.0 - (float(angle_difference_from_perpendicular) / 45.0))
    aspect_score = min(1.0, max(0.0, (float(budneck_aspect_ratio) - 1.2) / 2.0))
    signal_score = min(1.0, float(budneck_signal_to_background) / 4.0)
    score = (
        0.15 * distance_score
        + 0.25 * between_score
        + 0.25 * line_score
        + 0.20 * angle_score
        + 0.10 * aspect_score
        + 0.05 * signal_score
    )
    return float(max(0.0, min(score, 1.0)))


def build_pair_candidates(
    nucleus_candidates: pd.DataFrame,
    budneck_candidates: pd.DataFrame,
    pair_distance_min: float,
    pair_distance_max: float,
) -> pd.DataFrame:
    """Build nearby nucleus-pair candidates with bud-neck relationship features."""

    columns = [
        "condition",
        "frame",
        "nucleus_a_id",
        "nucleus_b_id",
        "distance",
        "nuclei_line_angle_deg",
        "nearest_budneck_id",
        "budneck_between_nuclei",
        "budneck_distance_to_line",
        "budneck_angle_deg",
        "angle_difference_from_perpendicular",
        "budneck_aspect_ratio",
        "budneck_signal_to_background",
        "preliminary_pair_score",
    ]
    if nucleus_candidates.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    grouped_nuclei = nucleus_candidates.groupby(["condition", "frame"], sort=True)
    budneck_groups = {
        (condition, int(frame)): group.copy()
        for (condition, frame), group in budneck_candidates.groupby(["condition", "frame"], sort=True)
    }

    for (condition, frame), nuclei_group in grouped_nuclei:
        nuclei_rows = list(nuclei_group.sort_values("nucleus_id").itertuples(index=False))
        bud_group = budneck_groups.get((condition, int(frame)), pd.DataFrame())
        bud_rows = list(bud_group.itertuples(index=False))
        for idx, nucleus_a in enumerate(nuclei_rows):
            for nucleus_b in nuclei_rows[idx + 1 :]:
                dy = float(nucleus_b.centroid_y - nucleus_a.centroid_y)
                dx = float(nucleus_b.centroid_x - nucleus_a.centroid_x)
                distance = float(math.hypot(dy, dx))
                if distance < pair_distance_min or distance > pair_distance_max:
                    continue

                line_angle = axis_angle_deg(dy, dx)
                best_row: dict[str, object] | None = None
                best_score = -1.0
                for bud in bud_rows:
                    dist_to_line, t = distance_point_to_segment(
                        (float(bud.centroid_y), float(bud.centroid_x)),
                        (float(nucleus_a.centroid_y), float(nucleus_a.centroid_x)),
                        (float(nucleus_b.centroid_y), float(nucleus_b.centroid_x)),
                    )
                    between = bool(0.10 <= t <= 0.90)
                    angle_diff = axis_difference_deg(float(bud.orientation_deg), (line_angle + 90.0) % 180.0)
                    score = preliminary_pair_score(
                        distance=distance,
                        budneck_between_nuclei=between,
                        budneck_distance_to_line=dist_to_line,
                        angle_difference_from_perpendicular=angle_diff,
                        budneck_aspect_ratio=float(bud.aspect_ratio),
                        budneck_signal_to_background=float(bud.signal_to_background),
                    )
                    if score > best_score:
                        best_score = score
                        best_row = {
                            "nearest_budneck_id": int(bud.budneck_id),
                            "budneck_between_nuclei": between,
                            "budneck_distance_to_line": float(dist_to_line),
                            "budneck_angle_deg": float(bud.orientation_deg),
                            "angle_difference_from_perpendicular": float(angle_diff),
                            "budneck_aspect_ratio": float(bud.aspect_ratio),
                            "budneck_signal_to_background": float(bud.signal_to_background),
                            "preliminary_pair_score": float(score),
                        }

                if best_row is None:
                    best_row = {
                        "nearest_budneck_id": np.nan,
                        "budneck_between_nuclei": False,
                        "budneck_distance_to_line": np.nan,
                        "budneck_angle_deg": np.nan,
                        "angle_difference_from_perpendicular": np.nan,
                        "budneck_aspect_ratio": np.nan,
                        "budneck_signal_to_background": np.nan,
                        "preliminary_pair_score": 0.0,
                    }

                rows.append(
                    {
                        "condition": condition,
                        "frame": int(frame),
                        "nucleus_a_id": int(nucleus_a.nucleus_id),
                        "nucleus_b_id": int(nucleus_b.nucleus_id),
                        "distance": distance,
                        "nuclei_line_angle_deg": float(line_angle),
                        **best_row,
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def suggest_thresholds(
    gfp_objects: pd.DataFrame,
    budneck_candidates: pd.DataFrame,
    pair_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Propose simple rule thresholds from empirical distributions."""

    rows: list[dict[str, object]] = []

    if not gfp_objects.empty:
        rows.append(
            {
                "feature": "gfp_area_min",
                "suggested_min": float(gfp_objects["area"].quantile(0.10)),
                "suggested_max": float(gfp_objects["area"].quantile(0.95)),
                "suggested_value": float(gfp_objects["area"].quantile(0.10)),
                "basis": "10th and 95th percentiles of GFP object areas",
            }
        )
        rows.append(
            {
                "feature": "gfp_max_intensity_min",
                "suggested_min": float(gfp_objects["max_intensity"].quantile(0.25)),
                "suggested_max": float(gfp_objects["max_intensity"].quantile(0.95)),
                "suggested_value": float(gfp_objects["max_intensity"].quantile(0.25)),
                "basis": "25th percentile of GFP object max intensity",
            }
        )

    if not budneck_candidates.empty:
        rows.append(
            {
                "feature": "mcherry_aspect_ratio_min",
                "suggested_min": float(budneck_candidates["aspect_ratio"].quantile(0.65)),
                "suggested_max": float(budneck_candidates["aspect_ratio"].quantile(0.98)),
                "suggested_value": float(budneck_candidates["aspect_ratio"].quantile(0.65)),
                "basis": "65th percentile of mCherry aspect ratio for elongated structures",
            }
        )
        rows.append(
            {
                "feature": "mcherry_signal_to_background_min",
                "suggested_min": float(budneck_candidates["signal_to_background"].quantile(0.50)),
                "suggested_max": float(budneck_candidates["signal_to_background"].quantile(0.95)),
                "suggested_value": float(budneck_candidates["signal_to_background"].quantile(0.50)),
                "basis": "Median mCherry signal-to-background for candidate filtering",
            }
        )

    if not pair_candidates.empty:
        high_score = pair_candidates[pair_candidates["preliminary_pair_score"] >= pair_candidates["preliminary_pair_score"].quantile(0.80)]
        source = high_score if not high_score.empty else pair_candidates
        rows.extend(
            [
                {
                    "feature": "nuclei_distance_min",
                    "suggested_min": float(source["distance"].quantile(0.10)),
                    "suggested_max": float(source["distance"].quantile(0.90)),
                    "suggested_value": float(source["distance"].quantile(0.10)),
                    "basis": "10th percentile of nearby nucleus distances among top-scoring pairs",
                },
                {
                    "feature": "nuclei_distance_max",
                    "suggested_min": float(source["distance"].quantile(0.10)),
                    "suggested_max": float(source["distance"].quantile(0.90)),
                    "suggested_value": float(source["distance"].quantile(0.90)),
                    "basis": "90th percentile of nearby nucleus distances among top-scoring pairs",
                },
                {
                    "feature": "angle_tolerance_deg",
                    "suggested_min": 0.0,
                    "suggested_max": float(source["angle_difference_from_perpendicular"].quantile(0.90)),
                    "suggested_value": float(source["angle_difference_from_perpendicular"].quantile(0.75)),
                    "basis": "75th percentile of deviation from perpendicular among top-scoring pairs",
                },
                {
                    "feature": "budneck_score_cutoff",
                    "suggested_min": float(pair_candidates["preliminary_pair_score"].quantile(0.70)),
                    "suggested_max": 1.0,
                    "suggested_value": float(pair_candidates["preliminary_pair_score"].quantile(0.80)),
                    "basis": "80th percentile of preliminary pair score",
                },
            ]
        )

    return pd.DataFrame(rows)

