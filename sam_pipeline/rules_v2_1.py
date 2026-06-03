"""Rules v2.1 with explicit Trans boundary features and partner diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation

from .image_utils import background_correct, percentile_normalize
from .rule_features import axis_angle_deg, axis_difference_deg, distance_point_to_segment, estimate_background
from .rules_v2 import (
    _count_lobes,
    _extract_patch,
    _merge_candidate_points,
    _select_local_mask,
    _watershed_two_compartments,
    nearest_budneck_for_nucleus,
)


def write_existing_feature_report(output_path: Path) -> None:
    """Write an explicit report of which Trans features rules_v2 already used."""

    text = """rules_v2 existing Trans morphology feature report

Previously used in rules_v2:
- local Trans patch thresholding with dark/bright polarity search
- selected local Trans component mask around seed nuclei and optional bud neck
- mask_area
- contour_point_count
- lobe_count
- neck_constriction_strength
- bud_area
- mother_area
- bud_to_mother_area_ratio
- boundary_connectedness
- has_small_attached_bud
- mask_polarity
- morphology_support_score

What this means biologically:
- rules_v2 did use Trans information, but mainly through generic local mask geometry
- it did not compute an explicit cell-wall graph, explicit boundary continuity, or a direct wall-supported same-component test for pair assignment
- contour_point_count was only a contour size summary, not a true cell-wall strength or continuity measurement
- same_trans_component was not saved explicitly
- boundary_strength_around_each_cell was not measured explicitly
- neck_constriction_width was not saved explicitly
- wall_continuity_score was not saved explicitly
- boundary_gap_between_nuclei was not measured explicitly

Conclusion:
- in rules_v2, Trans morphology was represented indirectly through generic morphology features
- explicit cell-wall / boundary evidence was not a first-class feature in the pair assignment logic
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _sample_line_points(start_yx: tuple[float, float], end_yx: tuple[float, float], n_samples: int = 64) -> np.ndarray:
    ys = np.linspace(float(start_yx[0]), float(end_yx[0]), n_samples)
    xs = np.linspace(float(start_yx[1]), float(end_yx[1]), n_samples)
    return np.stack([ys, xs], axis=1)


def _sample_image_nearest(image: np.ndarray, points_yx: np.ndarray) -> np.ndarray:
    ys = np.clip(np.rint(points_yx[:, 0]).astype(int), 0, image.shape[0] - 1)
    xs = np.clip(np.rint(points_yx[:, 1]).astype(int), 0, image.shape[1] - 1)
    return np.asarray(image[ys, xs], dtype=np.float32)


def _ring_mean(image: np.ndarray, center_yx: tuple[float, float], inner_radius: float = 5.0, outer_radius: float = 11.0) -> float:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    cy, cx = float(center_yx[0]), float(center_yx[1])
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    ring = (dist2 >= inner_radius**2) & (dist2 <= outer_radius**2)
    if not np.any(ring):
        return 0.0
    return float(np.mean(np.asarray(image[ring], dtype=np.float32)))


def _contour_json(mask: np.ndarray, offset_yx: tuple[int, int]) -> str:
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return "[]"
    contour = max(contours, key=len)
    step = max(1, len(contour) // 60)
    sampled = contour[::step]
    points = [[float(y + offset_yx[0]), float(x + offset_yx[1])] for y, x in sampled]
    return json.dumps(points)


def _choose_pair_component(mask: np.ndarray, seeds_patch: list[tuple[float, float]]) -> tuple[np.ndarray, bool]:
    labeled = measure.label(mask)
    if labeled.max() == 0 or len(seeds_patch) < 2:
        return np.zeros_like(mask, dtype=bool), False
    labels = []
    for sy, sx in seeds_patch[:2]:
        iy = int(np.clip(round(sy), 0, mask.shape[0] - 1))
        ix = int(np.clip(round(sx), 0, mask.shape[1] - 1))
        labels.append(int(labeled[iy, ix]))
    label_a, label_b = labels
    if label_a > 0 and label_a == label_b:
        return labeled == label_a, True
    if label_a > 0 or label_b > 0:
        comp = np.zeros_like(mask, dtype=bool)
        if label_a > 0:
            comp |= labeled == label_a
        if label_b > 0:
            comp |= labeled == label_b
        return comp, False
    return np.zeros_like(mask, dtype=bool), False


def _compute_width_profile_explicit(
    mask: np.ndarray,
    center_yx: tuple[float, float],
    axis_angle_deg_value: float,
    neck_yx: tuple[float, float] | None,
) -> tuple[float, float, float, float]:
    coords = np.argwhere(mask)
    if coords.shape[0] < 10:
        return 0.0, 0.0, 0.0, 0.0
    center = np.asarray(center_yx, dtype=np.float32)
    theta = math.radians(axis_angle_deg_value)
    axis_vec = np.asarray([math.sin(theta), math.cos(theta)], dtype=np.float32)
    perp_vec = np.asarray([math.cos(theta), -math.sin(theta)], dtype=np.float32)
    rel = coords.astype(np.float32) - center[None, :]
    longitudinal = rel @ axis_vec
    transverse = rel @ perp_vec
    bins = np.linspace(float(longitudinal.min()), float(longitudinal.max()), 19)
    widths: list[float] = []
    mids: list[float] = []
    for start, end in zip(bins[:-1], bins[1:]):
        use = (longitudinal >= start) & (longitudinal < end)
        if int(use.sum()) < 3:
            continue
        width = float(transverse[use].max() - transverse[use].min() + 1.0)
        widths.append(width)
        mids.append((start + end) / 2.0)
    if len(widths) < 3:
        return 0.0, 0.0, 0.0, 0.0
    widths_arr = np.asarray(widths, dtype=np.float32)
    mids_arr = np.asarray(mids, dtype=np.float32)
    if neck_yx is None:
        neck_position = 0.0
    else:
        neck_rel = np.asarray(neck_yx, dtype=np.float32) - center
        neck_position = float(neck_rel @ axis_vec)
    neck_index = int(np.argmin(np.abs(mids_arr - neck_position)))
    neck_width = float(widths_arr[neck_index])
    left_mean = float(np.mean(widths_arr[: max(1, len(widths_arr) // 4)]))
    right_mean = float(np.mean(widths_arr[-max(1, len(widths_arr) // 4) :]))
    reference_width = max(1.0, (left_mean + right_mean) / 2.0)
    constriction = float(np.clip(1.0 - neck_width / reference_width, 0.0, 1.0))
    connectedness = float(np.clip(1.0 - abs(neck_width - reference_width * 0.55) / max(reference_width, 1.0), 0.0, 1.0))
    return neck_width, reference_width, constriction, connectedness


def _distance_score(distance: float, ideal_min: float, ideal_max: float, max_distance: float) -> float:
    if distance < ideal_min:
        return float(np.clip(distance / ideal_min, 0.0, 1.0))
    if distance <= ideal_max:
        return 1.0
    if distance >= max_distance:
        return 0.0
    return float(np.clip(1.0 - (distance - ideal_max) / (max_distance - ideal_max), 0.0, 1.0))


def _budneck_score_from_pair_row(row: pd.Series) -> float:
    distance = float(row["distance"])
    line_distance = float(row["budneck_distance_to_line"]) if pd.notna(row["budneck_distance_to_line"]) else np.inf
    angle_diff = float(row["angle_difference_from_perpendicular"]) if pd.notna(row["angle_difference_from_perpendicular"]) else 180.0
    signal = float(row["budneck_signal_to_background"]) if pd.notna(row["budneck_signal_to_background"]) else 0.0
    aspect = float(row["budneck_aspect_ratio"]) if pd.notna(row["budneck_aspect_ratio"]) else 1.0
    projection = float(row["budneck_projection_fraction"]) if pd.notna(row["budneck_projection_fraction"]) else -1.0
    between = 1.0 if bool(row["budneck_between_nuclei"]) else 0.0
    line_score = max(0.0, 1.0 - line_distance / max(6.0, 0.18 * distance))
    angle_score = max(0.0, 1.0 - angle_diff / 55.0)
    signal_score = np.clip((signal - 1.2) / 2.5, 0.0, 1.0)
    aspect_score = np.clip((aspect - 1.15) / 1.8, 0.0, 1.0)
    if 0.0 <= projection <= 1.0:
        projection_score = max(0.0, 1.0 - abs(projection - 0.5) / 0.5)
    else:
        projection_score = 0.0
    score = (
        0.20 * float(row["preliminary_pair_score"])
        + 0.18 * between
        + 0.18 * line_score
        + 0.16 * angle_score
        + 0.14 * signal_score
        + 0.08 * aspect_score
        + 0.06 * projection_score
    )
    return float(np.clip(score, 0.0, 1.0))


def extract_pair_cell_wall_features(
    candidate_id: int,
    trans_frame: np.ndarray,
    nucleus_a_yx: tuple[float, float],
    nucleus_b_yx: tuple[float, float],
    budneck_row: pd.Series | None,
    distance: float,
) -> dict[str, object]:
    """Extract explicit Trans boundary features for one nucleus-pair candidate."""

    axis_angle = axis_angle_deg(nucleus_b_yx[0] - nucleus_a_yx[0], nucleus_b_yx[1] - nucleus_a_yx[1])
    midpoint_y = (float(nucleus_a_yx[0]) + float(nucleus_b_yx[0])) / 2.0
    midpoint_x = (float(nucleus_a_yx[1]) + float(nucleus_b_yx[1])) / 2.0
    budneck_yx = None if budneck_row is None else (float(budneck_row["centroid_y"]), float(budneck_row["centroid_x"]))
    if budneck_yx is not None:
        center_y = 0.55 * midpoint_y + 0.45 * budneck_yx[0]
        center_x = 0.55 * midpoint_x + 0.45 * budneck_yx[1]
    else:
        center_y, center_x = midpoint_y, midpoint_x
    patch_half = int(np.clip(distance * 0.75 + 20.0, 28.0, 60.0))
    patch, (y0, x0) = _extract_patch(trans_frame, center_y, center_x, half_size=patch_half)
    seeds_patch = [(float(nucleus_a_yx[0] - y0), float(nucleus_a_yx[1] - x0)), (float(nucleus_b_yx[0] - y0), float(nucleus_b_yx[1] - x0))]
    bud_patch = None if budneck_yx is None else (float(budneck_yx[0] - y0), float(budneck_yx[1] - x0))

    norm = percentile_normalize(patch, low=1.0, high=99.6)
    smooth = filters.gaussian(norm, sigma=1.0, preserve_range=True)
    inv = 1.0 - smooth
    edge_strength = np.maximum(filters.sobel(smooth), filters.sobel(inv))
    mask, polarity = _select_local_mask(patch, seeds_patch + ([] if bud_patch is None else [bud_patch]))
    working_mask, same_component = _choose_pair_component(mask, seeds_patch)
    if working_mask.sum() == 0:
        working_mask = mask
    contour_json = _contour_json(working_mask, (y0, x0))

    component_fraction = 0.0
    line_points_local = _sample_line_points((nucleus_a_yx[0] - y0, nucleus_a_yx[1] - x0), (nucleus_b_yx[0] - y0, nucleus_b_yx[1] - x0), n_samples=72)
    line_inside = _sample_image_nearest(working_mask.astype(np.float32), line_points_local)
    if line_inside.size > 0:
        component_fraction = float(np.mean(line_inside))
    boundary_gap = float(np.clip(1.0 - component_fraction, 0.0, 1.0))

    boundary_strength_a = _ring_mean(edge_strength, seeds_patch[0], inner_radius=5.0, outer_radius=11.0)
    boundary_strength_b = _ring_mean(edge_strength, seeds_patch[1], inner_radius=5.0, outer_radius=11.0)
    boundary_strength_mean = float((boundary_strength_a + boundary_strength_b) / 2.0)

    perimeter = segmentation.find_boundaries(working_mask, mode="outer")
    if np.any(perimeter):
        wall_strength_raw = float(np.mean(edge_strength[perimeter]))
        denom = float(np.percentile(edge_strength, 95.0)) + 1e-6
        wall_continuity_score = float(np.clip(wall_strength_raw / denom, 0.0, 1.0))
    else:
        wall_continuity_score = 0.0

    if budneck_yx is not None:
        bud_line_distance, bud_t = distance_point_to_segment(budneck_yx, nucleus_a_yx, nucleus_b_yx)
        midpoint_distance = float(math.hypot(budneck_yx[0] - midpoint_y, budneck_yx[1] - midpoint_x))
        off_segment = bool(bud_t < 0.05 or bud_t > 0.95)
    else:
        bud_line_distance, bud_t, midpoint_distance, off_segment = np.nan, np.nan, np.nan, True

    neck_width, reference_width, neck_constriction_score, connectedness = _compute_width_profile_explicit(
        working_mask,
        center_yx=((midpoint_y if budneck_yx is None else budneck_yx[0]) - y0, (midpoint_x if budneck_yx is None else budneck_yx[1]) - x0),
        axis_angle_deg_value=axis_angle,
        neck_yx=bud_patch,
    )
    lobe_count = _count_lobes(working_mask)
    mother_area, bud_area = _watershed_two_compartments(working_mask, seeds_patch[:2])
    area_ratio = float(bud_area / max(mother_area, 1)) if mother_area > 0 else 0.0
    bud_lobe_detected = bool(
        same_component
        and bud_area >= 10
        and mother_area >= 35
        and 0.04 <= area_ratio <= 0.90
        and lobe_count >= 2
    )
    cell_wall_connectedness_score = float(
        np.clip(
            0.24 * (1.0 if same_component else 0.0)
            + 0.22 * component_fraction
            + 0.18 * np.clip(boundary_strength_mean / 0.12, 0.0, 1.0)
            + 0.18 * wall_continuity_score
            + 0.18 * connectedness,
            0.0,
            1.0,
        )
    )

    return {
        "candidate_id": candidate_id,
        "same_trans_component": bool(same_component),
        "cell_wall_connectedness_score": cell_wall_connectedness_score,
        "boundary_strength_around_each_cell": boundary_strength_mean,
        "boundary_strength_cell_a": boundary_strength_a,
        "boundary_strength_cell_b": boundary_strength_b,
        "neck_constriction_width": neck_width,
        "neck_constriction_score": neck_constriction_score,
        "bud_lobe_detected": bud_lobe_detected,
        "mother_bud_area_ratio": area_ratio,
        "wall_continuity_score": wall_continuity_score,
        "boundary_gap_between_nuclei": boundary_gap,
        "line_inside_component_fraction": component_fraction,
        "budneck_projection_fraction": float(bud_t) if not pd.isna(bud_t) else np.nan,
        "budneck_midpoint_distance": midpoint_distance,
        "budneck_off_segment_flag": bool(off_segment),
        "mask_area": int(working_mask.sum()),
        "lobe_count": int(lobe_count),
        "bud_area": int(bud_area),
        "mother_area": int(mother_area),
        "mask_polarity": polarity,
        "mask_bbox_y0": int(y0),
        "mask_bbox_x0": int(x0),
        "mask_bbox_y1": int(y0 + patch.shape[0]),
        "mask_bbox_x1": int(x0 + patch.shape[1]),
        "contour_yx_json": contour_json,
    }


def build_cell_wall_features(
    trans_stacks: dict[str, np.ndarray],
    nuclei_v2_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    pair_v2_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute explicit cell-wall feature table for every pair candidate."""

    nuclei_groups = {
        (condition, int(frame)): group.copy()
        for (condition, frame), group in nuclei_v2_df.groupby(["condition", "frame"], sort=True)
    }
    bud_groups = {
        (condition, int(frame)): group.copy()
        for (condition, frame), group in budneck_df.groupby(["condition", "frame"], sort=True)
    }

    rows: list[dict[str, object]] = []
    candidate_id = 1
    for pair_row in pair_v2_df.itertuples(index=False):
        condition = str(pair_row.condition)
        frame = int(pair_row.frame)
        nuclei_frame = nuclei_groups.get((condition, frame), pd.DataFrame())
        trans_frame = np.asarray(trans_stacks[condition][frame], dtype=np.float32)
        if nuclei_frame.empty:
            continue
        a_match = nuclei_frame[nuclei_frame["nucleus_id"] == int(pair_row.nucleus_a_id)]
        b_match = nuclei_frame[nuclei_frame["nucleus_id"] == int(pair_row.nucleus_b_id)]
        if a_match.empty or b_match.empty:
            continue
        nuc_a = (float(a_match.iloc[0]["centroid_y"]), float(a_match.iloc[0]["centroid_x"]))
        nuc_b = (float(b_match.iloc[0]["centroid_y"]), float(b_match.iloc[0]["centroid_x"]))
        bud_frame = bud_groups.get((condition, frame), pd.DataFrame())
        bud_match = None
        if pd.notna(pair_row.nearest_budneck_id) and not bud_frame.empty:
            matched = bud_frame[bud_frame["budneck_id"] == int(pair_row.nearest_budneck_id)]
            if not matched.empty:
                bud_match = matched.iloc[0]
        feature_row = extract_pair_cell_wall_features(
            candidate_id=candidate_id,
            trans_frame=trans_frame,
            nucleus_a_yx=nuc_a,
            nucleus_b_yx=nuc_b,
            budneck_row=bud_match,
            distance=float(pair_row.distance),
        )
        rows.append(
            {
                "condition": condition,
                "frame": frame,
                "candidate_id": candidate_id,
                "nucleus_a_id": int(pair_row.nucleus_a_id),
                "nucleus_b_id": int(pair_row.nucleus_b_id),
                "distance": float(pair_row.distance),
                **feature_row,
            }
        )
        candidate_id += 1
    return pd.DataFrame(rows)


def _list_to_str(values: list[object]) -> str:
    if not values:
        return ""
    return ";".join(str(value) for value in values)


def _sorted_partner_candidates(group: pd.DataFrame, nucleus_id: int) -> pd.DataFrame:
    subset = group[(group["nucleus_a_id"] == nucleus_id) | (group["nucleus_b_id"] == nucleus_id)].copy()
    if subset.empty:
        return subset
    subset["partner_id"] = np.where(subset["nucleus_a_id"] == nucleus_id, subset["nucleus_b_id"], subset["nucleus_a_id"])
    subset = subset.sort_values(
        [
            "budneck_between_nuclei",
            "same_trans_component",
            "cell_wall_connectedness_score",
            "distance",
            "angle_difference_from_perpendicular",
            "budneck_score",
            "preliminary_pair_score",
        ],
        ascending=[False, False, False, True, True, False, False],
    )
    return subset


def _distance_rank(distance: float, ideal_min: float, ideal_max: float, max_distance: float) -> float:
    return _distance_score(distance, ideal_min, ideal_max, max_distance)


def build_pair_assignment_tables(
    nuclei_v2_df: pd.DataFrame,
    pair_v2_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    cell_wall_df: pd.DataFrame,
    pair_require_mutual_top_k: int,
    pair_ideal_distance_min: float,
    pair_ideal_distance_max: float,
    pair_max_distance: float,
    far_pair_extra_score_required: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add assignment diagnostics, mutual-top-k logic, and conflict resolution."""

    merged = cell_wall_df.merge(
        pair_v2_df,
        on=["condition", "frame", "nucleus_a_id", "nucleus_b_id", "distance"],
        how="left",
        suffixes=("", "_pair"),
    )
    if merged.empty:
        return merged, pd.DataFrame(), pd.DataFrame()

    merged["budneck_projection_fraction"] = merged["budneck_projection_fraction"].astype(float)
    merged["budneck_score"] = merged.apply(lambda row: _budneck_score_from_pair_row(row), axis=1)
    merged["distance_rank_score"] = merged["distance"].apply(
        lambda value: _distance_rank(float(value), pair_ideal_distance_min, pair_ideal_distance_max, pair_max_distance)
    )
    merged["pair_score"] = (
        0.30 * merged["cell_wall_connectedness_score"]
        + 0.18 * merged["wall_continuity_score"]
        + 0.18 * merged["budneck_score"]
        + 0.14 * merged["distance_rank_score"]
        + 0.10 * merged["same_trans_component"].astype(float)
        + 0.10 * merged["budneck_between_nuclei"].astype(float)
    ).clip(0.0, 1.0)

    debug_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    final_rows: list[pd.Series] = []

    for (condition, frame), group in merged.groupby(["condition", "frame"], sort=True):
        group = group.copy().reset_index(drop=True)
        nucleus_ids = sorted(set(group["nucleus_a_id"]).union(set(group["nucleus_b_id"])))
        partner_lists: dict[int, pd.DataFrame] = {}
        for nucleus_id in nucleus_ids:
            partner_lists[int(nucleus_id)] = _sorted_partner_candidates(group, int(nucleus_id))

        top_partner_map: dict[int, list[int]] = {}
        for nucleus_id, subset in partner_lists.items():
            top_partner_map[nucleus_id] = [int(value) for value in subset["partner_id"].head(pair_require_mutual_top_k).tolist()]

        group["is_mutual_nearest"] = False
        group["is_mutual_top2"] = False
        group["far_pair_warning"] = False
        group["possible_wrong_far_pair_flag"] = False
        group["assignment_reason"] = ""
        group["downgrade_reason"] = ""
        group["provisional_class"] = "reject_pair"
        group["provisional_accept"] = False
        group["supports_far_pair"] = False

        for idx, row in group.iterrows():
            a_id = int(row["nucleus_a_id"])
            b_id = int(row["nucleus_b_id"])
            list_a = top_partner_map.get(a_id, [])
            list_b = top_partner_map.get(b_id, [])
            is_mutual_nearest = bool(list_a[:1] == [b_id] and list_b[:1] == [a_id])
            is_mutual_top2 = bool(b_id in list_a[:pair_require_mutual_top_k] and a_id in list_b[:pair_require_mutual_top_k])
            group.at[idx, "is_mutual_nearest"] = is_mutual_nearest
            group.at[idx, "is_mutual_top2"] = is_mutual_top2

        for nucleus_id in nucleus_ids:
            subset = partner_lists[int(nucleus_id)]
            if len(subset) <= 1:
                continue
            competing_ids = [int(value) for value in subset["candidate_id"].tolist()]
            selected_id = int(subset.iloc[0]["candidate_id"])
            rejected_ids = [int(value) for value in subset["candidate_id"].tolist()[1:]]
            selected_row = subset.iloc[0]
            conflict_rows.append(
                {
                    "condition": condition,
                    "frame": int(frame),
                    "nucleus_id": int(nucleus_id),
                    "competing_pair_ids": _list_to_str(competing_ids),
                    "selected_pair_id": selected_id,
                    "rejected_pair_ids": _list_to_str(rejected_ids),
                    "selection_reason": "ranked_by_between_then_cell_wall_then_mutual_then_distance",
                    "selected_distance": float(selected_row["distance"]),
                    "selected_cell_wall_connectedness_score": float(selected_row["cell_wall_connectedness_score"]),
                    "selected_budneck_score": float(selected_row["budneck_score"]),
                    "selected_angle_difference": float(selected_row["angle_difference_from_perpendicular"]),
                    "selected_pair_score": float(selected_row["pair_score"]),
                }
            )

        for idx, row in group.iterrows():
            distance = float(row["distance"])
            angle = float(row["angle_difference_from_perpendicular"]) if pd.notna(row["angle_difference_from_perpendicular"]) else 180.0
            budneck_dist = float(row["budneck_distance_to_line"]) if pd.notna(row["budneck_distance_to_line"]) else np.inf
            between = bool(row["budneck_between_nuclei"])
            same_comp = bool(row["same_trans_component"])
            wall_score = float(row["cell_wall_connectedness_score"])
            continuity = float(row["wall_continuity_score"])
            constriction = float(row["neck_constriction_score"])
            pair_score = float(row["pair_score"])
            budneck_score = float(row["budneck_score"])
            proj = float(row["budneck_projection_fraction"]) if pd.notna(row["budneck_projection_fraction"]) else np.nan
            far_warning = distance > pair_ideal_distance_max
            if far_warning:
                group.at[idx, "far_pair_warning"] = True

            alternatives = group[
                (
                    (group["nucleus_a_id"] == row["nucleus_a_id"])
                    | (group["nucleus_b_id"] == row["nucleus_a_id"])
                    | (group["nucleus_a_id"] == row["nucleus_b_id"])
                    | (group["nucleus_b_id"] == row["nucleus_b_id"])
                )
                & (group["candidate_id"] != row["candidate_id"])
            ].copy()
            closer_similar = alternatives[
                (alternatives["distance"] + 4.0 < distance)
                & (alternatives["cell_wall_connectedness_score"] >= wall_score - 0.08)
            ]
            wrong_far = bool(far_warning and (not row["is_mutual_nearest"] or not closer_similar.empty))
            group.at[idx, "possible_wrong_far_pair_flag"] = wrong_far

            budneck_position_ok = between and not bool(row["budneck_off_segment_flag"]) and budneck_dist <= max(5.5, 0.16 * distance)
            strong_cell_wall = same_comp and wall_score >= 0.55 and continuity >= 0.20 and constriction >= 0.08
            mutual_ok = bool(row["is_mutual_nearest"]) or (
                bool(row["is_mutual_top2"]) and budneck_score >= 0.62 and wall_score >= 0.50
            ) or (
                wall_score >= 0.82 and budneck_score >= 0.70 and pd.notna(proj) and 0.20 <= proj <= 0.80
            )

            downgrade_reasons: list[str] = []
            if not between:
                downgrade_reasons.append("budneck_not_between_nuclei")
            if bool(row["budneck_off_segment_flag"]):
                downgrade_reasons.append("budneck_off_segment")
            if not same_comp:
                downgrade_reasons.append("different_trans_components")
            if wall_score < 0.50:
                downgrade_reasons.append("cell_wall_support_weak")
            if continuity < 0.15:
                downgrade_reasons.append("wall_continuity_low")
            if constriction < 0.08:
                downgrade_reasons.append("neck_constriction_weak")
            if not bool(row["is_mutual_top2"]):
                downgrade_reasons.append("mutual_topk_failed")
            if far_warning:
                downgrade_reasons.append("far_pair_warning")
            if wrong_far:
                downgrade_reasons.append("possible_wrong_far_pair")

            supports_far_pair = bool(
                distance <= pair_ideal_distance_max
                or (
                    pair_ideal_distance_max < distance <= pair_max_distance
                    and budneck_score >= 0.68 + far_pair_extra_score_required
                    and wall_score >= 0.62
                    and continuity >= 0.24
                    and bool(row["is_mutual_nearest"])
                    and not wrong_far
                )
            )
            group.at[idx, "supports_far_pair"] = supports_far_pair

            accept = bool(
                distance >= pair_ideal_distance_min
                and distance <= pair_max_distance
                and budneck_position_ok
                and strong_cell_wall
                and mutual_ok
                and supports_far_pair
                and pair_score >= 0.60
            )
            uncertain = bool(
                not accept
                and distance <= pair_max_distance + 2.0
                and between
                and not bool(row["budneck_off_segment_flag"])
                and (
                    (same_comp and wall_score >= 0.52 and budneck_score >= 0.58 and angle <= 48.0)
                    or (bool(row["is_mutual_top2"]) and wall_score >= 0.48 and budneck_score >= 0.56)
                    or wrong_far
                )
            )

            if accept:
                group.at[idx, "provisional_class"] = "mother_bud_pair"
                group.at[idx, "provisional_accept"] = True
                reason = "between_nuclei+cell_wall_support+mutual_partner+distance_ok"
            elif uncertain:
                group.at[idx, "provisional_class"] = "uncertain_pair"
                reason = "incomplete_or_conflicting_pair_evidence"
            else:
                reason = "insufficient_pair_evidence"
            group.at[idx, "assignment_reason"] = reason
            group.at[idx, "downgrade_reason"] = _list_to_str(sorted(set(downgrade_reasons)))

            a_partners = partner_lists[a_id]
            b_partners = partner_lists[b_id]
            debug_rows.append(
                {
                    "condition": condition,
                    "frame": int(frame),
                    "selected_pair_id": int(row["candidate_id"]),
                    "nucleus_a_id": a_id,
                    "nucleus_b_id": b_id,
                    "distance": distance,
                    "pair_score": pair_score,
                    "budneck_score": budneck_score,
                    "angle_difference_from_perpendicular": angle,
                    "budneck_between_nuclei": between,
                    "budneck_distance_to_line": budneck_dist,
                    "budneck_projection_fraction": proj,
                    "budneck_midpoint_distance": float(row["budneck_midpoint_distance"]) if pd.notna(row["budneck_midpoint_distance"]) else np.nan,
                    "budneck_off_segment_flag": bool(row["budneck_off_segment_flag"]),
                    "is_mutual_nearest": bool(row["is_mutual_nearest"]),
                    "is_mutual_top2": bool(row["is_mutual_top2"]),
                    "nucleus_a_nearest_ids": _list_to_str([int(value) for value in a_partners["partner_id"].head(3).tolist()]),
                    "nucleus_a_nearest_distances": _list_to_str([round(float(value), 3) for value in a_partners["distance"].head(3).tolist()]),
                    "nucleus_b_nearest_ids": _list_to_str([int(value) for value in b_partners["partner_id"].head(3).tolist()]),
                    "nucleus_b_nearest_distances": _list_to_str([round(float(value), 3) for value in b_partners["distance"].head(3).tolist()]),
                    "alternative_pair_ids": _list_to_str([int(value) for value in alternatives["candidate_id"].head(4).tolist()]),
                    "alternative_pair_scores": _list_to_str([round(float(value), 3) for value in alternatives["pair_score"].head(4).tolist()]),
                    "alternative_pair_distances": _list_to_str([round(float(value), 3) for value in alternatives["distance"].head(4).tolist()]),
                    "same_trans_component": same_comp,
                    "cell_wall_connectedness_score": wall_score,
                    "neck_constriction_score": constriction,
                    "wall_continuity_score": continuity,
                    "assignment_reason": reason,
                    "possible_wrong_far_pair_flag": wrong_far,
                    "downgrade_reason": group.at[idx, "downgrade_reason"],
                }
            )

        accepted = group[group["provisional_accept"]].copy()
        accepted = accepted.sort_values(
            [
                "budneck_between_nuclei",
                "same_trans_component",
                "cell_wall_connectedness_score",
                "is_mutual_top2",
                "distance",
                "angle_difference_from_perpendicular",
                "budneck_score",
                "preliminary_pair_score",
            ],
            ascending=[False, False, False, False, True, True, False, False],
        )
        used_nuclei: set[int] = set()
        selected_ids: set[int] = set()
        for row in accepted.itertuples(index=False):
            a_id = int(row.nucleus_a_id)
            b_id = int(row.nucleus_b_id)
            if a_id in used_nuclei or b_id in used_nuclei:
                continue
            selected_ids.add(int(row.candidate_id))
            used_nuclei.update({a_id, b_id})

        unresolved_rank_keep: set[int] = set()
        unresolved = group[
            (~group["candidate_id"].isin(list(selected_ids)))
            & (group["provisional_class"].isin(["mother_bud_pair", "uncertain_pair"]))
        ].copy()
        if not unresolved.empty:
            for nucleus_id in nucleus_ids:
                sub = unresolved[(unresolved["nucleus_a_id"] == nucleus_id) | (unresolved["nucleus_b_id"] == nucleus_id)].copy()
                if sub.empty:
                    continue
                sub = sub.sort_values(
                    [
                        "budneck_between_nuclei",
                        "same_trans_component",
                        "cell_wall_connectedness_score",
                        "is_mutual_top2",
                        "distance",
                        "angle_difference_from_perpendicular",
                        "budneck_score",
                        "preliminary_pair_score",
                    ],
                    ascending=[False, False, False, False, True, True, False, False],
                )
                unresolved_rank_keep.add(int(sub.iloc[0]["candidate_id"]))

        for idx, row in group.iterrows():
            if int(row["candidate_id"]) in selected_ids:
                row = row.copy()
                row["final_pair_class"] = "mother_bud_pair"
                final_rows.append(row)
            elif row["provisional_class"] == "mother_bud_pair":
                row = row.copy()
                if int(row["candidate_id"]) in unresolved_rank_keep or bool(row["possible_wrong_far_pair_flag"]):
                    row["final_pair_class"] = "uncertain_pair"
                    reason = row["downgrade_reason"]
                    row["downgrade_reason"] = f"{reason};conflict_lost_to_better_pair" if reason else "conflict_lost_to_better_pair"
                    final_rows.append(row)
            elif row["provisional_class"] == "uncertain_pair" and (
                int(row["candidate_id"]) in unresolved_rank_keep
                or bool(row["possible_wrong_far_pair_flag"])
                or "mutual_topk_failed" in str(row["downgrade_reason"])
                or "cell_wall_support_weak" in str(row["downgrade_reason"])
            ):
                row = row.copy()
                row["final_pair_class"] = "uncertain_pair"
                final_rows.append(row)

    debug_df = pd.DataFrame(debug_rows)
    conflict_df = pd.DataFrame(conflict_rows)
    final_pair_df = pd.DataFrame(final_rows)
    return debug_df, conflict_df, final_pair_df


def diagnose_missing_nucleus(
    gfp_stacks: dict[str, np.ndarray],
    nuclei_v2_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> pd.DataFrame:
    """Check for strong nearby GFP signal where a missed nucleus might exist."""

    nuclei_groups = {
        (condition, int(frame)): group.copy()
        for (condition, frame), group in nuclei_v2_df.groupby(["condition", "frame"], sort=True)
    }
    bud_groups = {
        (condition, int(frame)): group.copy()
        for (condition, frame), group in budneck_df.groupby(["condition", "frame"], sort=True)
    }

    rows: list[dict[str, object]] = []
    targets = classification_df[
        (classification_df["final_class"].isin(["early_bud_pair", "uncertain_pair"]))
        | (classification_df.get("possible_wrong_far_pair_flag", False) == True)
    ].copy()
    for row in targets.itertuples(index=False):
        condition = str(row.condition)
        frame = int(row.frame)
        gfp_frame = np.asarray(gfp_stacks[condition][frame], dtype=np.float32)
        bud_frame = bud_groups.get((condition, frame), pd.DataFrame())
        nuclei_frame = nuclei_groups.get((condition, frame), pd.DataFrame())
        if nuclei_frame.empty or pd.isna(row.nucleus_a_id):
            continue
        nucleus = nuclei_frame[nuclei_frame["nucleus_id"] == int(row.nucleus_a_id)]
        if nucleus.empty:
            continue
        nuc_y = float(nucleus.iloc[0]["centroid_y"])
        nuc_x = float(nucleus.iloc[0]["centroid_x"])
        bud_match = pd.DataFrame()
        if pd.notna(row.nearest_budneck_id) and not bud_frame.empty:
            bud_match = bud_frame[bud_frame["budneck_id"] == int(row.nearest_budneck_id)]
        if not bud_match.empty:
            bud_y = float(bud_match.iloc[0]["centroid_y"])
            bud_x = float(bud_match.iloc[0]["centroid_x"])
            vector_y = bud_y - nuc_y
            vector_x = bud_x - nuc_x
            expected_y = bud_y + 0.8 * vector_y
            expected_x = bud_x + 0.8 * vector_x
            reason = "budneck_extension_region"
        else:
            expected_y = nuc_y
            expected_x = nuc_x
            reason = "no_budneck_available"
        patch, (y0, x0) = _extract_patch(gfp_frame, expected_y, expected_x, half_size=8)
        if patch.size == 0:
            continue
        local_max = float(np.max(patch))
        local_mean = float(np.mean(patch))
        background = estimate_background(gfp_frame, (max(0, int(expected_y) - 4), max(0, int(expected_x) - 4), min(gfp_frame.shape[0], int(expected_y) + 5), min(gfp_frame.shape[1], int(expected_x) + 5)), margin=8)
        signal_to_background = float((local_max + 1.0) / (background + 1.0))
        possible = bool(signal_to_background >= 2.0 and local_max >= float(np.percentile(gfp_frame, 98.8)))
        rows.append(
            {
                "condition": condition,
                "frame": frame,
                "candidate_id": int(row.candidate_id),
                "nucleus_id": int(row.nucleus_a_id),
                "expected_partner_region_y": float(expected_y),
                "expected_partner_region_x": float(expected_x),
                "local_gfp_max": local_max,
                "local_gfp_mean": local_mean,
                "local_background": background,
                "signal_to_background": signal_to_background,
                "possible_missed_nucleus_nearby": possible,
                "diagnosis_reason": "strong_local_gfp_near_expected_partner" if possible else reason,
            }
        )
    return pd.DataFrame(rows)


def classify_rules_v2_1(
    trans_stacks: dict[str, np.ndarray],
    gfp_stacks: dict[str, np.ndarray],
    nuclei_v1_df: pd.DataFrame,
    nuclei_v2_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    pair_v2_df: pd.DataFrame,
    pair_require_mutual_top_k: int,
    pair_ideal_distance_min: float,
    pair_ideal_distance_max: float,
    pair_max_distance: float,
    far_pair_extra_score_required: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build explicit cell-wall features, conflict-resolved pair classes, and single/early-bud labels."""

    cell_wall_df = build_cell_wall_features(trans_stacks, nuclei_v2_df, budneck_df, pair_v2_df)
    debug_df, conflict_df, final_pair_df = build_pair_assignment_tables(
        nuclei_v2_df=nuclei_v2_df,
        pair_v2_df=pair_v2_df,
        budneck_df=budneck_df,
        cell_wall_df=cell_wall_df,
        pair_require_mutual_top_k=pair_require_mutual_top_k,
        pair_ideal_distance_min=pair_ideal_distance_min,
        pair_ideal_distance_max=pair_ideal_distance_max,
        pair_max_distance=pair_max_distance,
        far_pair_extra_score_required=far_pair_extra_score_required,
    )

    used_nuclei = set()
    classification_rows: list[dict[str, object]] = []
    if not final_pair_df.empty:
        for row in final_pair_df.itertuples(index=False):
            classification_rows.append(
                {
                    "condition": str(row.condition),
                    "frame": int(row.frame),
                    "candidate_id": int(row.candidate_id),
                    "nucleus_a_id": int(row.nucleus_a_id),
                    "nucleus_b_id": int(row.nucleus_b_id),
                    "distance": float(row.distance),
                    "pair_score": float(row.pair_score),
                    "budneck_score": float(row.budneck_score),
                    "same_trans_component": bool(row.same_trans_component),
                    "cell_wall_connectedness_score": float(row.cell_wall_connectedness_score),
                    "neck_constriction_width": float(row.neck_constriction_width),
                    "neck_constriction_score": float(row.neck_constriction_score),
                    "bud_lobe_detected": bool(row.bud_lobe_detected),
                    "mother_bud_area_ratio": float(row.mother_bud_area_ratio),
                    "wall_continuity_score": float(row.wall_continuity_score),
                    "is_mutual_top2": bool(row.is_mutual_top2),
                    "budneck_between_nuclei": bool(row.budneck_between_nuclei),
                    "final_class": str(row.final_pair_class),
                    "downgrade_reason": str(row.downgrade_reason),
                    "assignment_reason": str(row.assignment_reason),
                    "nearest_budneck_id": None if pd.isna(row.nearest_budneck_id) else int(row.nearest_budneck_id),
                    "budneck_distance_to_line": float(row.budneck_distance_to_line) if pd.notna(row.budneck_distance_to_line) else np.nan,
                    "budneck_projection_fraction": float(row.budneck_projection_fraction) if pd.notna(row.budneck_projection_fraction) else np.nan,
                    "budneck_midpoint_distance": float(row.budneck_midpoint_distance) if pd.notna(row.budneck_midpoint_distance) else np.nan,
                    "budneck_off_segment_flag": bool(row.budneck_off_segment_flag),
                    "possible_wrong_far_pair_flag": bool(row.possible_wrong_far_pair_flag),
                    "contour_yx_json": row.contour_yx_json,
                    "mask_bbox_y0": int(row.mask_bbox_y0),
                    "mask_bbox_x0": int(row.mask_bbox_x0),
                    "mask_bbox_y1": int(row.mask_bbox_y1),
                    "mask_bbox_x1": int(row.mask_bbox_x1),
                    "source_type": "pair_candidate",
                }
            )
            if str(row.final_pair_class) == "mother_bud_pair":
                used_nuclei.update({int(row.nucleus_a_id), int(row.nucleus_b_id)})

    # Early-bud and single-cell branch.
    nuclei_groups = {
        (condition, int(frame)): group.copy()
        for (condition, frame), group in nuclei_v2_df.groupby(["condition", "frame"], sort=True)
    }
    bud_groups = {
        (condition, int(frame)): group.copy()
        for (condition, frame), group in budneck_df.groupby(["condition", "frame"], sort=True)
    }
    next_candidate_id = int(cell_wall_df["candidate_id"].max()) + 1 if not cell_wall_df.empty else 1
    for condition, stack in gfp_stacks.items():
        trans_stack = trans_stacks[condition]
        for frame_index in range(int(stack.shape[0])):
            nuclei_frame = nuclei_groups.get((condition, frame_index), pd.DataFrame()).copy()
            bud_frame = bud_groups.get((condition, frame_index), pd.DataFrame()).copy()
            if nuclei_frame.empty:
                continue
            trans_frame = np.asarray(trans_stack[frame_index], dtype=np.float32)
            candidate_like_nuclei = set()
            for row in classification_rows:
                if (
                    row["condition"] == condition
                    and int(row["frame"]) == frame_index
                    and row["final_class"] == "uncertain_pair"
                    and bool(row["is_mutual_top2"])
                    and bool(row["same_trans_component"])
                    and float(row["cell_wall_connectedness_score"]) >= 0.60
                ):
                    candidate_like_nuclei.add(int(row["nucleus_a_id"]))
                    if pd.notna(row["nucleus_b_id"]):
                        candidate_like_nuclei.add(int(row["nucleus_b_id"]))
            for nucleus_row in nuclei_frame.sort_values(["is_high_confidence", "signal_to_background", "max_intensity"], ascending=[False, False, False]).itertuples(index=False):
                nucleus_id = int(nucleus_row.nucleus_id)
                if nucleus_id in used_nuclei or nucleus_id in candidate_like_nuclei:
                    continue
                if not bool(nucleus_row.is_high_confidence):
                    continue
                bud_choice = nearest_budneck_for_nucleus(pd.Series(nucleus_row._asdict()), bud_frame)
                if bud_choice is not None:
                    nucleus_yx = (float(nucleus_row.centroid_y), float(nucleus_row.centroid_x))
                    pseudo_partner_yx = (float(bud_choice["centroid_y"]) + (float(bud_choice["centroid_y"]) - nucleus_yx[0]) * 0.8,
                                         float(bud_choice["centroid_x"]) + (float(bud_choice["centroid_x"]) - nucleus_yx[1]) * 0.8)
                    single_features = extract_pair_cell_wall_features(
                        candidate_id=next_candidate_id,
                        trans_frame=trans_frame,
                        nucleus_a_yx=nucleus_yx,
                        nucleus_b_yx=pseudo_partner_yx,
                        budneck_row=bud_choice,
                        distance=float(math.hypot(pseudo_partner_yx[0] - nucleus_yx[0], pseudo_partner_yx[1] - nucleus_yx[1])),
                    )
                    early_bud_support = bool(
                        single_features["same_trans_component"]
                        and (
                            single_features["bud_lobe_detected"]
                            or (
                                single_features["neck_constriction_score"] >= 0.08
                                and single_features["wall_continuity_score"] >= 0.16
                                and 0.04 <= single_features["mother_bud_area_ratio"] <= 0.78
                            )
                        )
                        and float(bud_choice["signal_to_background"]) >= 1.8
                        and single_features["cell_wall_connectedness_score"] >= 0.52
                    )
                    if early_bud_support:
                        classification_rows.append(
                            {
                                "condition": condition,
                                "frame": frame_index,
                                "candidate_id": next_candidate_id,
                                "nucleus_a_id": nucleus_id,
                                "nucleus_b_id": np.nan,
                                "distance": float(math.hypot(float(bud_choice["centroid_y"]) - float(nucleus_row.centroid_y), float(bud_choice["centroid_x"]) - float(nucleus_row.centroid_x))),
                                "pair_score": float(
                                    0.40 * np.clip(float(nucleus_row.signal_to_background) / 5.0, 0.0, 1.0)
                                    + 0.30 * single_features["cell_wall_connectedness_score"]
                                    + 0.30 * single_features["wall_continuity_score"]
                                ),
                                "budneck_score": float(np.clip(float(bud_choice["signal_to_background"]) / 4.0, 0.0, 1.0)),
                                "same_trans_component": bool(single_features["same_trans_component"]),
                                "cell_wall_connectedness_score": float(single_features["cell_wall_connectedness_score"]),
                                "neck_constriction_width": float(single_features["neck_constriction_width"]),
                                "neck_constriction_score": float(single_features["neck_constriction_score"]),
                                "bud_lobe_detected": bool(single_features["bud_lobe_detected"]),
                                "mother_bud_area_ratio": float(single_features["mother_bud_area_ratio"]),
                                "wall_continuity_score": float(single_features["wall_continuity_score"]),
                                "is_mutual_top2": False,
                                "budneck_between_nuclei": True,
                                "final_class": "early_bud_pair",
                                "downgrade_reason": "",
                                "assignment_reason": "single_nucleus_plus_bud_lobe_plus_budneck_plus_cell_wall_support",
                                "nearest_budneck_id": int(bud_choice["budneck_id"]),
                                "budneck_distance_to_line": np.nan,
                                "budneck_projection_fraction": float(single_features["budneck_projection_fraction"]) if not pd.isna(single_features["budneck_projection_fraction"]) else np.nan,
                                "budneck_midpoint_distance": float(single_features["budneck_midpoint_distance"]) if not pd.isna(single_features["budneck_midpoint_distance"]) else np.nan,
                                "budneck_off_segment_flag": bool(single_features["budneck_off_segment_flag"]),
                                "possible_wrong_far_pair_flag": False,
                                "contour_yx_json": single_features["contour_yx_json"],
                                "mask_bbox_y0": int(single_features["mask_bbox_y0"]),
                                "mask_bbox_x0": int(single_features["mask_bbox_x0"]),
                                "mask_bbox_y1": int(single_features["mask_bbox_y1"]),
                                "mask_bbox_x1": int(single_features["mask_bbox_x1"]),
                                "source_type": "single_nucleus_bud_candidate",
                            }
                        )
                        used_nuclei.add(nucleus_id)
                        next_candidate_id += 1
                        continue

                classification_rows.append(
                    {
                        "condition": condition,
                        "frame": frame_index,
                        "candidate_id": next_candidate_id,
                        "nucleus_a_id": nucleus_id,
                        "nucleus_b_id": np.nan,
                        "distance": np.nan,
                        "pair_score": float(np.clip(float(nucleus_row.signal_to_background) / 5.0, 0.0, 1.0)),
                        "budneck_score": np.nan,
                        "same_trans_component": False,
                        "cell_wall_connectedness_score": 0.0,
                        "neck_constriction_width": 0.0,
                        "neck_constriction_score": 0.0,
                        "bud_lobe_detected": False,
                        "mother_bud_area_ratio": 0.0,
                        "wall_continuity_score": 0.0,
                        "is_mutual_top2": False,
                        "budneck_between_nuclei": False,
                        "final_class": "single_cell",
                        "downgrade_reason": "",
                        "assignment_reason": "no_reliable_budneck_or_budding_morphology",
                        "nearest_budneck_id": np.nan,
                        "budneck_distance_to_line": np.nan,
                        "budneck_projection_fraction": np.nan,
                        "budneck_midpoint_distance": np.nan,
                        "budneck_off_segment_flag": False,
                        "possible_wrong_far_pair_flag": False,
                        "contour_yx_json": "[]",
                        "mask_bbox_y0": np.nan,
                        "mask_bbox_x0": np.nan,
                        "mask_bbox_y1": np.nan,
                        "mask_bbox_x1": np.nan,
                        "source_type": "single_nucleus_candidate",
                    }
                )
                next_candidate_id += 1

    classification_df = pd.DataFrame(classification_rows)
    missing_nucleus_df = diagnose_missing_nucleus(
        gfp_stacks=gfp_stacks,
        nuclei_v2_df=nuclei_v2_df,
        budneck_df=budneck_df,
        classification_df=classification_df,
    )
    return cell_wall_df, debug_df, conflict_df, missing_nucleus_df, classification_df


def build_rules_v2_1_summary(
    nuclei_v1_df: pd.DataFrame,
    nuclei_v2_df: pd.DataFrame,
    v2_classification_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    debug_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact summary table for rules_v2.1."""

    rows: list[dict[str, object]] = []
    class_counts = classification_df["final_class"].value_counts()
    v2_pairs = v2_classification_df[v2_classification_df["rule_class"] == "mother_bud_pair"].copy()
    v2_pairs["pair_key"] = v2_pairs.apply(
        lambda row: f"{row['condition']}|{int(row['frame'])}|{int(min(row['nucleus_a_id'], row['nucleus_b_id']))}|{int(max(row['nucleus_a_id'], row['nucleus_b_id']))}",
        axis=1,
    )
    v21_pairs = classification_df[classification_df["final_class"] == "mother_bud_pair"].copy()
    if not v21_pairs.empty:
        v21_pairs["pair_key"] = v21_pairs.apply(
            lambda row: f"{row['condition']}|{int(row['frame'])}|{int(min(row['nucleus_a_id'], row['nucleus_b_id']))}|{int(max(row['nucleus_a_id'], row['nucleus_b_id']))}"
            if pd.notna(row["nucleus_b_id"])
            else "",
            axis=1,
        )
    downgraded_v2_pairs = 0
    if not v2_pairs.empty:
        v21_keys = set(v21_pairs["pair_key"].tolist()) if not v21_pairs.empty else set()
        downgraded_v2_pairs = int(sum(key not in v21_keys for key in v2_pairs["pair_key"]))

    rows.append(
        {
            "summary_level": "global",
            "mother_bud_pair": int(class_counts.get("mother_bud_pair", 0)),
            "early_bud_pair": int(class_counts.get("early_bud_pair", 0)),
            "single_cell": int(class_counts.get("single_cell", 0)),
            "uncertain_pair": int(class_counts.get("uncertain_pair", 0)),
            "nuclei_v1_total": int(len(nuclei_v1_df)),
            "nuclei_v2_total": int(len(nuclei_v2_df)),
            "recovered_vs_v1": int(nuclei_v2_df["recovered_vs_v1"].sum()),
            "v2_mother_pairs_downgraded_or_rejected": downgraded_v2_pairs,
            "downgraded_due_to_cell_wall_support": int(classification_df["downgrade_reason"].fillna("").str.contains("cell_wall_support_weak").sum()),
            "far_pair_warnings": int(debug_df["possible_wrong_far_pair_flag"].sum()) if not debug_df.empty else 0,
            "conflicts_resolved": int(len(conflict_df)),
            "early_bud_supported_by_bud_lobe": int(classification_df[(classification_df["final_class"] == "early_bud_pair") & (classification_df["bud_lobe_detected"] == True)].shape[0]),
        }
    )
    return pd.DataFrame(rows)
