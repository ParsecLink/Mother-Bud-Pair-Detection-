"""Rules v2 feature extraction and classification helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation

from .image_utils import background_correct, percentile_normalize
from .rule_features import axis_angle_deg, axis_difference_deg, build_pair_candidates, estimate_background


@dataclass(frozen=True)
class MorphologyResult:
    center_y: float
    center_x: float
    mask_area: int
    contour_point_count: int
    lobe_count: int
    neck_constriction_strength: float
    bud_area: int
    mother_area: int
    bud_to_mother_area_ratio: float
    boundary_connectedness: float
    has_small_attached_bud: bool
    mask_polarity: str
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    morphology_support_score: float


def _extract_patch(image: np.ndarray, center_y: float, center_x: float, half_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    y = int(round(center_y))
    x = int(round(center_x))
    y0 = max(0, y - half_size)
    x0 = max(0, x - half_size)
    y1 = min(image.shape[0], y + half_size)
    x1 = min(image.shape[1], x + half_size)
    return np.asarray(image[y0:y1, x0:x1], dtype=np.float32), (y0, x0)


def _merge_candidate_points(points: list[tuple[float, float, float, str]], min_distance: float = 4.0) -> list[tuple[float, float, float, str]]:
    if not points:
        return []
    points = sorted(points, key=lambda item: item[2], reverse=True)
    kept: list[tuple[float, float, float, str]] = []
    for y, x, score, source in points:
        if any(math.hypot(y - ky, x - kx) <= min_distance for ky, kx, _, _ in kept):
            continue
        kept.append((float(y), float(x), float(score), str(source)))
    return kept


def _component_around_point(mask: np.ndarray, center_y: int, center_x: int) -> np.ndarray:
    if mask.size == 0:
        return np.zeros_like(mask, dtype=bool)
    labeled = measure.label(mask)
    if 0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1]:
        label = int(labeled[center_y, center_x])
        if label > 0:
            return labeled == label
    if labeled.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    props = measure.regionprops(labeled)
    best = min(props, key=lambda prop: math.hypot(prop.centroid[0] - center_y, prop.centroid[1] - center_x))
    return labeled == int(best.label)


def detect_nuclei_v2_frame(image: np.ndarray, condition: str, frame_index: int) -> list[dict[str, object]]:
    """Detect GFP nuclei with high- and low-confidence tiers on raw projected GFP."""

    raw = np.asarray(image, dtype=np.float32)
    normalized = percentile_normalize(raw, low=0.3, high=99.95)
    corrected = background_correct(raw, gaussian_sigma=0.8, tophat_radius=7)
    local_background = filters.gaussian(corrected, sigma=5.0, preserve_range=True)
    local_contrast = np.clip(corrected - 0.7 * local_background, 0.0, None)
    log_response = np.clip(-ndi.gaussian_laplace(local_contrast, sigma=1.25), 0.0, None)
    if np.max(log_response) > 0:
        log_response = log_response / float(np.max(log_response))

    positive = log_response[log_response > 0]
    if positive.size == 0:
        return []

    high_abs = max(float(np.percentile(positive, 98.8)) * 0.55, 0.018)
    low_abs = max(float(np.percentile(positive, 97.2)) * 0.35, 0.010)
    peaks_high = feature.peak_local_max(
        log_response,
        min_distance=5,
        threshold_abs=high_abs,
        exclude_border=False,
    )
    peaks_low = feature.peak_local_max(
        log_response,
        min_distance=4,
        threshold_abs=low_abs,
        exclude_border=False,
    )
    blobs = feature.blob_log(
        local_contrast,
        min_sigma=1.0,
        max_sigma=4.5,
        num_sigma=7,
        threshold=max(float(np.percentile(local_contrast, 99.0)) * 0.12, 0.008),
    )

    point_pool: list[tuple[float, float, float, str]] = []
    for y, x in peaks_high:
        point_pool.append((float(y), float(x), float(log_response[int(y), int(x)]), "peak_high"))
    for y, x in peaks_low:
        point_pool.append((float(y), float(x), float(log_response[int(y), int(x)]) * 0.9, "peak_low"))
    for y, x, sigma in blobs:
        yy = int(round(float(y)))
        xx = int(round(float(x)))
        point_pool.append((float(y), float(x), float(log_response[yy, xx]) * 0.95, "blob_log"))
    merged_points = _merge_candidate_points(point_pool, min_distance=4.0)

    raw_peak_high_cutoff = float(np.percentile(raw, 99.1))
    raw_peak_low_cutoff = float(np.percentile(raw, 98.4))
    rows: list[dict[str, object]] = []
    nucleus_id = 1
    for y, x, response_score, source in merged_points:
        patch, (y0, x0) = _extract_patch(raw, y, x, half_size=8)
        corrected_patch, _ = _extract_patch(local_contrast, y, x, half_size=8)
        local_y = int(round(y)) - y0
        local_x = int(round(x)) - x0
        if not (0 <= local_y < patch.shape[0] and 0 <= local_x < patch.shape[1]):
            continue
        raw_peak = float(raw[int(round(y)), int(round(x))])
        corrected_peak = float(local_contrast[int(round(y)), int(round(x))])
        background = estimate_background(raw, (max(0, int(round(y)) - 4), max(0, int(round(x)) - 4), min(raw.shape[0], int(round(y)) + 5), min(raw.shape[1], int(round(x)) + 5)), margin=5)
        signal_to_background = float((raw_peak + 1.0) / (background + 1.0))

        local_threshold = max(float(np.percentile(corrected_patch, 85.0)), corrected_peak * 0.42)
        local_mask = corrected_patch > local_threshold
        local_mask = morphology.binary_opening(local_mask, morphology.disk(1))
        local_mask = morphology.remove_small_objects(local_mask, 1)
        component = _component_around_point(local_mask, local_y, local_x)
        area = int(component.sum()) if component.any() else 1
        if component.any():
            mean_intensity = float(patch[component].mean())
        else:
            mean_intensity = raw_peak

        is_high = bool(
            signal_to_background >= 3.0
            and raw_peak >= raw_peak_high_cutoff
            and 2 <= area <= 180
        )
        is_low = bool(
            signal_to_background >= 1.8
            and raw_peak >= raw_peak_low_cutoff
            and 1 <= area <= 240
        )
        if not (is_high or is_low):
            continue

        confidence_class = "high_confidence_nucleus" if is_high else "low_confidence_nucleus"
        rows.append(
            {
                "condition": condition,
                "frame": int(frame_index),
                "nucleus_id": nucleus_id,
                "centroid_y": float(y),
                "centroid_x": float(x),
                "area": area,
                "max_intensity": raw_peak,
                "mean_intensity": mean_intensity,
                "signal_to_background": signal_to_background,
                "local_response": float(response_score),
                "source_method": source,
                "confidence_class": confidence_class,
                "is_high_confidence": bool(is_high),
                "is_low_confidence": bool(not is_high and is_low),
                "missing_nucleus_flag": False,
            }
        )
        nucleus_id += 1
    return rows


def build_nucleus_candidates_v2(
    gfp_stacks: dict[str, np.ndarray],
    v1_nuclei: pd.DataFrame,
    match_distance: float = 4.0,
) -> pd.DataFrame:
    """Detect v2 nuclei across all conditions and mark which are new relative to v1."""

    rows: list[dict[str, object]] = []
    for condition, stack in gfp_stacks.items():
        for frame_index in range(int(stack.shape[0])):
            rows.extend(detect_nuclei_v2_frame(stack[frame_index], condition, frame_index))
    nuclei_df = pd.DataFrame(rows)
    if nuclei_df.empty:
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
                "local_response",
                "source_method",
                "confidence_class",
                "is_high_confidence",
                "is_low_confidence",
                "missing_nucleus_flag",
                "matched_to_v1",
                "recovered_vs_v1",
            ]
        )

    matched_to_v1: list[bool] = []
    for row in nuclei_df.itertuples(index=False):
        subset = v1_nuclei[(v1_nuclei["condition"] == row.condition) & (v1_nuclei["frame"] == row.frame)]
        if subset.empty:
            matched_to_v1.append(False)
            continue
        distances = np.sqrt((subset["centroid_y"] - row.centroid_y) ** 2 + (subset["centroid_x"] - row.centroid_x) ** 2)
        matched_to_v1.append(bool((distances <= match_distance).any()))
    nuclei_df["matched_to_v1"] = matched_to_v1
    nuclei_df["recovered_vs_v1"] = ~nuclei_df["matched_to_v1"]
    return nuclei_df


def _seed_disk_mask(shape: tuple[int, int], seeds: list[tuple[float, float]], radius: int = 8) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for y, x in seeds:
        mask |= (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    return mask


def _select_local_mask(
    trans_patch: np.ndarray,
    seeds_patch: list[tuple[float, float]],
) -> tuple[np.ndarray, str]:
    norm = percentile_normalize(trans_patch, low=1.0, high=99.6)
    smooth = filters.gaussian(norm, sigma=1.0, preserve_range=True)
    inv = 1.0 - smooth

    def _make_binary(source: np.ndarray) -> np.ndarray:
        local = source > filters.threshold_local(source, block_size=31, offset=-0.01)
        global_mask = source > filters.threshold_otsu(source)
        binary = local | global_mask
        binary = morphology.binary_closing(binary, morphology.disk(2))
        binary = morphology.binary_opening(binary, morphology.disk(1))
        binary = morphology.remove_small_objects(binary, 25)
        binary = morphology.remove_small_holes(binary, area_threshold=60)
        return np.asarray(binary, dtype=bool)

    candidates = [("dark", _make_binary(inv)), ("bright", _make_binary(smooth))]
    best_mask = np.zeros_like(trans_patch, dtype=bool)
    best_polarity = "fallback"
    best_score = -1e9
    seed_center_y = float(np.mean([seed[0] for seed in seeds_patch]))
    seed_center_x = float(np.mean([seed[1] for seed in seeds_patch]))
    for polarity, binary in candidates:
        labeled = measure.label(binary)
        if labeled.max() == 0:
            continue
        for prop in measure.regionprops(labeled):
            component = labeled == int(prop.label)
            seed_hits = 0
            for sy, sx in seeds_patch:
                iy = int(round(sy))
                ix = int(round(sx))
                if 0 <= iy < component.shape[0] and 0 <= ix < component.shape[1] and component[iy, ix]:
                    seed_hits += 1
            center_distance = math.hypot(prop.centroid[0] - seed_center_y, prop.centroid[1] - seed_center_x)
            area_penalty = abs(float(prop.area) - 700.0) / 700.0
            score = 3.0 * seed_hits - 0.5 * area_penalty - 0.02 * center_distance
            if score > best_score:
                best_score = score
                best_mask = component
                best_polarity = polarity

    if best_mask.sum() == 0:
        best_mask = _seed_disk_mask(trans_patch.shape, seeds_patch, radius=10)
        best_polarity = "seed_fallback"
    return best_mask, best_polarity


def _count_lobes(mask: np.ndarray) -> int:
    if mask.sum() == 0:
        return 0
    distance = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(distance, min_distance=8, threshold_abs=2.0, labels=mask)
    return int(max(1, min(len(peaks), 4)))


def _compute_width_profile(
    mask: np.ndarray,
    center_yx: tuple[float, float],
    axis_angle_deg_value: float,
    neck_yx: tuple[float, float] | None,
) -> tuple[float, float]:
    coords = np.argwhere(mask)
    if coords.shape[0] < 10:
        return 0.0, 0.0
    center = np.asarray(center_yx, dtype=np.float32)
    theta = math.radians(axis_angle_deg_value)
    axis_vec = np.asarray([math.sin(theta), math.cos(theta)], dtype=np.float32)
    perp_vec = np.asarray([math.cos(theta), -math.sin(theta)], dtype=np.float32)
    rel = coords.astype(np.float32) - center[None, :]
    longitudinal = rel @ axis_vec
    transverse = rel @ perp_vec
    if longitudinal.size < 8:
        return 0.0, 0.0

    bins = np.linspace(float(longitudinal.min()), float(longitudinal.max()), 17)
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
        return 0.0, 0.0
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
    connectedness = float(np.clip(neck_width / reference_width, 0.0, 1.0))
    return constriction, connectedness


def _watershed_two_compartments(mask: np.ndarray, seed_points_patch: list[tuple[float, float]]) -> tuple[int, int]:
    if mask.sum() == 0 or len(seed_points_patch) < 2:
        return int(mask.sum()), 0
    markers = np.zeros(mask.shape, dtype=np.int32)
    placed = 0
    for index, (sy, sx) in enumerate(seed_points_patch[:2], start=1):
        iy = int(round(sy))
        ix = int(round(sx))
        if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
            markers[iy, ix] = index
            placed += 1
    if placed < 2:
        return int(mask.sum()), 0
    distance = ndi.distance_transform_edt(mask)
    labels = segmentation.watershed(-distance, markers, mask=mask)
    areas = [int((labels == label).sum()) for label in [1, 2]]
    if min(areas) == 0:
        return int(mask.sum()), 0
    return max(areas), min(areas)


def extract_trans_morphology(
    trans_frame: np.ndarray,
    seed_points_global: list[tuple[float, float]],
    axis_angle_deg_value: float,
    budneck_yx: tuple[float, float] | None,
    split_mode: str,
    patch_half_size: int,
) -> MorphologyResult:
    """Extract local budding morphology around a candidate structure in Trans."""

    center_y = float(np.mean([point[0] for point in seed_points_global]))
    center_x = float(np.mean([point[1] for point in seed_points_global]))
    if budneck_yx is not None:
        center_y = 0.6 * center_y + 0.4 * float(budneck_yx[0])
        center_x = 0.6 * center_x + 0.4 * float(budneck_yx[1])
    patch, (y0, x0) = _extract_patch(trans_frame, center_y, center_x, half_size=patch_half_size)
    seeds_patch = [(float(y - y0), float(x - x0)) for y, x in seed_points_global]
    budneck_patch = None if budneck_yx is None else (float(budneck_yx[0] - y0), float(budneck_yx[1] - x0))

    mask, polarity = _select_local_mask(patch, seeds_patch + ([] if budneck_patch is None else [budneck_patch]))
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    if props:
        prop = max(props, key=lambda item: item.area)
        contour_point_count = int(sum(len(contour) for contour in measure.find_contours(mask.astype(float), 0.5)))
        bbox_y0, bbox_x0, bbox_y1, bbox_x1 = prop.bbox
        centroid_y = float(prop.centroid[0] + y0)
        centroid_x = float(prop.centroid[1] + x0)
    else:
        contour_point_count = 0
        bbox_y0 = bbox_x0 = 0
        bbox_y1, bbox_x1 = patch.shape
        centroid_y = center_y
        centroid_x = center_x

    lobe_count = _count_lobes(mask)
    neck_constriction_strength, boundary_connectedness = _compute_width_profile(
        mask,
        center_yx=(centroid_y - y0, centroid_x - x0),
        axis_angle_deg_value=axis_angle_deg_value,
        neck_yx=budneck_patch,
    )

    if split_mode == "pair" and len(seeds_patch) >= 2:
        mother_area, bud_area = _watershed_two_compartments(mask, seeds_patch[:2])
    else:
        coords = np.argwhere(mask)
        if coords.size == 0:
            mother_area = 0
            bud_area = 0
        else:
            theta = math.radians(axis_angle_deg_value)
            axis_vec = np.asarray([math.sin(theta), math.cos(theta)], dtype=np.float32)
            center_local = np.asarray([centroid_y - y0, centroid_x - x0], dtype=np.float32)
            rel = coords.astype(np.float32) - center_local[None, :]
            longitudinal = rel @ axis_vec
            if budneck_patch is not None:
                neck_rel = np.asarray(budneck_patch, dtype=np.float32) - center_local
                split_value = float(neck_rel @ axis_vec)
            else:
                split_value = float(np.median(longitudinal))
            bud_area = int(np.sum(longitudinal > split_value))
            mother_area = int(np.sum(longitudinal <= split_value))
            if mother_area < bud_area:
                mother_area, bud_area = bud_area, mother_area

    ratio = float(bud_area / max(mother_area, 1)) if mother_area > 0 else 0.0
    has_small_attached_bud = bool(
        bud_area >= 12
        and mother_area >= 40
        and 0.05 <= ratio <= 0.85
        and neck_constriction_strength >= 0.08
        and boundary_connectedness >= 0.05
    )
    morphology_support_score = float(
        0.35 * min(lobe_count / 2.0, 1.0)
        + 0.25 * neck_constriction_strength
        + 0.20 * min(boundary_connectedness / 0.45, 1.0)
        + 0.20 * (1.0 if has_small_attached_bud else 0.0)
    )

    return MorphologyResult(
        center_y=centroid_y,
        center_x=centroid_x,
        mask_area=int(mask.sum()),
        contour_point_count=contour_point_count,
        lobe_count=lobe_count,
        neck_constriction_strength=neck_constriction_strength,
        bud_area=int(bud_area),
        mother_area=int(mother_area),
        bud_to_mother_area_ratio=ratio,
        boundary_connectedness=boundary_connectedness,
        has_small_attached_bud=has_small_attached_bud,
        mask_polarity=polarity,
        bbox_y0=int(bbox_y0 + y0),
        bbox_x0=int(bbox_x0 + x0),
        bbox_y1=int(bbox_y1 + y0),
        bbox_x1=int(bbox_x1 + x0),
        morphology_support_score=morphology_support_score,
    )


def nearest_budneck_for_nucleus(
    nucleus_row: pd.Series,
    budneck_frame_df: pd.DataFrame,
    max_distance: float = 28.0,
) -> pd.Series | None:
    """Pick the best nearby bud-neck candidate for a nucleus."""

    if budneck_frame_df.empty:
        return None
    dx = budneck_frame_df["centroid_x"] - float(nucleus_row["centroid_x"])
    dy = budneck_frame_df["centroid_y"] - float(nucleus_row["centroid_y"])
    distance = np.sqrt(dx**2 + dy**2)
    nearby = budneck_frame_df[(distance >= 6.0) & (distance <= max_distance)].copy()
    if nearby.empty:
        return None
    nearby["nucleus_distance"] = np.sqrt(
        (nearby["centroid_y"] - float(nucleus_row["centroid_y"])) ** 2
        + (nearby["centroid_x"] - float(nucleus_row["centroid_x"])) ** 2
    )
    nearby["selection_score"] = (
        0.45 * np.clip(nearby["signal_to_background"] / 4.0, 0.0, 1.0)
        + 0.30 * np.clip((nearby["aspect_ratio"] - 1.1) / 2.0, 0.0, 1.0)
        + 0.25 * np.clip(1.0 - (nearby["nucleus_distance"] - 8.0) / 20.0, 0.0, 1.0)
    )
    return nearby.sort_values("selection_score", ascending=False).iloc[0]


def classify_rules_v2(
    trans_stacks: dict[str, np.ndarray],
    nucleus_candidates_v2: pd.DataFrame,
    budneck_candidates: pd.DataFrame,
    pair_candidates_v2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify structures into mother-bud, early-bud, and single-cell using morphology."""

    morphology_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    structure_id = 1
    morphology_id = 1

    nuclei_groups = {
        (condition, int(frame)): group.copy()
        for (condition, frame), group in nucleus_candidates_v2.groupby(["condition", "frame"], sort=True)
    }
    bud_groups = {
        (condition, int(frame)): group.copy()
        for (condition, frame), group in budneck_candidates.groupby(["condition", "frame"], sort=True)
    }
    pair_groups = {
        (condition, int(frame)): group.copy()
        for (condition, frame), group in pair_candidates_v2.groupby(["condition", "frame"], sort=True)
    }

    for condition, trans_stack in trans_stacks.items():
        frame_total = int(trans_stack.shape[0])
        for frame_index in range(frame_total):
            nuclei_frame = nuclei_groups.get((condition, frame_index), pd.DataFrame()).copy()
            bud_frame = bud_groups.get((condition, frame_index), pd.DataFrame()).copy()
            pair_frame = pair_groups.get((condition, frame_index), pd.DataFrame()).copy()
            if nuclei_frame.empty:
                continue

            trans_frame = np.asarray(trans_stack[frame_index], dtype=np.float32)
            high_conf_lookup = {
                int(row.nucleus_id): bool(row.is_high_confidence)
                for row in nuclei_frame.itertuples(index=False)
            }
            nucleus_pos = {
                int(row.nucleus_id): (float(row.centroid_y), float(row.centroid_x))
                for row in nuclei_frame.itertuples(index=False)
            }

            candidate_pairs: list[dict[str, object]] = []
            for row in pair_frame.itertuples(index=False):
                a_id = int(row.nucleus_a_id)
                b_id = int(row.nucleus_b_id)
                if a_id not in nucleus_pos or b_id not in nucleus_pos:
                    continue
                nuc_a = nucleus_pos[a_id]
                nuc_b = nucleus_pos[b_id]
                axis_angle = float(row.nuclei_line_angle_deg)
                budneck_match = None
                if pd.notna(row.nearest_budneck_id) and not bud_frame.empty:
                    matched = bud_frame[bud_frame["budneck_id"] == int(row.nearest_budneck_id)]
                    if not matched.empty:
                        budneck_match = matched.iloc[0]
                budneck_yx = None if budneck_match is None else (float(budneck_match["centroid_y"]), float(budneck_match["centroid_x"]))
                patch_half = int(np.clip(row.distance * 0.7 + 20.0, 28.0, 54.0))
                morph = extract_trans_morphology(
                    trans_frame=trans_frame,
                    seed_points_global=[nuc_a, nuc_b],
                    axis_angle_deg_value=axis_angle,
                    budneck_yx=budneck_yx,
                    split_mode="pair",
                    patch_half_size=patch_half,
                )
                morphology_rows.append(
                    {
                        "condition": condition,
                        "frame": frame_index,
                        "morphology_id": morphology_id,
                        "candidate_kind": "pair_candidate",
                        "nucleus_ids": f"{a_id};{b_id}",
                        "budneck_id": None if budneck_match is None else int(budneck_match["budneck_id"]),
                        "mask_area": morph.mask_area,
                        "contour_point_count": morph.contour_point_count,
                        "lobe_count": morph.lobe_count,
                        "neck_constriction_strength": morph.neck_constriction_strength,
                        "bud_area": morph.bud_area,
                        "mother_area": morph.mother_area,
                        "bud_to_mother_area_ratio": morph.bud_to_mother_area_ratio,
                        "boundary_connectedness": morph.boundary_connectedness,
                        "has_small_attached_bud": morph.has_small_attached_bud,
                        "mask_polarity": morph.mask_polarity,
                        "bbox_y0": morph.bbox_y0,
                        "bbox_x0": morph.bbox_x0,
                        "bbox_y1": morph.bbox_y1,
                        "bbox_x1": morph.bbox_x1,
                        "center_y": morph.center_y,
                        "center_x": morph.center_x,
                        "morphology_support_score": morph.morphology_support_score,
                    }
                )
                ratio_ok = 0.08 <= morph.bud_to_mother_area_ratio <= 1.15
                morphology_ok = (
                    morph.lobe_count >= 2
                    and morph.neck_constriction_strength >= 0.08
                    and morph.boundary_connectedness >= 0.05
                    and (morph.has_small_attached_bud or ratio_ok)
                )
                nuclei_high_count = int(high_conf_lookup.get(a_id, False)) + int(high_conf_lookup.get(b_id, False))
                support_score = float(
                    0.45 * float(row.preliminary_pair_score)
                    + 0.20 * np.clip(1.0 - float(row.angle_difference_from_perpendicular) / 50.0, 0.0, 1.0)
                    + 0.25 * morph.morphology_support_score
                    + 0.10 * min(nuclei_high_count / 2.0, 1.0)
                )
                pair_v1_reject_like = bool(
                    float(row.preliminary_pair_score) < 0.55
                    or float(row.angle_difference_from_perpendicular) > 55.0
                    or not bool(row.budneck_between_nuclei)
                    or float(row.distance) < 15.0
                    or float(row.distance) > 48.0
                )
                candidate_pairs.append(
                    {
                        "condition": condition,
                        "frame": frame_index,
                        "nucleus_a_id": a_id,
                        "nucleus_b_id": b_id,
                        "nearest_budneck_id": None if budneck_match is None else int(budneck_match["budneck_id"]),
                        "distance": float(row.distance),
                        "preliminary_pair_score": float(row.preliminary_pair_score),
                        "angle_difference_from_perpendicular": float(row.angle_difference_from_perpendicular),
                        "budneck_between_nuclei": bool(row.budneck_between_nuclei),
                        "budneck_signal_to_background": float(row.budneck_signal_to_background) if pd.notna(row.budneck_signal_to_background) else np.nan,
                        "morphology_id": morphology_id,
                        "lobe_count": morph.lobe_count,
                        "neck_constriction_strength": morph.neck_constriction_strength,
                        "bud_to_mother_area_ratio": morph.bud_to_mother_area_ratio,
                        "boundary_connectedness": morph.boundary_connectedness,
                        "has_small_attached_bud": morph.has_small_attached_bud,
                        "nuclei_high_confidence_count": nuclei_high_count,
                        "morphology_support_score": morph.morphology_support_score,
                        "combined_support_score": support_score,
                        "pair_v1_reject_like": pair_v1_reject_like,
                        "candidate_accepts_pair": bool(
                            bool(row.budneck_between_nuclei)
                            and 16.0 <= float(row.distance) <= 48.0
                            and float(row.angle_difference_from_perpendicular) <= 45.0
                            and float(row.preliminary_pair_score) >= 0.50
                            and (np.isnan(row.budneck_signal_to_background) or float(row.budneck_signal_to_background) >= 1.8)
                            and morphology_ok
                            and support_score >= 0.54
                            and nuclei_high_count >= 1
                        ),
                    }
                )
                morphology_id += 1

            used_nuclei: set[int] = set()
            used_budnecks: set[int] = set()
            pair_candidates_df = pd.DataFrame(candidate_pairs)
            accepted_pairs = pd.DataFrame()
            if not pair_candidates_df.empty:
                accepted_pairs = pair_candidates_df[pair_candidates_df["candidate_accepts_pair"]].copy()
                accepted_pairs = accepted_pairs.sort_values(
                    ["combined_support_score", "preliminary_pair_score", "morphology_support_score"],
                    ascending=[False, False, False],
                )
                for row in accepted_pairs.itertuples(index=False):
                    a_id = int(row.nucleus_a_id)
                    b_id = int(row.nucleus_b_id)
                    bud_id = None if pd.isna(row.nearest_budneck_id) else int(row.nearest_budneck_id)
                    if a_id in used_nuclei or b_id in used_nuclei:
                        continue
                    if bud_id is not None and bud_id in used_budnecks:
                        continue
                    class_rows.append(
                        {
                            "condition": condition,
                            "frame": frame_index,
                            "target_id": structure_id,
                            "rule_class": "mother_bud_pair",
                            "rule_reason": "two_nuclei_plus_budneck_plus_trans_morphology",
                            "nucleus_a_id": a_id,
                            "nucleus_b_id": b_id,
                            "nearest_budneck_id": bud_id,
                            "nuclei_count": 2,
                            "high_confidence_nuclei_count": int(row.nuclei_high_confidence_count),
                            "low_confidence_nuclei_count": 2 - int(row.nuclei_high_confidence_count),
                            "missing_nucleus_flag": False,
                            "distance": float(row.distance),
                            "preliminary_pair_score": float(row.preliminary_pair_score),
                            "angle_difference_from_perpendicular": float(row.angle_difference_from_perpendicular),
                            "morphology_id": int(row.morphology_id),
                            "morphology_support_score": float(row.morphology_support_score),
                            "combined_support_score": float(row.combined_support_score),
                            "lobe_count": int(row.lobe_count),
                            "neck_constriction_strength": float(row.neck_constriction_strength),
                            "bud_to_mother_area_ratio": float(row.bud_to_mother_area_ratio),
                            "boundary_connectedness": float(row.boundary_connectedness),
                            "has_small_attached_bud": bool(row.has_small_attached_bud),
                            "v1_reject_like": bool(row.pair_v1_reject_like),
                            "source_type": "pair_candidate_v2",
                        }
                    )
                    used_nuclei.update({a_id, b_id})
                    if bud_id is not None:
                        used_budnecks.add(bud_id)
                    structure_id += 1

            for nucleus_row in nuclei_frame.sort_values(["is_high_confidence", "signal_to_background", "max_intensity"], ascending=[False, False, False]).itertuples(index=False):
                nucleus_id = int(nucleus_row.nucleus_id)
                if nucleus_id in used_nuclei:
                    continue
                if not bool(nucleus_row.is_high_confidence):
                    continue
                budneck_choice = nearest_budneck_for_nucleus(pd.Series(nucleus_row._asdict()), bud_frame[~bud_frame["budneck_id"].isin(list(used_budnecks))])
                early_row = None
                if budneck_choice is not None:
                    axis_angle = axis_angle_deg(
                        float(budneck_choice["centroid_y"]) - float(nucleus_row.centroid_y),
                        float(budneck_choice["centroid_x"]) - float(nucleus_row.centroid_x),
                    )
                    morph = extract_trans_morphology(
                        trans_frame=trans_frame,
                        seed_points_global=[(float(nucleus_row.centroid_y), float(nucleus_row.centroid_x))],
                        axis_angle_deg_value=axis_angle,
                        budneck_yx=(float(budneck_choice["centroid_y"]), float(budneck_choice["centroid_x"])),
                        split_mode="single",
                        patch_half_size=36,
                    )
                    morphology_rows.append(
                        {
                            "condition": condition,
                            "frame": frame_index,
                            "morphology_id": morphology_id,
                            "candidate_kind": "single_nucleus_candidate",
                            "nucleus_ids": str(nucleus_id),
                            "budneck_id": int(budneck_choice["budneck_id"]),
                            "mask_area": morph.mask_area,
                            "contour_point_count": morph.contour_point_count,
                            "lobe_count": morph.lobe_count,
                            "neck_constriction_strength": morph.neck_constriction_strength,
                            "bud_area": morph.bud_area,
                            "mother_area": morph.mother_area,
                            "bud_to_mother_area_ratio": morph.bud_to_mother_area_ratio,
                            "boundary_connectedness": morph.boundary_connectedness,
                            "has_small_attached_bud": morph.has_small_attached_bud,
                            "mask_polarity": morph.mask_polarity,
                            "bbox_y0": morph.bbox_y0,
                            "bbox_x0": morph.bbox_x0,
                            "bbox_y1": morph.bbox_y1,
                            "bbox_x1": morph.bbox_x1,
                            "center_y": morph.center_y,
                            "center_x": morph.center_x,
                            "morphology_support_score": morph.morphology_support_score,
                        }
                    )
                    early_support_score = float(
                        0.30 * np.clip(float(budneck_choice["signal_to_background"]) / 4.0, 0.0, 1.0)
                        + 0.25 * np.clip((float(budneck_choice["aspect_ratio"]) - 1.1) / 2.0, 0.0, 1.0)
                        + 0.30 * morph.morphology_support_score
                        + 0.15 * np.clip(float(nucleus_row.signal_to_background) / 5.0, 0.0, 1.0)
                    )
                    early_accept = bool(
                        float(budneck_choice["signal_to_background"]) >= 2.0
                        and morph.bud_area >= 12
                        and morph.mother_area >= 45
                        and 0.05 <= morph.bud_to_mother_area_ratio <= 0.65
                        and morph.neck_constriction_strength >= 0.10
                        and morph.boundary_connectedness >= 0.05
                        and (morph.has_small_attached_bud or morph.lobe_count >= 2)
                        and early_support_score >= 0.50
                    )
                    early_row = {
                        "condition": condition,
                        "frame": frame_index,
                        "target_id": structure_id,
                        "rule_class": "early_bud_pair" if early_accept else "single_cell",
                        "rule_reason": "single_nucleus_plus_budneck_plus_trans_morphology" if early_accept else "single_nucleus_without_strong_budding_support",
                        "nucleus_a_id": nucleus_id,
                        "nucleus_b_id": np.nan,
                        "nearest_budneck_id": int(budneck_choice["budneck_id"]),
                        "nuclei_count": 1,
                        "high_confidence_nuclei_count": 1,
                        "low_confidence_nuclei_count": 0,
                        "missing_nucleus_flag": bool(early_accept),
                        "distance": float(math.hypot(float(budneck_choice["centroid_y"]) - float(nucleus_row.centroid_y), float(budneck_choice["centroid_x"]) - float(nucleus_row.centroid_x))),
                        "preliminary_pair_score": np.nan,
                        "angle_difference_from_perpendicular": np.nan,
                        "morphology_id": morphology_id,
                        "morphology_support_score": float(morph.morphology_support_score),
                        "combined_support_score": early_support_score,
                        "lobe_count": int(morph.lobe_count),
                        "neck_constriction_strength": float(morph.neck_constriction_strength),
                        "bud_to_mother_area_ratio": float(morph.bud_to_mother_area_ratio),
                        "boundary_connectedness": float(morph.boundary_connectedness),
                        "has_small_attached_bud": bool(morph.has_small_attached_bud),
                        "v1_reject_like": bool(early_accept),
                        "source_type": "single_nucleus_candidate",
                    }
                    morphology_id += 1
                    if early_accept:
                        used_nuclei.add(nucleus_id)
                        used_budnecks.add(int(budneck_choice["budneck_id"]))
                if early_row is None:
                    morph = extract_trans_morphology(
                        trans_frame=trans_frame,
                        seed_points_global=[(float(nucleus_row.centroid_y), float(nucleus_row.centroid_x))],
                        axis_angle_deg_value=0.0,
                        budneck_yx=None,
                        split_mode="single",
                        patch_half_size=30,
                    )
                    morphology_rows.append(
                        {
                            "condition": condition,
                            "frame": frame_index,
                            "morphology_id": morphology_id,
                            "candidate_kind": "single_nucleus_candidate",
                            "nucleus_ids": str(nucleus_id),
                            "budneck_id": np.nan,
                            "mask_area": morph.mask_area,
                            "contour_point_count": morph.contour_point_count,
                            "lobe_count": morph.lobe_count,
                            "neck_constriction_strength": morph.neck_constriction_strength,
                            "bud_area": morph.bud_area,
                            "mother_area": morph.mother_area,
                            "bud_to_mother_area_ratio": morph.bud_to_mother_area_ratio,
                            "boundary_connectedness": morph.boundary_connectedness,
                            "has_small_attached_bud": morph.has_small_attached_bud,
                            "mask_polarity": morph.mask_polarity,
                            "bbox_y0": morph.bbox_y0,
                            "bbox_x0": morph.bbox_x0,
                            "bbox_y1": morph.bbox_y1,
                            "bbox_x1": morph.bbox_x1,
                            "center_y": morph.center_y,
                            "center_x": morph.center_x,
                            "morphology_support_score": morph.morphology_support_score,
                        }
                    )
                    early_row = {
                        "condition": condition,
                        "frame": frame_index,
                        "target_id": structure_id,
                        "rule_class": "single_cell",
                        "rule_reason": "isolated_high_confidence_nucleus",
                        "nucleus_a_id": nucleus_id,
                        "nucleus_b_id": np.nan,
                        "nearest_budneck_id": np.nan,
                        "nuclei_count": 1,
                        "high_confidence_nuclei_count": 1,
                        "low_confidence_nuclei_count": 0,
                        "missing_nucleus_flag": False,
                        "distance": np.nan,
                        "preliminary_pair_score": np.nan,
                        "angle_difference_from_perpendicular": np.nan,
                        "morphology_id": morphology_id,
                        "morphology_support_score": float(morph.morphology_support_score),
                        "combined_support_score": float(
                            0.55 * np.clip(float(nucleus_row.signal_to_background) / 6.0, 0.0, 1.0)
                            + 0.45 * float(morph.morphology_support_score)
                        ),
                        "lobe_count": int(morph.lobe_count),
                        "neck_constriction_strength": float(morph.neck_constriction_strength),
                        "bud_to_mother_area_ratio": float(morph.bud_to_mother_area_ratio),
                        "boundary_connectedness": float(morph.boundary_connectedness),
                        "has_small_attached_bud": bool(morph.has_small_attached_bud),
                        "v1_reject_like": False,
                        "source_type": "single_nucleus_candidate",
                    }
                    morphology_id += 1
                class_rows.append(early_row)
                structure_id += 1

    morphology_df = pd.DataFrame(morphology_rows)
    classification_df = pd.DataFrame(class_rows)
    return morphology_df, classification_df


def build_rules_v2_summary(
    classification_df: pd.DataFrame,
    nuclei_v2_df: pd.DataFrame,
    nuclei_v1_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize v2 class counts and recovery metrics."""

    rows: list[dict[str, object]] = []
    class_counts = classification_df["rule_class"].value_counts()
    rows.append(
        {
            "summary_level": "global",
            "condition": None,
            "frame": None,
            "mother_bud_pair": int(class_counts.get("mother_bud_pair", 0)),
            "early_bud_pair": int(class_counts.get("early_bud_pair", 0)),
            "single_cell": int(class_counts.get("single_cell", 0)),
            "nuclei_v1_total": int(len(nuclei_v1_df)),
            "nuclei_v2_total": int(len(nuclei_v2_df)),
            "nuclei_v2_recovered_vs_v1": int(nuclei_v2_df["recovered_vs_v1"].sum()) if not nuclei_v2_df.empty else 0,
            "accepted_v1_reject_like_pairs": int(classification_df[(classification_df["rule_class"] == "mother_bud_pair") & (classification_df["v1_reject_like"] == True)].shape[0]),
        }
    )
    for condition, group in classification_df.groupby("condition", sort=True):
        counts = group["rule_class"].value_counts()
        nuclei_v2_condition = nuclei_v2_df[nuclei_v2_df["condition"] == condition]
        nuclei_v1_condition = nuclei_v1_df[nuclei_v1_df["condition"] == condition]
        rows.append(
            {
                "summary_level": "condition",
                "condition": condition,
                "frame": None,
                "mother_bud_pair": int(counts.get("mother_bud_pair", 0)),
                "early_bud_pair": int(counts.get("early_bud_pair", 0)),
                "single_cell": int(counts.get("single_cell", 0)),
                "nuclei_v1_total": int(len(nuclei_v1_condition)),
                "nuclei_v2_total": int(len(nuclei_v2_condition)),
                "nuclei_v2_recovered_vs_v1": int(nuclei_v2_condition["recovered_vs_v1"].sum()) if not nuclei_v2_condition.empty else 0,
                "accepted_v1_reject_like_pairs": int(group[(group["rule_class"] == "mother_bud_pair") & (group["v1_reject_like"] == True)].shape[0]),
            }
        )
    return pd.DataFrame(rows)
