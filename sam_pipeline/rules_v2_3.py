"""Rules v2.3 with adjacency-aware, bud-centered mother-bud assignment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from .image_utils import percentile_normalize
from .rule_features import axis_angle_deg, axis_difference_deg, distance_point_to_segment


@dataclass
class FrameRegionContext:
    """Per-frame region information used for adjacency-aware rule checks."""

    labels: np.ndarray
    component_labels: np.ndarray
    edge_strength: np.ndarray
    nucleus_to_cell: dict[int, int]
    nucleus_to_component: dict[int, int]
    adjacency_source: str


def _threshold_union(image: np.ndarray) -> np.ndarray:
    """Build a permissive Trans foreground mask for adjacency fallback."""

    image = np.asarray(image, dtype=np.float32)

    def _binary(source: np.ndarray) -> np.ndarray:
        local = source > filters.threshold_local(source, block_size=31, offset=-0.01)
        global_mask = source > filters.threshold_otsu(source)
        binary = local | global_mask
        binary = morphology.binary_closing(binary, morphology.disk(2))
        binary = morphology.binary_opening(binary, morphology.disk(1))
        binary = morphology.remove_small_objects(binary, 40)
        binary = morphology.remove_small_holes(binary, area_threshold=96)
        return np.asarray(binary, dtype=bool)

    smooth = filters.gaussian(image, sigma=1.0, preserve_range=True)
    inv = 1.0 - smooth
    bright = _binary(smooth)
    dark = _binary(inv)
    foreground = bright | dark
    foreground = morphology.binary_closing(foreground, morphology.disk(2))
    foreground = morphology.remove_small_objects(foreground, 56)
    foreground = morphology.remove_small_holes(foreground, area_threshold=128)
    return np.asarray(foreground, dtype=bool)


def _nearest_foreground_point(mask: np.ndarray, center_y: float, center_x: float, max_radius: int = 14) -> tuple[int, int] | None:
    """Return the nearest foreground pixel to a nucleus center."""

    iy = int(round(center_y))
    ix = int(round(center_x))
    if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and bool(mask[iy, ix]):
        return iy, ix
    y0 = max(0, iy - max_radius)
    y1 = min(mask.shape[0], iy + max_radius + 1)
    x0 = max(0, ix - max_radius)
    x1 = min(mask.shape[1], ix + max_radius + 1)
    patch = mask[y0:y1, x0:x1]
    coords = np.argwhere(patch)
    if coords.size == 0:
        return None
    coords = coords.astype(np.float32)
    coords[:, 0] += y0
    coords[:, 1] += x0
    distances = np.sqrt((coords[:, 0] - center_y) ** 2 + (coords[:, 1] - center_x) ** 2)
    best = coords[int(np.argmin(distances))]
    return int(best[0]), int(best[1])


def _assign_nucleus_to_label(label_image: np.ndarray, center_y: float, center_x: float, max_radius: int = 14) -> int:
    """Assign a nucleus center to the nearest positive label."""

    iy = int(round(center_y))
    ix = int(round(center_x))
    if 0 <= iy < label_image.shape[0] and 0 <= ix < label_image.shape[1]:
        label = int(label_image[iy, ix])
        if label > 0:
            return label
    y0 = max(0, iy - max_radius)
    y1 = min(label_image.shape[0], iy + max_radius + 1)
    x0 = max(0, ix - max_radius)
    x1 = min(label_image.shape[1], ix + max_radius + 1)
    patch = label_image[y0:y1, x0:x1]
    coords = np.argwhere(patch > 0)
    if coords.size == 0:
        return 0
    coords = coords.astype(np.float32)
    coords[:, 0] += y0
    coords[:, 1] += x0
    distances = np.sqrt((coords[:, 0] - center_y) ** 2 + (coords[:, 1] - center_x) ** 2)
    best = coords[int(np.argmin(distances))]
    return int(label_image[int(best[0]), int(best[1])])


def build_frame_region_context(
    trans_frame: np.ndarray,
    nuclei_frame: pd.DataFrame,
    sam_mask_frame: np.ndarray | None = None,
) -> FrameRegionContext:
    """Build a frame-level adjacency context from masks or Trans morphology."""

    trans_frame = np.asarray(trans_frame, dtype=np.float32)
    norm = percentile_normalize(trans_frame, low=1.0, high=99.6)
    edge_strength = np.asarray(filters.sobel(filters.gaussian(norm, sigma=1.0, preserve_range=True)), dtype=np.float32)

    if sam_mask_frame is not None:
        labels = np.asarray(sam_mask_frame, dtype=np.int32)
        if labels.ndim == 3:
            labels = labels.max(axis=0)
        if labels.shape == trans_frame.shape and int(labels.max()) > 0:
            component_labels = measure.label(labels > 0)
            nucleus_to_cell: dict[int, int] = {}
            nucleus_to_component: dict[int, int] = {}
            for row in nuclei_frame.itertuples(index=False):
                cell_id = _assign_nucleus_to_label(labels, float(row.centroid_y), float(row.centroid_x), max_radius=10)
                nucleus_to_cell[int(row.nucleus_id)] = cell_id
                nucleus_to_component[int(row.nucleus_id)] = _assign_nucleus_to_label(component_labels, float(row.centroid_y), float(row.centroid_x), max_radius=10)
            return FrameRegionContext(
                labels=labels,
                component_labels=component_labels,
                edge_strength=edge_strength,
                nucleus_to_cell=nucleus_to_cell,
                nucleus_to_component=nucleus_to_component,
                adjacency_source="existing_sam_masks",
            )

    foreground = _threshold_union(norm)
    component_labels = measure.label(foreground)
    markers = np.zeros_like(component_labels, dtype=np.int32)
    nucleus_to_cell: dict[int, int] = {}
    nucleus_to_component: dict[int, int] = {}
    next_marker = 1
    occupied: set[tuple[int, int]] = set()
    for row in nuclei_frame.sort_values(["is_high_confidence", "nucleus_id"], ascending=[False, True]).itertuples(index=False):
        nearest = _nearest_foreground_point(foreground, float(row.centroid_y), float(row.centroid_x), max_radius=14)
        if nearest is None:
            nucleus_to_cell[int(row.nucleus_id)] = 0
            nucleus_to_component[int(row.nucleus_id)] = 0
            continue
        py, px = nearest
        if (py, px) in occupied:
            found = None
            for radius in range(1, 4):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        qy = int(np.clip(py + dy, 0, markers.shape[0] - 1))
                        qx = int(np.clip(px + dx, 0, markers.shape[1] - 1))
                        if not bool(foreground[qy, qx]) or (qy, qx) in occupied:
                            continue
                        found = (qy, qx)
                        break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is not None:
                py, px = found
        occupied.add((py, px))
        markers[py, px] = next_marker
        nucleus_to_cell[int(row.nucleus_id)] = next_marker
        nucleus_to_component[int(row.nucleus_id)] = int(component_labels[py, px]) if int(component_labels[py, px]) > 0 else 0
        next_marker += 1

    labels = segmentation.watershed(edge_strength, markers, mask=foreground)
    for nucleus_id, cell_id in list(nucleus_to_cell.items()):
        if cell_id <= 0:
            continue
        cyx = nuclei_frame.loc[nuclei_frame["nucleus_id"] == nucleus_id, ["centroid_y", "centroid_x"]]
        if cyx.empty:
            continue
        cy = float(cyx.iloc[0]["centroid_y"])
        cx = float(cyx.iloc[0]["centroid_x"])
        assigned = _assign_nucleus_to_label(labels, cy, cx, max_radius=12)
        nucleus_to_cell[nucleus_id] = assigned
        nucleus_to_component[nucleus_id] = _assign_nucleus_to_label(component_labels, cy, cx, max_radius=12)

    return FrameRegionContext(
        labels=np.asarray(labels, dtype=np.int32),
        component_labels=np.asarray(component_labels, dtype=np.int32),
        edge_strength=edge_strength,
        nucleus_to_cell=nucleus_to_cell,
        nucleus_to_component=nucleus_to_component,
        adjacency_source="trans_morphology_fallback",
    )


def _pair_boundary_metrics(
    labels: np.ndarray,
    edge_strength: np.ndarray,
    cell_a_id: int,
    cell_b_id: int,
) -> tuple[bool, float, int, float]:
    """Measure whether two assigned cell regions are locally adjacent."""

    if cell_a_id <= 0 or cell_b_id <= 0:
        return False, float("inf"), 0, 0.0
    if cell_a_id == cell_b_id:
        return True, 0.0, 0, 0.0

    mask_a = labels == cell_a_id
    mask_b = labels == cell_b_id
    if not np.any(mask_a) or not np.any(mask_b):
        return False, float("inf"), 0, 0.0

    boundary_a = segmentation.find_boundaries(mask_a, mode="outer")
    boundary_b = segmentation.find_boundaries(mask_b, mode="outer")
    if not np.any(boundary_a):
        boundary_a = mask_a
    if not np.any(boundary_b):
        boundary_b = mask_b

    dil_a = morphology.binary_dilation(boundary_a, morphology.disk(1))
    dil_b = morphology.binary_dilation(boundary_b, morphology.disk(1))
    touch = dil_a & dil_b
    shared_boundary_length = int(np.count_nonzero(touch))

    distance_to_b = ndi.distance_transform_edt(~mask_b)
    cell_boundary_distance = float(np.min(distance_to_b[boundary_a])) if np.any(boundary_a) else float("inf")

    if shared_boundary_length > 0:
        denom = float(np.percentile(edge_strength, 95.0)) + 1e-6
        shared_boundary_strength = float(np.clip(np.mean(edge_strength[touch]) / denom, 0.0, 1.0))
    else:
        shared_boundary_strength = float(np.exp(-cell_boundary_distance / 2.5))

    cells_are_adjacent = bool(shared_boundary_length > 0 or cell_boundary_distance <= 2.5)
    return cells_are_adjacent, cell_boundary_distance, shared_boundary_length, shared_boundary_strength


def build_cell_adjacency_features(
    trans_stacks: dict[str, np.ndarray],
    nuclei_v2_df: pd.DataFrame,
    pair_base_df: pd.DataFrame,
    sam_mask_stacks: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, object]]]:
    """Assign nuclei to Trans/SAM regions and compute adjacency features for each pair."""

    sam_mask_stacks = sam_mask_stacks or {}
    nuclei_groups = {
        (str(condition), int(frame)): group.copy()
        for (condition, frame), group in nuclei_v2_df.groupby(["condition", "frame"], sort=True)
    }

    frame_meta: dict[tuple[str, int], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for (condition, frame), pair_frame in pair_base_df.groupby(["condition", "frame"], sort=True):
        nuclei_frame = nuclei_groups.get((str(condition), int(frame)), pd.DataFrame())
        if nuclei_frame.empty:
            continue
        sam_mask_frame = None
        if condition in sam_mask_stacks:
            stack = np.asarray(sam_mask_stacks[condition])
            if stack.ndim >= 3 and int(frame) < int(stack.shape[0]):
                sam_mask_frame = stack[int(frame)]
        context = build_frame_region_context(
            trans_frame=trans_stacks[str(condition)][int(frame)],
            nuclei_frame=nuclei_frame,
            sam_mask_frame=sam_mask_frame,
        )
        frame_meta[(str(condition), int(frame))] = {
            "nucleus_to_cell": context.nucleus_to_cell,
            "nucleus_to_component": context.nucleus_to_component,
            "adjacency_source": context.adjacency_source,
        }
        for row in pair_frame.itertuples(index=False):
            nucleus_a_id = int(row.nucleus_a_id)
            nucleus_b_id = int(row.nucleus_b_id)
            cell_a_id = int(context.nucleus_to_cell.get(nucleus_a_id, 0))
            cell_b_id = int(context.nucleus_to_cell.get(nucleus_b_id, 0))
            component_a_id = int(context.nucleus_to_component.get(nucleus_a_id, 0))
            component_b_id = int(context.nucleus_to_component.get(nucleus_b_id, 0))
            same_cell_region = bool(cell_a_id > 0 and cell_a_id == cell_b_id)
            cells_are_adjacent, boundary_distance, shared_length, shared_strength = _pair_boundary_metrics(
                context.labels,
                context.edge_strength,
                cell_a_id,
                cell_b_id,
            )
            same_component = bool(component_a_id > 0 and component_a_id == component_b_id)
            adjacency_supports_pair = bool(
                same_cell_region
                or (
                    cells_are_adjacent
                    and (
                        shared_length >= 3
                        or boundary_distance <= 1.5
                        or same_component
                    )
                )
            )
            rows.append(
                {
                    "condition": str(condition),
                    "frame": int(frame),
                    "candidate_id": int(row.candidate_id),
                    "nucleus_a_id": nucleus_a_id,
                    "nucleus_b_id": nucleus_b_id,
                    "cell_a_id": cell_a_id,
                    "cell_b_id": cell_b_id,
                    "same_cell_region": same_cell_region,
                    "cells_are_adjacent": cells_are_adjacent,
                    "cell_boundary_distance": float(boundary_distance),
                    "shared_boundary_length": int(shared_length),
                    "shared_boundary_strength": float(shared_strength),
                    "cell_wall_connectedness_score": float(row.cell_wall_connectedness_score),
                    "adjacency_source": context.adjacency_source,
                    "adjacency_supports_pair": adjacency_supports_pair,
                }
            )
    return pd.DataFrame(rows), frame_meta


def build_nucleus_line_budneck_geometry(
    nuclei_v2_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    pair_base_df: pd.DataFrame,
    distance_max: float,
    projection_min: float,
    projection_max: float,
    perpendicular_tolerance_deg: float,
) -> pd.DataFrame:
    """Measure whether the nucleus line crosses the candidate bud neck."""

    nuclei_groups = {
        (str(condition), int(frame)): group.copy()
        for (condition, frame), group in nuclei_v2_df.groupby(["condition", "frame"], sort=True)
    }
    bud_groups = {
        (str(condition), int(frame)): group.copy()
        for (condition, frame), group in budneck_df.groupby(["condition", "frame"], sort=True)
    }
    rows: list[dict[str, object]] = []
    for row in pair_base_df.itertuples(index=False):
        nuclei_frame = nuclei_groups.get((str(row.condition), int(row.frame)), pd.DataFrame())
        bud_frame = bud_groups.get((str(row.condition), int(row.frame)), pd.DataFrame())
        a_match = nuclei_frame[nuclei_frame["nucleus_id"] == int(row.nucleus_a_id)]
        b_match = nuclei_frame[nuclei_frame["nucleus_id"] == int(row.nucleus_b_id)]
        if a_match.empty or b_match.empty:
            continue
        bud_id = int(row.nearest_budneck_id) if pd.notna(row.nearest_budneck_id) else 0
        bud_match = bud_frame[bud_frame["budneck_id"] == bud_id] if bud_id > 0 else pd.DataFrame()
        if bud_match.empty:
            rows.append(
                {
                    "condition": str(row.condition),
                    "frame": int(row.frame),
                    "candidate_id": int(row.candidate_id),
                    "nucleus_a_id": int(row.nucleus_a_id),
                    "nucleus_b_id": int(row.nucleus_b_id),
                    "budneck_id": np.nan,
                    "nucleus_line_angle_deg": float(row.nuclei_line_angle_deg),
                    "budneck_angle_deg": np.nan,
                    "angle_difference_from_perpendicular": np.nan,
                    "budneck_distance_to_nucleus_line": np.nan,
                    "budneck_projection_fraction": np.nan,
                    "budneck_between_nuclei": False,
                    "nucleus_line_hits_budneck": False,
                    "line_neck_supports_pair": False,
                }
            )
            continue
        bud = bud_match.iloc[0]
        a_yx = (float(a_match.iloc[0]["centroid_y"]), float(a_match.iloc[0]["centroid_x"]))
        b_yx = (float(b_match.iloc[0]["centroid_y"]), float(b_match.iloc[0]["centroid_x"]))
        bud_yx = (float(bud["centroid_y"]), float(bud["centroid_x"]))
        line_distance, projection_fraction = distance_point_to_segment(bud_yx, a_yx, b_yx)
        angle_difference = axis_difference_deg(float(bud["orientation_deg"]), (float(row.nuclei_line_angle_deg) + 90.0) % 180.0)
        budneck_between = bool(0.0 <= projection_fraction <= 1.0)
        nucleus_line_hits = bool(
            projection_min <= projection_fraction <= projection_max
            and line_distance <= distance_max
        )
        line_supports = bool(
            nucleus_line_hits
            and angle_difference <= perpendicular_tolerance_deg
        )
        rows.append(
            {
                "condition": str(row.condition),
                "frame": int(row.frame),
                "candidate_id": int(row.candidate_id),
                "nucleus_a_id": int(row.nucleus_a_id),
                "nucleus_b_id": int(row.nucleus_b_id),
                "budneck_id": int(bud["budneck_id"]),
                "nucleus_line_angle_deg": float(row.nuclei_line_angle_deg),
                "budneck_angle_deg": float(bud["orientation_deg"]),
                "angle_difference_from_perpendicular": float(angle_difference),
                "budneck_distance_to_nucleus_line": float(line_distance),
                "budneck_projection_fraction": float(projection_fraction),
                "budneck_between_nuclei": budneck_between,
                "nucleus_line_hits_budneck": nucleus_line_hits,
                "line_neck_supports_pair": line_supports,
            }
        )
    return pd.DataFrame(rows)


def _build_pair_candidate_table(
    pair_v2_df: pd.DataFrame,
    cell_wall_df: pd.DataFrame,
    pair_debug_v2_1_df: pd.DataFrame,
    v2_1_classification_df: pd.DataFrame,
    adjacency_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge pair candidates with v2.1 and new v2.3 feature layers."""

    base = cell_wall_df.merge(
        pair_v2_df,
        on=["condition", "frame", "nucleus_a_id", "nucleus_b_id", "distance"],
        how="left",
        suffixes=("", "_pair"),
    )
    pair_debug = pair_debug_v2_1_df.rename(columns={"selected_pair_id": "candidate_id"})
    base = base.merge(
        pair_debug[
            [
                "condition",
                "frame",
                "candidate_id",
                "pair_score",
                "budneck_score",
                "is_mutual_nearest",
                "is_mutual_top2",
                "possible_wrong_far_pair_flag",
                "assignment_reason",
                "downgrade_reason",
            ]
        ],
        on=["condition", "frame", "candidate_id"],
        how="left",
    )
    previous = v2_1_classification_df[
        ["condition", "frame", "candidate_id", "final_class", "source_type"]
    ].rename(columns={"final_class": "previous_v2_1_class", "source_type": "previous_source_type"})
    base = base.merge(previous, on=["condition", "frame", "candidate_id"], how="left")
    adjacency_keep = adjacency_df[
        [
            "condition",
            "frame",
            "candidate_id",
            "nucleus_a_id",
            "nucleus_b_id",
            "cell_a_id",
            "cell_b_id",
            "same_cell_region",
            "cells_are_adjacent",
            "cell_boundary_distance",
            "shared_boundary_length",
            "shared_boundary_strength",
            "adjacency_source",
            "adjacency_supports_pair",
        ]
    ]
    geometry_keep = geometry_df[
        [
            "condition",
            "frame",
            "candidate_id",
            "nucleus_a_id",
            "nucleus_b_id",
            "budneck_id",
            "angle_difference_from_perpendicular",
            "budneck_distance_to_nucleus_line",
            "budneck_projection_fraction",
            "budneck_between_nuclei",
            "nucleus_line_hits_budneck",
            "line_neck_supports_pair",
        ]
    ].rename(
        columns={
            "budneck_id": "geometry_budneck_id",
            "budneck_between_nuclei": "geometry_budneck_between_nuclei",
            "angle_difference_from_perpendicular": "geometry_angle_difference_from_perpendicular",
            "budneck_projection_fraction": "geometry_budneck_projection_fraction",
        }
    )
    base = base.merge(adjacency_keep, on=["condition", "frame", "candidate_id", "nucleus_a_id", "nucleus_b_id"], how="left")
    base = base.merge(geometry_keep, on=["condition", "frame", "candidate_id", "nucleus_a_id", "nucleus_b_id"], how="left")
    base["budneck_between_nuclei"] = base["geometry_budneck_between_nuclei"].fillna(base["budneck_between_nuclei"])
    base["angle_difference_from_perpendicular"] = base["geometry_angle_difference_from_perpendicular"].fillna(base["angle_difference_from_perpendicular"])
    base["budneck_projection_fraction"] = base["geometry_budneck_projection_fraction"].fillna(base["budneck_projection_fraction"])
    if "nearest_budneck_id" in base.columns:
        base["nearest_budneck_id"] = base["nearest_budneck_id"].fillna(base["geometry_budneck_id"])
    base = base.drop(
        columns=[
            "geometry_budneck_id",
            "geometry_budneck_between_nuclei",
            "geometry_angle_difference_from_perpendicular",
            "geometry_budneck_projection_fraction",
        ]
    )
    return base


def _local_partner_flags(group: pd.DataFrame) -> pd.DataFrame:
    """Add bud-centered wrong-partner diagnostics inside one frame."""

    group = group.copy()
    group["closer_adjacent_partner_exists"] = False
    group["selected_partner_is_farther_than_alternative"] = False
    group["wrong_far_assignment_warning"] = False
    group["local_budneck_rank"] = np.nan

    for bud_id, bud_group in group.groupby("nearest_budneck_id", dropna=True, sort=False):
        ordered = bud_group.sort_values(
            [
                "cells_are_adjacent",
                "nucleus_line_hits_budneck",
                "budneck_between_nuclei",
                "is_mutual_top2",
                "distance",
                "cell_wall_connectedness_score",
                "angle_difference_from_perpendicular",
                "budneck_score",
                "pair_score",
            ],
            ascending=[False, False, False, False, True, False, True, False, False],
        ).copy()
        for rank, row in enumerate(ordered.itertuples(index=False), start=1):
            group.loc[group["candidate_id"] == row.candidate_id, "local_budneck_rank"] = rank
        for row in ordered.itertuples(index=False):
            alternatives = ordered[
                ordered["candidate_id"] != row.candidate_id
            ].copy()
            share_nucleus = alternatives[
                (alternatives["nucleus_a_id"] == row.nucleus_a_id)
                | (alternatives["nucleus_b_id"] == row.nucleus_a_id)
                | (alternatives["nucleus_a_id"] == row.nucleus_b_id)
                | (alternatives["nucleus_b_id"] == row.nucleus_b_id)
            ]
            closer_alt = share_nucleus[
                share_nucleus["cells_are_adjacent"].astype(bool)
                & share_nucleus["nucleus_line_hits_budneck"].astype(bool)
                & (share_nucleus["distance"] + 2.0 < float(row.distance))
            ]
            closer_exists = not closer_alt.empty
            farther_than_alt = bool(closer_exists and float(row.distance) > float(closer_alt["distance"].min()))
            wrong_far = bool(
                farther_than_alt
                and (not bool(row.is_mutual_top2) or int(group.loc[group["candidate_id"] == row.candidate_id, "local_budneck_rank"].iloc[0]) > 1)
            )
            group.loc[group["candidate_id"] == row.candidate_id, "closer_adjacent_partner_exists"] = closer_exists
            group.loc[group["candidate_id"] == row.candidate_id, "selected_partner_is_farther_than_alternative"] = farther_than_alt
            group.loc[group["candidate_id"] == row.candidate_id, "wrong_far_assignment_warning"] = wrong_far
    return group


def _pair_priority_dataframe(group: pd.DataFrame) -> pd.DataFrame:
    """Return candidates sorted by biological ranking priority."""

    return group.sort_values(
        [
            "cells_are_adjacent",
            "nucleus_line_hits_budneck",
            "budneck_between_nuclei",
            "closer_adjacent_partner_exists",
            "is_mutual_top2",
            "distance",
            "cell_wall_connectedness_score",
            "angle_difference_from_perpendicular",
            "budneck_score",
            "pair_score",
        ],
        ascending=[False, False, False, True, False, True, False, True, False, False],
    )


def _classify_pair_rows(
    pair_df: pd.DataFrame,
    pair_require_mutual_top_k: int,
    pair_ideal_distance_min: float,
    pair_ideal_distance_max: float,
    pair_max_distance: float,
    far_pair_extra_score_required: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply adjacency-aware pair classification and global conflict resolution."""

    pair_df = pair_df.copy()
    pair_df["decision_reason"] = ""
    pair_df["downgrade_reason"] = ""
    pair_df["final_class"] = "uncertain_pair"
    pair_df["strict_mother_candidate"] = False

    debug_rows: list[dict[str, object]] = []
    selected_rows: list[pd.Series] = []

    for (condition, frame), group in pair_df.groupby(["condition", "frame"], sort=True):
        group = _local_partner_flags(group)
        group["strict_mother_candidate"] = False
        group["decision_reason"] = ""
        group["downgrade_reason"] = ""

        for idx, row in group.iterrows():
            distance = float(row["distance"])
            far_pair = distance > pair_ideal_distance_max
            distance_ok = pair_ideal_distance_min <= distance <= pair_max_distance
            budneck_between = bool(row["budneck_between_nuclei"])
            line_hits = bool(row["nucleus_line_hits_budneck"])
            adjacency_ok = bool(row["adjacency_supports_pair"])
            cells_adjacent = bool(row["cells_are_adjacent"]) or bool(row["same_cell_region"])
            mutual_ok = bool(row["is_mutual_nearest"]) or bool(row["is_mutual_top2"])
            wall_ok = bool(
                float(row["cell_wall_connectedness_score"]) >= 0.55
                and float(row["wall_continuity_score"]) >= 0.20
                and float(row["neck_constriction_score"]) >= 0.02
            )
            if far_pair:
                wall_ok = wall_ok and float(row["pair_score"]) >= 0.60 + far_pair_extra_score_required
            same_component_ok = bool(row["same_trans_component"]) or bool(row["same_cell_region"])
            better_local_partner = bool(row["closer_adjacent_partner_exists"])

            strict_candidate = bool(
                distance_ok
                and cells_adjacent
                and adjacency_ok
                and budneck_between
                and line_hits
                and same_component_ok
                and wall_ok
                and (
                    mutual_ok
                    or (
                        bool(row["cells_are_adjacent"])
                        and float(row["shared_boundary_length"]) >= 3
                        and float(row["shared_boundary_strength"]) >= 0.25
                    )
                )
                and not better_local_partner
            )

            reasons: list[str] = []
            if not cells_adjacent or not adjacency_ok:
                reasons.append("nonadjacent_cells")
            if not budneck_between:
                reasons.append("budneck_not_between_nuclei")
            if not line_hits:
                reasons.append("line_misses_budneck")
            if not same_component_ok:
                reasons.append("same_trans_component_false")
            if not wall_ok:
                reasons.append("cell_wall_support_weak")
            if not mutual_ok and float(row["shared_boundary_length"]) < 3:
                reasons.append(f"not_mutual_top{pair_require_mutual_top_k}")
            if far_pair:
                reasons.append("far_pair")
            if better_local_partner:
                reasons.append("closer_adjacent_partner_exists")
            if bool(row["wrong_far_assignment_warning"]):
                reasons.append("wrong_far_assignment_warning")

            group.at[idx, "strict_mother_candidate"] = strict_candidate
            if strict_candidate:
                group.at[idx, "decision_reason"] = "adjacent_cells+bud_between_nuclei+line_hits_budneck+wall_support"
                group.at[idx, "final_class"] = "mother_bud_pair"
            elif not cells_adjacent or not adjacency_ok:
                group.at[idx, "decision_reason"] = "rejected_nonadjacent_or_unconnected_cells"
                group.at[idx, "final_class"] = "rejected_nonadjacent_pair"
            elif budneck_between and line_hits and (wall_ok or same_component_ok):
                group.at[idx, "decision_reason"] = "partial_budding_evidence_but_conflicting_partner_or_geometry"
                group.at[idx, "final_class"] = "uncertain_pair"
            else:
                group.at[idx, "decision_reason"] = "insufficient_pair_evidence"
                group.at[idx, "final_class"] = "uncertain_pair"
            group.at[idx, "downgrade_reason"] = ";".join(sorted(set(reasons)))

        accepted = _pair_priority_dataframe(group[group["strict_mother_candidate"]].copy())
        used_nuclei: set[int] = set()
        used_budnecks: set[int] = set()
        for row in accepted.itertuples(index=False):
            bud_id = int(row.nearest_budneck_id) if pd.notna(row.nearest_budneck_id) else 0
            a_id = int(row.nucleus_a_id)
            b_id = int(row.nucleus_b_id)
            if a_id in used_nuclei or b_id in used_nuclei or (bud_id > 0 and bud_id in used_budnecks):
                mask = group["candidate_id"] == int(row.candidate_id)
                group.loc[mask, "final_class"] = "uncertain_pair"
                current = str(group.loc[mask, "downgrade_reason"].iloc[0])
                extra = "conflict_lost_to_higher_ranked_pair"
                group.loc[mask, "downgrade_reason"] = extra if not current else f"{current};{extra}"
                group.loc[mask, "decision_reason"] = "conflict_with_higher_ranked_adjacent_pair"
                continue
            used_nuclei.update({a_id, b_id})
            if bud_id > 0:
                used_budnecks.add(bud_id)
            selected_rows.append(group.loc[group["candidate_id"] == int(row.candidate_id)].iloc[0])

        frame_debug = group.copy()
        frame_debug["assignment_priority"] = range(1, len(frame_debug) + 1)
        debug_rows.extend(frame_debug.to_dict(orient="records"))

    debug_df = pd.DataFrame(debug_rows)
    selected_df = pd.DataFrame(selected_rows)
    return debug_df, selected_df


def _build_early_bud_rows(
    v2_1_classification_df: pd.DataFrame,
    nuclei_v2_df: pd.DataFrame,
    frame_meta: dict[tuple[str, int], dict[str, object]],
    selected_mother_df: pd.DataFrame,
    pair_debug_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only bud-centered early-bud cases with no valid second nucleus."""

    early_source = v2_1_classification_df[
        v2_1_classification_df["source_type"] == "single_nucleus_bud_candidate"
    ].copy()
    if early_source.empty:
        return early_source

    used_budnecks = set(
        int(value)
        for value in selected_mother_df["nearest_budneck_id"].dropna().astype(int).tolist()
    )
    used_nuclei = set(int(value) for value in selected_mother_df["nucleus_a_id"].dropna().astype(int).tolist())
    used_nuclei.update(int(value) for value in selected_mother_df["nucleus_b_id"].dropna().astype(int).tolist())

    pair_groups = {
        (str(condition), int(frame), int(bud_id)): group.copy()
        for (condition, frame, bud_id), group in pair_debug_df.dropna(subset=["nearest_budneck_id"]).groupby(
            ["condition", "frame", "nearest_budneck_id"],
            sort=True,
        )
    }

    kept_rows: list[dict[str, object]] = []
    nuclei_lookup = {
        (str(row.condition), int(row.frame), int(row.nucleus_id)): row
        for row in nuclei_v2_df.itertuples(index=False)
    }
    for row in early_source.itertuples(index=False):
        bud_id = int(row.nearest_budneck_id) if pd.notna(row.nearest_budneck_id) else 0
        nucleus_id = int(row.nucleus_a_id)
        if bud_id <= 0 or bud_id in used_budnecks or nucleus_id in used_nuclei:
            continue
        pair_group = pair_groups.get((str(row.condition), int(row.frame), bud_id), pd.DataFrame())
        valid_second_nucleus = False
        if not pair_group.empty:
            local = pair_group[
                (
                    (pair_group["nucleus_a_id"] == nucleus_id)
                    | (pair_group["nucleus_b_id"] == nucleus_id)
                )
                & pair_group["cells_are_adjacent"].astype(bool)
                & pair_group["nucleus_line_hits_budneck"].astype(bool)
                & pair_group["budneck_between_nuclei"].astype(bool)
                & (~pair_group["closer_adjacent_partner_exists"].astype(bool))
            ]
            valid_second_nucleus = not local.empty
        nucleus_key = (str(row.condition), int(row.frame), nucleus_id)
        nucleus_info = nuclei_lookup.get(nucleus_key)
        low_confidence_ok = nucleus_info is not None and bool(getattr(nucleus_info, "is_high_confidence", False))
        morphology_ok = bool(
            row.bud_lobe_detected
            and bool(row.same_trans_component)
            and float(row.cell_wall_connectedness_score) >= 0.55
            and float(row.wall_continuity_score) >= 0.18
            and 0.05 <= float(row.mother_bud_area_ratio) <= 0.95
        )
        if valid_second_nucleus or not morphology_ok or not low_confidence_ok:
            continue
        meta = frame_meta.get((str(row.condition), int(row.frame)), {})
        cell_a_id = int(meta.get("nucleus_to_cell", {}).get(nucleus_id, 0))
        kept_rows.append(
            {
                "condition": str(row.condition),
                "frame": int(row.frame),
                "candidate_id": int(row.candidate_id),
                "budneck_id": bud_id,
                "nucleus_a_id": nucleus_id,
                "nucleus_b_id": np.nan,
                "cell_a_id": cell_a_id,
                "cell_b_id": np.nan,
                "distance": float(row.distance),
                "pair_score": float(row.pair_score),
                "budneck_score": float(row.budneck_score),
                "cells_are_adjacent": True,
                "cell_boundary_distance": 0.0,
                "shared_boundary_length": 0,
                "nucleus_line_hits_budneck": True,
                "budneck_distance_to_nucleus_line": np.nan,
                "budneck_projection_fraction": float(row.budneck_projection_fraction),
                "budneck_between_nuclei": True,
                "closer_adjacent_partner_exists": False,
                "wrong_far_assignment_warning": False,
                "same_trans_component": bool(row.same_trans_component),
                "cell_wall_connectedness_score": float(row.cell_wall_connectedness_score),
                "neck_constriction_score": float(row.neck_constriction_score),
                "wall_continuity_score": float(row.wall_continuity_score),
                "same_cell_region": True,
                "adjacency_supports_pair": True,
                "shared_boundary_strength": np.nan,
                "bud_lobe_detected": bool(row.bud_lobe_detected),
                "mother_bud_area_ratio": float(row.mother_bud_area_ratio),
                "is_mutual_top2": False,
                "previous_v2_1_class": str(row.final_class),
                "final_class": "early_bud_pair",
                "decision_reason": "one_nucleus+bud_lobe+budneck+no_valid_second_nucleus",
                "downgrade_reason": "",
                "nearest_budneck_id": bud_id,
                "angle_difference_from_perpendicular": np.nan,
                "contour_yx_json": str(row.contour_yx_json),
                "mask_bbox_y0": float(row.mask_bbox_y0),
                "mask_bbox_x0": float(row.mask_bbox_x0),
                "mask_bbox_y1": float(row.mask_bbox_y1),
                "mask_bbox_x1": float(row.mask_bbox_x1),
                "source_type": "single_nucleus_bud_candidate",
                "adjacency_source": str(meta.get("adjacency_source", "trans_morphology_fallback")),
            }
        )
        used_nuclei.add(nucleus_id)
        used_budnecks.add(bud_id)
    return pd.DataFrame(kept_rows)


def _build_single_cell_rows(
    nuclei_v2_df: pd.DataFrame,
    frame_meta: dict[tuple[str, int], dict[str, object]],
    used_nuclei: set[tuple[str, int, int]],
    start_candidate_id: int,
) -> pd.DataFrame:
    """Build single-cell rows for nuclei not used in pair or early-bud classes."""

    rows: list[dict[str, object]] = []
    next_candidate_id = int(start_candidate_id)
    for row in nuclei_v2_df.itertuples(index=False):
        key = (str(row.condition), int(row.frame), int(row.nucleus_id))
        if key in used_nuclei:
            continue
        meta = frame_meta.get((str(row.condition), int(row.frame)), {})
        cell_a_id = int(meta.get("nucleus_to_cell", {}).get(int(row.nucleus_id), 0))
        rows.append(
            {
                "condition": str(row.condition),
                "frame": int(row.frame),
                "candidate_id": next_candidate_id,
                "budneck_id": np.nan,
                "nucleus_a_id": int(row.nucleus_id),
                "nucleus_b_id": np.nan,
                "cell_a_id": cell_a_id,
                "cell_b_id": np.nan,
                "distance": np.nan,
                "pair_score": np.nan,
                "budneck_score": np.nan,
                "cells_are_adjacent": False,
                "cell_boundary_distance": np.nan,
                "shared_boundary_length": 0,
                "nucleus_line_hits_budneck": False,
                "budneck_distance_to_nucleus_line": np.nan,
                "budneck_projection_fraction": np.nan,
                "budneck_between_nuclei": False,
                "closer_adjacent_partner_exists": False,
                "wrong_far_assignment_warning": False,
                "same_trans_component": False,
                "cell_wall_connectedness_score": np.nan,
                "neck_constriction_score": np.nan,
                "wall_continuity_score": np.nan,
                "same_cell_region": False,
                "adjacency_supports_pair": False,
                "shared_boundary_strength": np.nan,
                "bud_lobe_detected": False,
                "mother_bud_area_ratio": np.nan,
                "is_mutual_top2": False,
                "previous_v2_1_class": np.nan,
                "final_class": "single_cell",
                "decision_reason": "no_adjacent_budneck_supported_pair",
                "downgrade_reason": "",
                "nearest_budneck_id": np.nan,
                "angle_difference_from_perpendicular": np.nan,
                "contour_yx_json": "[]",
                "mask_bbox_y0": np.nan,
                "mask_bbox_x0": np.nan,
                "mask_bbox_y1": np.nan,
                "mask_bbox_x1": np.nan,
                "source_type": "single_from_unused_nucleus",
                "adjacency_source": str(meta.get("adjacency_source", "trans_morphology_fallback")),
            }
        )
        next_candidate_id += 1
    return pd.DataFrame(rows)


def build_rules_v2_3(
    trans_stacks: dict[str, np.ndarray],
    nuclei_v2_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    pair_v2_df: pd.DataFrame,
    cell_wall_df: pd.DataFrame,
    pair_debug_v2_1_df: pd.DataFrame,
    v2_1_classification_df: pd.DataFrame,
    sam_mask_stacks: dict[str, np.ndarray] | None,
    pair_require_mutual_top_k: int,
    pair_ideal_distance_min: float,
    pair_ideal_distance_max: float,
    pair_max_distance: float,
    far_pair_extra_score_required: float,
    budneck_line_distance_max: float,
    budneck_projection_min: float,
    budneck_projection_max: float,
    budneck_perpendicular_tolerance_deg: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build adjacency-aware v2.3 tables and final biological classes."""

    pair_base_df = cell_wall_df.merge(
        pair_v2_df,
        on=["condition", "frame", "nucleus_a_id", "nucleus_b_id", "distance"],
        how="left",
    )
    adjacency_df, frame_meta = build_cell_adjacency_features(
        trans_stacks=trans_stacks,
        nuclei_v2_df=nuclei_v2_df,
        pair_base_df=pair_base_df,
        sam_mask_stacks=sam_mask_stacks,
    )
    geometry_df = build_nucleus_line_budneck_geometry(
        nuclei_v2_df=nuclei_v2_df,
        budneck_df=budneck_df,
        pair_base_df=pair_base_df,
        distance_max=budneck_line_distance_max,
        projection_min=budneck_projection_min,
        projection_max=budneck_projection_max,
        perpendicular_tolerance_deg=budneck_perpendicular_tolerance_deg,
    )
    pair_df = _build_pair_candidate_table(
        pair_v2_df=pair_v2_df,
        cell_wall_df=cell_wall_df,
        pair_debug_v2_1_df=pair_debug_v2_1_df,
        v2_1_classification_df=v2_1_classification_df,
        adjacency_df=adjacency_df,
        geometry_df=geometry_df,
    )
    pair_debug_df, selected_mother_df = _classify_pair_rows(
        pair_df=pair_df,
        pair_require_mutual_top_k=pair_require_mutual_top_k,
        pair_ideal_distance_min=pair_ideal_distance_min,
        pair_ideal_distance_max=pair_ideal_distance_max,
        pair_max_distance=pair_max_distance,
        far_pair_extra_score_required=far_pair_extra_score_required,
    )

    pair_classification_df = pair_debug_df[
        [
            "condition",
            "frame",
            "candidate_id",
            "nearest_budneck_id",
            "nucleus_a_id",
            "nucleus_b_id",
            "cell_a_id",
            "cell_b_id",
            "distance",
            "pair_score",
            "budneck_score",
            "cells_are_adjacent",
            "cell_boundary_distance",
            "shared_boundary_length",
            "shared_boundary_strength",
            "nucleus_line_hits_budneck",
            "budneck_distance_to_nucleus_line",
            "budneck_projection_fraction",
            "budneck_between_nuclei",
            "closer_adjacent_partner_exists",
            "wrong_far_assignment_warning",
            "same_trans_component",
            "cell_wall_connectedness_score",
            "neck_constriction_score",
            "wall_continuity_score",
            "same_cell_region",
            "adjacency_supports_pair",
            "bud_lobe_detected",
            "mother_bud_area_ratio",
            "is_mutual_top2",
            "previous_v2_1_class",
            "final_class",
            "decision_reason",
            "downgrade_reason",
            "angle_difference_from_perpendicular",
            "contour_yx_json",
            "mask_bbox_y0",
            "mask_bbox_x0",
            "mask_bbox_y1",
            "mask_bbox_x1",
            "adjacency_source",
        ]
    ].copy()
    pair_classification_df = pair_classification_df.rename(columns={"nearest_budneck_id": "budneck_id"})
    pair_classification_df["source_type"] = "pair_candidate"

    early_df = _build_early_bud_rows(
        v2_1_classification_df=v2_1_classification_df,
        nuclei_v2_df=nuclei_v2_df,
        frame_meta=frame_meta,
        selected_mother_df=selected_mother_df,
        pair_debug_df=pair_debug_df,
    )

    used_nuclei: set[tuple[str, int, int]] = set()
    for row in pair_classification_df[pair_classification_df["final_class"] == "mother_bud_pair"].itertuples(index=False):
        used_nuclei.add((str(row.condition), int(row.frame), int(row.nucleus_a_id)))
        used_nuclei.add((str(row.condition), int(row.frame), int(row.nucleus_b_id)))
    for row in early_df.itertuples(index=False):
        used_nuclei.add((str(row.condition), int(row.frame), int(row.nucleus_a_id)))

    start_candidate_id = int(max(pair_classification_df["candidate_id"].max(), early_df["candidate_id"].max() if not early_df.empty else 0)) + 1
    single_df = _build_single_cell_rows(
        nuclei_v2_df=nuclei_v2_df,
        frame_meta=frame_meta,
        used_nuclei=used_nuclei,
        start_candidate_id=start_candidate_id,
    )

    classification_df = pd.concat([pair_classification_df, early_df, single_df], ignore_index=True, sort=False)
    classification_df["budneck_id"] = classification_df["budneck_id"].astype("float64")
    return adjacency_df, geometry_df, pair_debug_df, classification_df


def build_rules_v2_3_summary(
    v2_1_classification_df: pd.DataFrame,
    adjacency_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    pair_debug_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the v2.3 rule-refinement outcomes."""

    counts = classification_df["final_class"].value_counts()
    changed_mask = classification_df["previous_v2_1_class"].notna() & (classification_df["previous_v2_1_class"] != classification_df["final_class"])
    rows = [
        {"metric": "mother_bud_pair", "value": int(counts.get("mother_bud_pair", 0))},
        {"metric": "early_bud_pair", "value": int(counts.get("early_bud_pair", 0))},
        {"metric": "single_cell", "value": int(counts.get("single_cell", 0))},
        {"metric": "uncertain_pair", "value": int(counts.get("uncertain_pair", 0))},
        {"metric": "rejected_nonadjacent_pair", "value": int(counts.get("rejected_nonadjacent_pair", 0))},
        {
            "metric": "pairs_rejected_because_nonadjacent",
            "value": int(
                classification_df[
                    (classification_df["final_class"] == "rejected_nonadjacent_pair")
                    & (~classification_df["cells_are_adjacent"].astype(bool))
                ].shape[0]
            ),
        },
        {
            "metric": "pairs_downgraded_because_line_missed_budneck",
            "value": int(
                classification_df[
                    classification_df["downgrade_reason"].fillna("").str.contains("line_misses_budneck")
                    & classification_df["final_class"].isin(["uncertain_pair", "rejected_nonadjacent_pair"])
                ].shape[0]
            ),
        },
        {
            "metric": "wrong_far_assignment_warning_remaining",
            "value": int(classification_df["wrong_far_assignment_warning"].fillna(False).astype(bool).sum()),
        },
        {
            "metric": "used_existing_sam_masks_for_adjacency",
            "value": int((adjacency_df["adjacency_source"] == "existing_sam_masks").sum()),
        },
        {
            "metric": "used_trans_morphology_fallback_for_adjacency",
            "value": int((adjacency_df["adjacency_source"] == "trans_morphology_fallback").sum()),
        },
        {
            "metric": "v2_1_classes_changed_in_v2_3",
            "value": int(changed_mask.sum()),
        },
        {
            "metric": "line_neck_supports_pair_false",
            "value": int((~geometry_df["line_neck_supports_pair"].astype(bool)).sum()),
        },
        {
            "metric": "wrong_far_assignments_fixed_from_v2_1",
            "value": int(
                classification_df[
                    classification_df["previous_v2_1_class"].isin(["mother_bud_pair", "uncertain_pair"])
                    & classification_df["wrong_far_assignment_warning"].fillna(False).astype(bool)
                    & (classification_df["final_class"] != "mother_bud_pair")
                ].shape[0]
            ),
        },
    ]

    if not v2_1_classification_df.empty:
        v2_1_counts = v2_1_classification_df["final_class"].value_counts()
        rows.extend(
            [
                {"metric": "previous_v2_1_mother_bud_pair", "value": int(v2_1_counts.get("mother_bud_pair", 0))},
                {"metric": "previous_v2_1_early_bud_pair", "value": int(v2_1_counts.get("early_bud_pair", 0))},
            ]
        )
    return pd.DataFrame(rows)
