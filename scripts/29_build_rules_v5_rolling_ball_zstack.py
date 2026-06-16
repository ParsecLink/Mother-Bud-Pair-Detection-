#!/usr/bin/env python
"""Build rules_v5 with rolling-ball Z-stack preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    BUDNECK_LINE_DISTANCE_MAX,
    BUDNECK_PERPENDICULAR_TOLERANCE_DEG,
    BUDNECK_PROJECTION_MAX,
    BUDNECK_PROJECTION_MIN,
    FAR_PAIR_EXTRA_SCORE_REQUIRED,
    MERGED_RGB_ENHANCED_DIR,
    PAIR_DISTANCE_MAX,
    PAIR_DISTANCE_MIN,
    PAIR_IDEAL_DISTANCE_MAX,
    PAIR_IDEAL_DISTANCE_MIN,
    PAIR_MAX_DISTANCE,
    PAIR_REQUIRE_MUTUAL_TOP_K,
    PROJECTED_DIR,
    ROLLING_BALL_RADII_TO_TEST,
    ROLLING_BALL_THRESHOLD_METHODS,
    RULES_V3_ZSTACK_TABLES_DIR,
    RULES_V4_RANDOM_WALKER_TABLES_DIR,
    RULES_V5_ROLLING_BALL_DIAGNOSTICS_DIR,
    RULES_V5_ROLLING_BALL_DIR,
    RULES_V5_ROLLING_BALL_FIGURES_DIR,
    RULES_V5_ROLLING_BALL_OVERLAYS_DIR,
    RULES_V5_ROLLING_BALL_SAMPLED_REVIEW_DIR,
    RULES_V5_ROLLING_BALL_TABLES_DIR,
    ZSTACK_MAX_REPRESENTATIVE_FIELDS_PER_CONDITION,
    ZSTACK_PLANES_PER_FIELD,
)
from my_sam_pipeline.clean_biological_view import (  # noqa: E402
    build_clean_biological_classification,
    build_clean_biological_summary,
    build_clean_review_manifest,
    build_manual_review_csv,
)
from my_sam_pipeline.io_utils import collect_condition_triplets, ensure_dir, read_tiff_stack  # noqa: E402
from my_sam_pipeline.rolling_ball_zstack import (  # noqa: E402
    ROLLING_BALL_CONFIGS,
    build_rolling_ball_best_z_objects,
    build_rolling_ball_candidates,
    build_rolling_ball_detection_summary,
    build_rolling_ball_threshold_mask,
    build_v3_v4_v5_comparison,
    build_v5_rule_summary,
    channel_radius,
    evaluate_rolling_ball_plane,
    rolling_ball_correct_plane,
    save_review_patch,
    save_rolling_ball_diagnostic_figure,
    save_rolling_ball_zstack_overlay,
)
from my_sam_pipeline.rules_v2_1 import classify_rules_v2_1  # noqa: E402
from my_sam_pipeline.rules_v2_3 import build_rules_v2_3  # noqa: E402
from my_sam_pipeline.rules_v3_zstack import (  # noqa: E402
    adapt_zstack_budnecks,
    adapt_zstack_nuclei,
    add_zstack_metadata_to_classification,
    build_zstack_pair_candidates,
    save_rules_v3_zstack_overlay,
)
from my_sam_pipeline.zstack_detection import choose_representative_fields, reshape_stack_to_fields, save_candidate_feature_figure, save_object_count_figure  # noqa: E402


IMAGE_DIR = PROJECT_ROOT.parent / "image"


def iter_conditions(projected_dir: Path) -> list[str]:
    """Find conditions with projected Trans stacks."""

    return sorted(path.name[: -len("_Trans.tif")] for path in projected_dir.glob("*_Trans.tif"))


def load_projected_stacks() -> tuple[dict[str, object], dict[str, object]]:
    """Load projected Trans and GFP stacks for biological rule functions."""

    trans_stacks: dict[str, object] = {}
    gfp_stacks: dict[str, object] = {}
    for condition in iter_conditions(PROJECTED_DIR):
        trans_stacks[condition] = read_tiff_stack(PROJECTED_DIR / f"{condition}_Trans.tif")
        gfp_stacks[condition] = read_tiff_stack(PROJECTED_DIR / f"{condition}_GFP_projected.tif")
    return trans_stacks, gfp_stacks


def load_field_stacks(triplet: dict[str, object]) -> dict[str, np.ndarray]:
    """Load raw GFP/mCherry stacks as [field, z, y, x]."""

    return {
        "GFP": reshape_stack_to_fields(read_tiff_stack(Path(triplet["gfp_path"])), ZSTACK_PLANES_PER_FIELD),
        "mCherry": reshape_stack_to_fields(read_tiff_stack(Path(triplet["mcherry_path"])), ZSTACK_PLANES_PER_FIELD),
    }


def background_png(condition: str, frame: int) -> Path:
    """Return the already-created enhanced merged image path for a condition/frame."""

    return MERGED_RGB_ENHANCED_DIR / f"{condition}_frame_{frame:03d}.png"


def diagnostic_keys_for_triplets(triplets: list[dict[str, object]], max_conditions: int = 4) -> set[tuple[str, int, int, str]]:
    """Select a compact set of planes for radius diagnostics."""

    keys: set[tuple[str, int, int, str]] = set()
    for triplet in triplets[:max_conditions]:
        condition = str(triplet["condition"])
        for channel in ("GFP", "mCherry"):
            keys.add((condition, 0, 0, channel))
            keys.add((condition, 0, min(5, ZSTACK_PLANES_PER_FIELD - 1), channel))
    return keys


def add_radius_diagnostic_rows(
    *,
    rows: list[dict[str, object]],
    condition: str,
    field_or_frame: int,
    z_plane: int,
    channel: str,
    raw_plane: np.ndarray,
) -> None:
    """Append threshold diagnostics for non-default rolling-ball radii on sampled planes."""

    default_radius = channel_radius(channel)
    config = ROLLING_BALL_CONFIGS[channel]
    for radius in ROLLING_BALL_RADII_TO_TEST:
        if int(radius) == int(default_radius):
            continue
        corrected = rolling_ball_correct_plane(raw_plane, radius=int(radius), gaussian_sigma=config.gaussian_sigma)
        for method in ROLLING_BALL_THRESHOLD_METHODS:
            mask, threshold_value = build_rolling_ball_threshold_mask(corrected["corrected_norm"], method, config)
            from my_sam_pipeline.zstack_detection import quick_plane_quality  # Local import keeps script header compact.

            quality_score, object_count = quick_plane_quality(mask, channel, config)
            rows.append(
                {
                    "condition": condition,
                    "field": int(field_or_frame),
                    "field_or_frame": int(field_or_frame),
                    "z_plane": int(z_plane),
                    "channel": channel,
                    "rolling_ball_radius": int(radius),
                    "threshold_method": method,
                    "threshold_value": float(threshold_value),
                    "foreground_pixel_count": int(mask.sum()),
                    "foreground_fraction": float(mask.mean()),
                    "object_count": int(object_count),
                    "quality_score": float(quality_score),
                    "is_selected": False,
                    "diagnostic_only": True,
                    "background_method": str(corrected["background_method"]),
                }
            )


def generate_detection_overlays(
    *,
    triplets: list[dict[str, object]],
    best_objects_df: pd.DataFrame,
) -> int:
    """Save rolling-ball all-Z overlays for representative fields."""

    overlay_dir = RULES_V5_ROLLING_BALL_OVERLAYS_DIR / "detection"
    representative_fields = choose_representative_fields(best_objects_df, ZSTACK_MAX_REPRESENTATIVE_FIELDS_PER_CONDITION)
    triplet_lookup = {str(triplet["condition"]): triplet for triplet in triplets}
    overlay_count = 0
    for condition, field_or_frame in representative_fields:
        stacks = load_field_stacks(triplet_lookup[condition])
        safe_condition = condition.replace("/", "_")
        for channel in ("GFP", "mCherry"):
            planes_data: list[dict[str, object]] = []
            for z_plane in range(ZSTACK_PLANES_PER_FIELD):
                raw_plane = stacks[channel][field_or_frame, z_plane]
                diagnostics, _, labels, corrected, selected_method = evaluate_rolling_ball_plane(
                    condition=condition,
                    field_or_frame=field_or_frame,
                    z_plane=z_plane,
                    channel=channel,
                    raw_plane=raw_plane,
                    radius=channel_radius(channel),
                )
                selected = next(row for row in diagnostics if row["is_selected"])
                mask, _ = build_rolling_ball_threshold_mask(
                    corrected["corrected_norm"],
                    str(selected["threshold_method"]),
                    ROLLING_BALL_CONFIGS[channel],
                )
                plane_objects_df = best_objects_df[
                    (best_objects_df["condition"] == condition)
                    & (best_objects_df["field_or_frame"] == field_or_frame)
                    & (best_objects_df["channel"] == channel)
                    & (best_objects_df["z_plane"] == z_plane)
                ].copy()
                planes_data.append(
                    {
                        "raw_norm": corrected["raw_norm"],
                        "background_norm": corrected["background_norm"],
                        "corrected_norm": corrected["corrected_norm"],
                        "binary_mask": mask,
                        "labels": labels,
                        "threshold_method": selected_method,
                        "objects": plane_objects_df.to_dict("records"),
                    }
                )
            save_rolling_ball_zstack_overlay(
                output_path=overlay_dir / f"{safe_condition}_field_{field_or_frame:03d}_{channel}.png",
                condition=condition,
                field_or_frame=field_or_frame,
                channel=channel,
                planes_data=planes_data,
            )
            overlay_count += 1
    return overlay_count


def generate_biological_overlays(
    *,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    max_frames: int = 60,
) -> int:
    """Save frame-level biological-rule overlays without boxes."""

    overlay_dir = RULES_V5_ROLLING_BALL_OVERLAYS_DIR / "biological_rules"
    interesting = classification_df[classification_df["final_class"] != "single_cell"].copy()
    if interesting.empty:
        interesting = classification_df.head(max_frames).copy()
    frame_keys = (
        interesting[["condition", "frame"]]
        .drop_duplicates()
        .sort_values(["condition", "frame"])
        .head(max_frames)
        .itertuples(index=False)
    )
    overlay_count = 0
    for row in frame_keys:
        condition = str(row.condition)
        frame = int(row.frame)
        bg = background_png(condition, frame)
        if not bg.exists():
            continue
        frame_nuclei = nuclei_df[(nuclei_df["condition"] == condition) & (nuclei_df["frame"] == frame)].copy()
        frame_bud = budneck_df[(budneck_df["condition"] == condition) & (budneck_df["frame"] == frame)].copy()
        frame_class = classification_df[(classification_df["condition"] == condition) & (classification_df["frame"] == frame)].copy()
        safe_condition = condition.replace("/", "_")
        save_rules_v3_zstack_overlay(
            background_png=bg,
            output_path=overlay_dir / f"{safe_condition}_frame_{frame:03d}_rules_v5.png",
            nuclei_df=frame_nuclei,
            budneck_df=frame_bud,
            classification_df=frame_class,
        )
        overlay_count += 1
    return overlay_count


def build_manual_review_table(
    *,
    nucleus_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    rng_seed: int = 55,
) -> pd.DataFrame:
    """Create the v5 manual review table requested for candidate/object review."""

    rng = np.random.default_rng(rng_seed)

    def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df.empty or len(df) <= n:
            return df.copy()
        return df.loc[sorted(rng.choice(df.index.to_numpy(), size=n, replace=False))].copy()

    rows: list[dict[str, object]] = []
    for row in sample(nucleus_df, 100).itertuples(index=False):
        rows.append(
            {
                "condition": str(row.condition),
                "field": int(row.frame),
                "z_plane": int(row.z_plane),
                "candidate_id": int(row.nucleus_id),
                "object_type": "nucleus_candidate",
                "final_class": "",
                "nucleus_a_id": int(row.nucleus_id),
                "nucleus_b_id": np.nan,
                "budneck_id": np.nan,
                "best_z_plane": int(row.best_z_plane),
                "area": float(row.area),
                "signal_to_background": float(row.signal_to_background),
                "aspect_ratio": float(row.aspect_ratio),
                "cells_are_adjacent": np.nan,
                "nucleus_line_hits_budneck": np.nan,
                "budneck_between_nuclei": np.nan,
                "manual_class": "",
                "manual_notes": "",
            }
        )
    for row in sample(budneck_df, 100).itertuples(index=False):
        rows.append(
            {
                "condition": str(row.condition),
                "field": int(row.frame),
                "z_plane": int(row.z_plane),
                "candidate_id": int(row.budneck_id),
                "object_type": "budneck_candidate",
                "final_class": "",
                "nucleus_a_id": np.nan,
                "nucleus_b_id": np.nan,
                "budneck_id": int(row.budneck_id),
                "best_z_plane": int(row.best_z_plane),
                "area": float(row.area),
                "signal_to_background": float(row.signal_to_background),
                "aspect_ratio": float(row.aspect_ratio),
                "cells_are_adjacent": np.nan,
                "nucleus_line_hits_budneck": np.nan,
                "budneck_between_nuclei": np.nan,
                "manual_class": "",
                "manual_notes": "",
            }
        )
    class_samples = [
        clean_df[clean_df["clean_class"] == "mother_bud_pair"],
        clean_df[clean_df["clean_class"] == "early_bud_pair"],
        sample(clean_df[clean_df["clean_class"] == "rejected_pair_candidate"], 50),
        sample(clean_df[clean_df["clean_class"] == "true_uncertain_review"], 50),
    ]
    for class_df in class_samples:
        for row in class_df.itertuples(index=False):
            rows.append(
                {
                    "condition": str(row.condition),
                    "field": int(row.frame),
                    "z_plane": np.nan,
                    "candidate_id": int(row.candidate_id),
                    "object_type": "biological_rule_candidate",
                    "final_class": str(row.clean_class),
                    "nucleus_a_id": row.nucleus_a_id,
                    "nucleus_b_id": row.nucleus_b_id,
                    "budneck_id": row.budneck_id,
                    "best_z_plane": np.nan,
                    "area": np.nan,
                    "signal_to_background": np.nan,
                    "aspect_ratio": np.nan,
                    "cells_are_adjacent": bool(row.cells_are_adjacent),
                    "nucleus_line_hits_budneck": bool(row.nucleus_line_hits_budneck),
                    "budneck_between_nuclei": bool(row.budneck_between_nuclei),
                    "manual_class": "",
                    "manual_notes": "",
                }
            )
    return pd.DataFrame(rows)


def save_sampled_review_patches(review_df: pd.DataFrame, nucleus_df: pd.DataFrame, budneck_df: pd.DataFrame) -> int:
    """Save simple crop patches for manual review rows."""

    nucleus_lookup = {
        (str(row.condition), int(row.frame), int(row.nucleus_id)): (float(row.centroid_y), float(row.centroid_x))
        for row in nucleus_df.itertuples(index=False)
    }
    bud_lookup = {
        (str(row.condition), int(row.frame), int(row.budneck_id)): (float(row.centroid_y), float(row.centroid_x))
        for row in budneck_df.itertuples(index=False)
    }
    count = 0
    for idx, row in enumerate(review_df.head(350).itertuples(index=False), start=1):
        condition = str(row.condition)
        frame = int(row.field)
        centers: list[tuple[float, float]] = []
        if pd.notna(row.nucleus_a_id):
            point = nucleus_lookup.get((condition, frame, int(row.nucleus_a_id)))
            if point is not None:
                centers.append(point)
        if pd.notna(row.nucleus_b_id):
            point = nucleus_lookup.get((condition, frame, int(row.nucleus_b_id)))
            if point is not None:
                centers.append(point)
        if pd.notna(row.budneck_id):
            point = bud_lookup.get((condition, frame, int(row.budneck_id)))
            if point is not None:
                centers.append(point)
        if not centers:
            continue
        center_y = float(np.mean([point[0] for point in centers]))
        center_x = float(np.mean([point[1] for point in centers]))
        safe_condition = condition.replace("/", "_")
        save_review_patch(
            output_path=RULES_V5_ROLLING_BALL_SAMPLED_REVIEW_DIR / f"{idx:04d}_{safe_condition}_frame_{frame:03d}_{row.object_type}_{row.candidate_id}.png",
            background_png=background_png(condition, frame),
            center_y=center_y,
            center_x=center_x,
        )
        count += 1
    return count


def main() -> None:
    for directory in [
        RULES_V5_ROLLING_BALL_DIR,
        RULES_V5_ROLLING_BALL_TABLES_DIR,
        RULES_V5_ROLLING_BALL_FIGURES_DIR,
        RULES_V5_ROLLING_BALL_OVERLAYS_DIR,
        RULES_V5_ROLLING_BALL_OVERLAYS_DIR / "detection",
        RULES_V5_ROLLING_BALL_OVERLAYS_DIR / "biological_rules",
        RULES_V5_ROLLING_BALL_SAMPLED_REVIEW_DIR,
        RULES_V5_ROLLING_BALL_DIAGNOSTICS_DIR,
    ]:
        ensure_dir(directory)

    triplets = collect_condition_triplets(IMAGE_DIR, projection_group_size=ZSTACK_PLANES_PER_FIELD)
    diagnostic_keys = diagnostic_keys_for_triplets(triplets)
    print(f"Found {len(triplets)} valid condition triplets", flush=True)
    print(f"Rolling-ball radii tested in diagnostics: {ROLLING_BALL_RADII_TO_TEST}", flush=True)

    diagnostic_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    diagnostic_figure_count = 0

    for triplet in triplets:
        condition = str(triplet["condition"])
        stacks = load_field_stacks(triplet)
        field_count = int(stacks["GFP"].shape[0])
        print(f"[{condition}] rolling-ball detection: {field_count} fields x {ZSTACK_PLANES_PER_FIELD} Z planes", flush=True)
        for field_or_frame in range(field_count):
            for channel in ("GFP", "mCherry"):
                for z_plane in range(ZSTACK_PLANES_PER_FIELD):
                    raw_plane = stacks[channel][field_or_frame, z_plane]
                    radius = channel_radius(channel)
                    plane_diagnostics, plane_objects, _, _, _ = evaluate_rolling_ball_plane(
                        condition=condition,
                        field_or_frame=field_or_frame,
                        z_plane=z_plane,
                        channel=channel,
                        raw_plane=raw_plane,
                        radius=radius,
                    )
                    for row in plane_diagnostics:
                        row["diagnostic_only"] = False
                    diagnostic_rows.extend(plane_diagnostics)
                    object_rows.extend(plane_objects)
                    if (condition, field_or_frame, z_plane, channel) in diagnostic_keys:
                        add_radius_diagnostic_rows(
                            rows=diagnostic_rows,
                            condition=condition,
                            field_or_frame=field_or_frame,
                            z_plane=z_plane,
                            channel=channel,
                            raw_plane=raw_plane,
                        )
                        safe_condition = condition.replace("/", "_")
                        save_rolling_ball_diagnostic_figure(
                            output_path=RULES_V5_ROLLING_BALL_DIAGNOSTICS_DIR / f"{safe_condition}_field_{field_or_frame:03d}_z{z_plane:02d}_{channel}.png",
                            raw_plane=raw_plane,
                            channel=channel,
                            condition=condition,
                            field_or_frame=field_or_frame,
                            z_plane=z_plane,
                        )
                        diagnostic_figure_count += 1

    diagnostics_df = pd.DataFrame(diagnostic_rows)
    objects_df = pd.DataFrame(object_rows)
    best_objects_df = build_rolling_ball_best_z_objects(objects_df)
    nucleus_raw_df, budneck_raw_df = build_rolling_ball_candidates(best_objects_df)
    detection_summary_df = build_rolling_ball_detection_summary(
        diagnostics_df=diagnostics_df[~diagnostics_df["diagnostic_only"].fillna(False).astype(bool)].copy(),
        objects_df=objects_df,
        best_objects_df=best_objects_df,
        nucleus_df=nucleus_raw_df,
        budneck_df=budneck_raw_df,
    )

    diagnostics_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_threshold_diagnostics.csv", index=False)
    objects_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_fluorescence_objects.csv", index=False)
    best_objects_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_best_z_objects.csv", index=False)
    nucleus_raw_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_nucleus_candidates.csv", index=False)
    budneck_raw_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_budneck_candidates.csv", index=False)
    detection_summary_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_detection_summary.csv", index=False)
    save_object_count_figure(RULES_V5_ROLLING_BALL_FIGURES_DIR, objects_df)
    save_candidate_feature_figure(RULES_V5_ROLLING_BALL_FIGURES_DIR, nucleus_raw_df, budneck_raw_df)

    nuclei_v5 = adapt_zstack_nuclei(nucleus_raw_df)
    budneck_v5 = adapt_zstack_budnecks(budneck_raw_df)
    pair_v5 = build_zstack_pair_candidates(
        nucleus_df=nuclei_v5,
        budneck_df=budneck_v5,
        pair_distance_min=PAIR_DISTANCE_MIN,
        pair_distance_max=PAIR_DISTANCE_MAX,
    )
    nuclei_v5.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_adapted_nucleus_candidates.csv", index=False)
    budneck_v5.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_adapted_budneck_candidates.csv", index=False)
    pair_v5.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_pair_candidates.csv", index=False)

    trans_stacks, gfp_stacks = load_projected_stacks()
    cell_wall_df, debug_v2_1_df, conflict_df, missing_df, class_v2_1_df = classify_rules_v2_1(
        trans_stacks=trans_stacks,
        gfp_stacks=gfp_stacks,
        nuclei_v1_df=nuclei_v5,
        nuclei_v2_df=nuclei_v5,
        budneck_df=budneck_v5,
        pair_v2_df=pair_v5,
        pair_require_mutual_top_k=PAIR_REQUIRE_MUTUAL_TOP_K,
        pair_ideal_distance_min=PAIR_IDEAL_DISTANCE_MIN,
        pair_ideal_distance_max=PAIR_IDEAL_DISTANCE_MAX,
        pair_max_distance=PAIR_MAX_DISTANCE,
        far_pair_extra_score_required=FAR_PAIR_EXTRA_SCORE_REQUIRED,
    )
    adjacency_df, geometry_df, pair_debug_df, classification_df = build_rules_v2_3(
        trans_stacks=trans_stacks,
        nuclei_v2_df=nuclei_v5,
        budneck_df=budneck_v5,
        pair_v2_df=pair_v5,
        cell_wall_df=cell_wall_df,
        pair_debug_v2_1_df=debug_v2_1_df,
        v2_1_classification_df=class_v2_1_df,
        sam_mask_stacks={},
        pair_require_mutual_top_k=PAIR_REQUIRE_MUTUAL_TOP_K,
        pair_ideal_distance_min=PAIR_IDEAL_DISTANCE_MIN,
        pair_ideal_distance_max=PAIR_IDEAL_DISTANCE_MAX,
        pair_max_distance=PAIR_MAX_DISTANCE,
        far_pair_extra_score_required=FAR_PAIR_EXTRA_SCORE_REQUIRED,
        budneck_line_distance_max=BUDNECK_LINE_DISTANCE_MAX,
        budneck_projection_min=BUDNECK_PROJECTION_MIN,
        budneck_projection_max=BUDNECK_PROJECTION_MAX,
        budneck_perpendicular_tolerance_deg=BUDNECK_PERPENDICULAR_TOLERANCE_DEG,
    )
    classification_df = add_zstack_metadata_to_classification(classification_df, nuclei_v5, budneck_v5)
    classification_df["preprocessing_source"] = "rolling_ball_zstack"
    clean_df = build_clean_biological_classification(classification_df)
    clean_summary_df = build_clean_biological_summary(clean_df)
    clean_manifest_df = build_clean_review_manifest(clean_df)
    clean_review_df = build_manual_review_csv(clean_manifest_df)

    v3_classification_df = pd.read_csv(RULES_V3_ZSTACK_TABLES_DIR / "rules_v3_zstack_classification.csv") if (RULES_V3_ZSTACK_TABLES_DIR / "rules_v3_zstack_classification.csv").exists() else None
    v4_clean_df = pd.read_csv(RULES_V4_RANDOM_WALKER_TABLES_DIR / "clean_biological_classification_random_walker.csv") if (RULES_V4_RANDOM_WALKER_TABLES_DIR / "clean_biological_classification_random_walker.csv").exists() else None
    v3_nucleus_count = len(pd.read_csv(RULES_V3_ZSTACK_TABLES_DIR / "zstack_adapted_nucleus_candidates.csv")) if (RULES_V3_ZSTACK_TABLES_DIR / "zstack_adapted_nucleus_candidates.csv").exists() else 0
    v3_budneck_count = len(pd.read_csv(RULES_V3_ZSTACK_TABLES_DIR / "zstack_adapted_budneck_candidates.csv")) if (RULES_V3_ZSTACK_TABLES_DIR / "zstack_adapted_budneck_candidates.csv").exists() else 0
    v3_pair_count = len(pd.read_csv(RULES_V3_ZSTACK_TABLES_DIR / "zstack_pair_candidates_v3.csv")) if (RULES_V3_ZSTACK_TABLES_DIR / "zstack_pair_candidates_v3.csv").exists() else 0
    v4_nucleus_count = len(pd.read_csv(RULES_V4_RANDOM_WALKER_TABLES_DIR / "random_walker_adapted_nucleus_candidates.csv")) if (RULES_V4_RANDOM_WALKER_TABLES_DIR / "random_walker_adapted_nucleus_candidates.csv").exists() else 0
    v4_budneck_count = len(pd.read_csv(RULES_V4_RANDOM_WALKER_TABLES_DIR / "random_walker_adapted_budneck_candidates.csv")) if (RULES_V4_RANDOM_WALKER_TABLES_DIR / "random_walker_adapted_budneck_candidates.csv").exists() else 0
    v4_pair_count = len(pd.read_csv(RULES_V4_RANDOM_WALKER_TABLES_DIR / "random_walker_pair_candidates.csv")) if (RULES_V4_RANDOM_WALKER_TABLES_DIR / "random_walker_pair_candidates.csv").exists() else 0
    comparison_df = build_v3_v4_v5_comparison(
        v5_detection_summary=detection_summary_df,
        v5_pair_df=pair_v5,
        v5_clean_df=clean_df,
        v5_classification_df=classification_df,
        v3_classification_df=v3_classification_df,
        v4_clean_df=v4_clean_df,
        v3_nucleus_count=v3_nucleus_count,
        v3_budneck_count=v3_budneck_count,
        v4_nucleus_count=v4_nucleus_count,
        v4_budneck_count=v4_budneck_count,
        v3_pair_count=v3_pair_count,
        v4_pair_count=v4_pair_count,
    )
    summary_df = build_v5_rule_summary(
        pair_df=pair_v5,
        classification_df=classification_df,
        clean_df=clean_df,
        detection_summary_df=detection_summary_df,
        comparison_df=comparison_df,
    )

    cell_wall_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_cell_wall_features.csv", index=False)
    debug_v2_1_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_pair_assignment_debug_v2_1.csv", index=False)
    conflict_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_pair_conflicts_resolved.csv", index=False)
    missing_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_missing_nucleus_diagnostics.csv", index=False)
    class_v2_1_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rolling_ball_rules_v2_1_classification.csv", index=False)
    adjacency_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "cell_adjacency_features.csv", index=False)
    geometry_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "nucleus_line_budneck_geometry.csv", index=False)
    pair_debug_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "pair_assignment_debug_v5_rolling_ball.csv", index=False)
    classification_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rules_v5_rolling_ball_classification.csv", index=False)
    clean_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "clean_biological_classification_rolling_ball.csv", index=False)
    clean_summary_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "clean_biological_summary_rolling_ball.csv", index=False)
    clean_manifest_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "clean_biological_review_manifest_rolling_ball.csv", index=False)
    clean_review_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "manual_review_clean_biological_rolling_ball.csv", index=False)
    comparison_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "v3_v4_v5_comparison.csv", index=False)
    summary_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "rules_v5_rolling_ball_summary.csv", index=False)

    detection_overlay_count = generate_detection_overlays(triplets=triplets, best_objects_df=best_objects_df)
    biological_overlay_count = generate_biological_overlays(nuclei_df=nuclei_v5, budneck_df=budneck_v5, classification_df=classification_df)
    manual_review_df = build_manual_review_table(nucleus_df=nuclei_v5, budneck_df=budneck_v5, clean_df=clean_df)
    manual_review_df.to_csv(RULES_V5_ROLLING_BALL_TABLES_DIR / "manual_review_v5_rolling_ball.csv", index=False)
    patch_count = save_sampled_review_patches(manual_review_df, nuclei_v5, budneck_v5)

    print("Rolling-ball v5 complete", flush=True)
    print(f"  diagnostic figures: {diagnostic_figure_count}", flush=True)
    print(f"  detected GFP objects: {int((objects_df['channel'] == 'GFP').sum()) if not objects_df.empty else 0}", flush=True)
    print(f"  detected mCherry objects: {int((objects_df['channel'] == 'mCherry').sum()) if not objects_df.empty else 0}", flush=True)
    print(f"  nucleus candidates: {len(nuclei_v5)}", flush=True)
    print(f"  bud-neck candidates: {len(budneck_v5)}", flush=True)
    print(f"  pair candidates: {len(pair_v5)}", flush=True)
    print("Clean biological class counts:", flush=True)
    print(clean_df["clean_class"].value_counts().to_string() if not clean_df.empty else "none", flush=True)
    print(f"  detection overlays: {detection_overlay_count}", flush=True)
    print(f"  biological overlays: {biological_overlay_count}", flush=True)
    print(f"  review patches: {patch_count}", flush=True)
    print(f"Tables saved to: {RULES_V5_ROLLING_BALL_TABLES_DIR}", flush=True)


if __name__ == "__main__":
    main()
