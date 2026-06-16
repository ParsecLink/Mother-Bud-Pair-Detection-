#!/usr/bin/env python
"""Run biological matching using all rolling-ball best-Z objects.

This is an intentionally permissive trial. It skips the candidate filtering
step and converts every best-Z GFP object into a possible nucleus and every
best-Z mCherry object into a possible bud-neck object.
"""

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
)
from my_sam_pipeline.clean_biological_view import (  # noqa: E402
    build_clean_biological_classification,
    build_clean_biological_summary,
    build_clean_review_manifest,
    build_manual_review_csv,
)
from my_sam_pipeline.io_utils import ensure_dir, read_tiff_stack  # noqa: E402
from my_sam_pipeline.rules_v2_1 import classify_rules_v2_1  # noqa: E402
from my_sam_pipeline.rules_v2_3 import build_rules_v2_3  # noqa: E402
from my_sam_pipeline.rules_v3_zstack import (  # noqa: E402
    adapt_zstack_budnecks,
    adapt_zstack_nuclei,
    add_zstack_metadata_to_classification,
    build_zstack_pair_candidates,
    save_rules_v3_zstack_overlay,
)


TRIAL_DIR = PROJECT_ROOT / "rule_discovery" / "rules_v5_all_objects_matching"
TABLES_DIR = TRIAL_DIR / "tables"
OVERLAYS_DIR = TRIAL_DIR / "overlays"
FULL_OBJECT_OVERLAYS_DIR = OVERLAYS_DIR / "full_all_detected_objects"
FULL_RULE_OVERLAYS_DIR = OVERLAYS_DIR / "full_rule_results"
SOURCE_TABLES_DIR = PROJECT_ROOT / "rule_discovery" / "rules_v5_rolling_ball_zstack" / "tables"


def iter_conditions(projected_dir: Path) -> list[str]:
    """Find conditions with projected Trans stacks."""

    return sorted(path.name[: -len("_Trans.tif")] for path in projected_dir.glob("*_Trans.tif"))


def load_projected_stacks() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load projected Trans and GFP stacks needed by the existing rule code."""

    trans_stacks: dict[str, np.ndarray] = {}
    gfp_stacks: dict[str, np.ndarray] = {}
    for condition in iter_conditions(PROJECTED_DIR):
        trans_stacks[condition] = read_tiff_stack(PROJECTED_DIR / f"{condition}_Trans.tif")
        gfp_stacks[condition] = read_tiff_stack(PROJECTED_DIR / f"{condition}_GFP_projected.tif")
    return trans_stacks, gfp_stacks


def prepare_all_best_z_inputs(best_objects_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create unfiltered GFP and mCherry input tables for the z-stack adapters."""

    best = best_objects_df[best_objects_df["is_best_z"].astype(bool)].copy()
    gfp = best[best["channel"] == "GFP"].copy()
    mcherry = best[best["channel"] == "mCherry"].copy()
    for df in (gfp, mcherry):
        df["candidate_score"] = df["object_quality_score"].astype(float)
        # Mark as high confidence only to let every object enter downstream matching.
        # The source_mode column records that these were not filtered biology candidates.
        df["candidate_class"] = "high_confidence"
        df["source_mode"] = "all_best_z_detected_objects_no_candidate_filter"
    return gfp, mcherry


def add_source_mode(df: pd.DataFrame, source_mode: str) -> pd.DataFrame:
    """Attach source metadata for auditability."""

    out = df.copy()
    out["source_mode"] = source_mode
    return out


def build_summary(
    *,
    all_gfp_df: pd.DataFrame,
    all_mcherry_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    object_overlay_count: int,
    rule_overlay_count: int,
) -> pd.DataFrame:
    """Summarize the all-object matching trial."""

    raw_counts = classification_df["final_class"].value_counts() if not classification_df.empty else pd.Series(dtype=int)
    clean_counts = clean_df["clean_class"].value_counts() if not clean_df.empty else pd.Series(dtype=int)
    rows = [
        {"metric": "source", "value": "rules_v5_rolling_ball_zstack rolling_ball_best_z_objects.csv", "detail": ""},
        {"metric": "filtering_skipped", "value": 1, "detail": "all best-Z GFP/mCherry detected objects were used"},
        {"metric": "all_best_z_gfp_objects_as_nuclei", "value": int(len(all_gfp_df)), "detail": ""},
        {"metric": "all_best_z_mcherry_objects_as_budnecks", "value": int(len(all_mcherry_df)), "detail": ""},
        {"metric": "pair_candidates", "value": int(len(pair_df)), "detail": ""},
        {"metric": "raw_mother_bud_pair", "value": int(raw_counts.get("mother_bud_pair", 0)), "detail": "v2.3/v3-style raw rule class"},
        {"metric": "raw_early_bud_pair", "value": int(raw_counts.get("early_bud_pair", 0)), "detail": "v2.3/v3-style raw rule class"},
        {"metric": "raw_single_cell", "value": int(raw_counts.get("single_cell", 0)), "detail": "v2.3/v3-style raw rule class"},
        {"metric": "raw_uncertain_pair", "value": int(raw_counts.get("uncertain_pair", 0)), "detail": "v2.3/v3-style raw rule class"},
        {"metric": "raw_rejected_nonadjacent_pair", "value": int(raw_counts.get("rejected_nonadjacent_pair", 0)), "detail": "v2.3/v3-style raw rule class"},
        {"metric": "clean_mother_bud_pair", "value": int(clean_counts.get("mother_bud_pair", 0)), "detail": ""},
        {"metric": "clean_early_bud_pair", "value": int(clean_counts.get("early_bud_pair", 0)), "detail": ""},
        {"metric": "clean_single_cell", "value": int(clean_counts.get("single_cell", 0)), "detail": ""},
        {"metric": "clean_rejected_pair_candidate", "value": int(clean_counts.get("rejected_pair_candidate", 0)), "detail": ""},
        {"metric": "clean_true_uncertain_review", "value": int(clean_counts.get("true_uncertain_review", 0)), "detail": ""},
        {"metric": "full_detected_object_overlay_count", "value": int(object_overlay_count), "detail": str(FULL_OBJECT_OVERLAYS_DIR)},
        {"metric": "full_rule_result_overlay_count", "value": int(rule_overlay_count), "detail": str(FULL_RULE_OVERLAYS_DIR)},
    ]
    return pd.DataFrame(rows)


def build_filtered_vs_all_comparison(clean_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    """Compare normal v5 filtered results to the all-object trial."""

    filtered_clean_path = SOURCE_TABLES_DIR / "clean_biological_classification_rolling_ball.csv"
    filtered_nuc_path = SOURCE_TABLES_DIR / "rolling_ball_adapted_nucleus_candidates.csv"
    filtered_bud_path = SOURCE_TABLES_DIR / "rolling_ball_adapted_budneck_candidates.csv"
    filtered_pair_path = SOURCE_TABLES_DIR / "rolling_ball_pair_candidates.csv"
    filtered_clean = pd.read_csv(filtered_clean_path) if filtered_clean_path.exists() else pd.DataFrame()
    filtered_counts = filtered_clean["clean_class"].value_counts() if not filtered_clean.empty else pd.Series(dtype=int)
    all_counts = clean_df["clean_class"].value_counts() if not clean_df.empty else pd.Series(dtype=int)
    metrics = [
        ("nucleus_inputs", len(pd.read_csv(filtered_nuc_path)) if filtered_nuc_path.exists() else 0, int((SOURCE_TABLES_DIR / "rolling_ball_best_z_objects.csv").exists())),
        ("budneck_inputs", len(pd.read_csv(filtered_bud_path)) if filtered_bud_path.exists() else 0, 0),
        ("pair_candidates", len(pd.read_csv(filtered_pair_path)) if filtered_pair_path.exists() else 0, len(pair_df)),
    ]
    rows: list[dict[str, object]] = []
    best = pd.read_csv(SOURCE_TABLES_DIR / "rolling_ball_best_z_objects.csv")
    all_nuclei_count = int(((best["channel"] == "GFP") & best["is_best_z"].astype(bool)).sum())
    all_bud_count = int(((best["channel"] == "mCherry") & best["is_best_z"].astype(bool)).sum())
    rows.extend(
        [
            {"metric": "nucleus_inputs", "filtered_v5": metrics[0][1], "all_objects_trial": all_nuclei_count, "delta_all_minus_filtered": all_nuclei_count - int(metrics[0][1])},
            {"metric": "budneck_inputs", "filtered_v5": metrics[1][1], "all_objects_trial": all_bud_count, "delta_all_minus_filtered": all_bud_count - int(metrics[1][1])},
            {"metric": "pair_candidates", "filtered_v5": metrics[2][1], "all_objects_trial": int(metrics[2][2]), "delta_all_minus_filtered": int(metrics[2][2]) - int(metrics[2][1])},
        ]
    )
    for class_name in ["mother_bud_pair", "early_bud_pair", "single_cell", "rejected_pair_candidate", "true_uncertain_review"]:
        filtered_value = int(filtered_counts.get(class_name, 0))
        all_value = int(all_counts.get(class_name, 0))
        rows.append(
            {
                "metric": class_name,
                "filtered_v5": filtered_value,
                "all_objects_trial": all_value,
                "delta_all_minus_filtered": all_value - filtered_value,
            }
        )
    return pd.DataFrame(rows)


def background_png(condition: str, frame: int) -> Path:
    """Return full-frame merged RGB background."""

    return MERGED_RGB_ENHANCED_DIR / f"{condition}_frame_{frame:03d}.png"


def frame_keys_from_projected_stacks() -> list[tuple[str, int]]:
    """Return every condition/frame key with a projected Trans frame."""

    keys: list[tuple[str, int]] = []
    for condition in iter_conditions(PROJECTED_DIR):
        stack = read_tiff_stack(PROJECTED_DIR / f"{condition}_Trans.tif")
        for frame in range(int(stack.shape[0])):
            keys.append((condition, frame))
    return keys


def save_full_overlays(
    *,
    nuclei_df: pd.DataFrame,
    budneck_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    output_dir: Path,
    include_rule_lines: bool,
) -> int:
    """Save one full-frame overlay per image, optionally with rule lines."""

    ensure_dir(output_dir)
    empty_class = pd.DataFrame(columns=["final_class"])
    count = 0
    for condition, frame in frame_keys_from_projected_stacks():
        bg = background_png(condition, frame)
        if not bg.exists():
            continue
        frame_nuclei = nuclei_df[(nuclei_df["condition"] == condition) & (nuclei_df["frame"] == frame)].copy()
        frame_bud = budneck_df[(budneck_df["condition"] == condition) & (budneck_df["frame"] == frame)].copy()
        frame_class = (
            classification_df[(classification_df["condition"] == condition) & (classification_df["frame"] == frame)].copy()
            if include_rule_lines
            else empty_class.copy()
        )
        safe_condition = condition.replace("/", "_")
        suffix = "rules" if include_rule_lines else "objects"
        save_rules_v3_zstack_overlay(
            background_png=bg,
            output_path=output_dir / f"{safe_condition}_frame_{frame:03d}_{suffix}_full.png",
            nuclei_df=frame_nuclei,
            budneck_df=frame_bud,
            classification_df=frame_class,
        )
        count += 1
    return count


def main() -> None:
    for directory in [TABLES_DIR, FULL_OBJECT_OVERLAYS_DIR, FULL_RULE_OVERLAYS_DIR]:
        ensure_dir(directory)

    best_objects = pd.read_csv(SOURCE_TABLES_DIR / "rolling_ball_best_z_objects.csv")
    all_gfp_raw, all_mcherry_raw = prepare_all_best_z_inputs(best_objects)
    nuclei_all = add_source_mode(adapt_zstack_nuclei(all_gfp_raw), "all_best_z_gfp_objects_no_candidate_filter")
    budneck_all = add_source_mode(adapt_zstack_budnecks(all_mcherry_raw), "all_best_z_mcherry_objects_no_candidate_filter")
    pair_all = build_zstack_pair_candidates(
        nucleus_df=nuclei_all,
        budneck_df=budneck_all,
        pair_distance_min=PAIR_DISTANCE_MIN,
        pair_distance_max=PAIR_DISTANCE_MAX,
    )

    all_gfp_raw.to_csv(TABLES_DIR / "all_best_z_gfp_objects_used_as_nuclei.csv", index=False)
    all_mcherry_raw.to_csv(TABLES_DIR / "all_best_z_mcherry_objects_used_as_budnecks.csv", index=False)
    nuclei_all.to_csv(TABLES_DIR / "all_objects_adapted_nucleus_candidates.csv", index=False)
    budneck_all.to_csv(TABLES_DIR / "all_objects_adapted_budneck_candidates.csv", index=False)
    pair_all.to_csv(TABLES_DIR / "all_objects_pair_candidates.csv", index=False)

    trans_stacks, gfp_stacks = load_projected_stacks()
    cell_wall_df, debug_v2_1_df, conflict_df, missing_df, class_v2_1_df = classify_rules_v2_1(
        trans_stacks=trans_stacks,
        gfp_stacks=gfp_stacks,
        nuclei_v1_df=nuclei_all,
        nuclei_v2_df=nuclei_all,
        budneck_df=budneck_all,
        pair_v2_df=pair_all,
        pair_require_mutual_top_k=PAIR_REQUIRE_MUTUAL_TOP_K,
        pair_ideal_distance_min=PAIR_IDEAL_DISTANCE_MIN,
        pair_ideal_distance_max=PAIR_IDEAL_DISTANCE_MAX,
        pair_max_distance=PAIR_MAX_DISTANCE,
        far_pair_extra_score_required=FAR_PAIR_EXTRA_SCORE_REQUIRED,
    )
    adjacency_df, geometry_df, pair_debug_df, classification_df = build_rules_v2_3(
        trans_stacks=trans_stacks,
        nuclei_v2_df=nuclei_all,
        budneck_df=budneck_all,
        pair_v2_df=pair_all,
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
    classification_df = add_zstack_metadata_to_classification(classification_df, nuclei_all, budneck_all)
    classification_df["preprocessing_source"] = "rolling_ball_all_best_z_objects_no_candidate_filter"
    clean_df = build_clean_biological_classification(classification_df)

    cell_wall_df.to_csv(TABLES_DIR / "all_objects_cell_wall_features.csv", index=False)
    debug_v2_1_df.to_csv(TABLES_DIR / "all_objects_pair_assignment_debug_v2_1.csv", index=False)
    conflict_df.to_csv(TABLES_DIR / "all_objects_pair_conflicts_resolved.csv", index=False)
    missing_df.to_csv(TABLES_DIR / "all_objects_missing_nucleus_diagnostics.csv", index=False)
    class_v2_1_df.to_csv(TABLES_DIR / "all_objects_rules_v2_1_classification.csv", index=False)
    adjacency_df.to_csv(TABLES_DIR / "cell_adjacency_features.csv", index=False)
    geometry_df.to_csv(TABLES_DIR / "nucleus_line_budneck_geometry.csv", index=False)
    pair_debug_df.to_csv(TABLES_DIR / "pair_assignment_debug_all_objects.csv", index=False)
    classification_df.to_csv(TABLES_DIR / "rules_v5_all_objects_classification.csv", index=False)
    clean_df.to_csv(TABLES_DIR / "clean_biological_classification_all_objects.csv", index=False)
    build_clean_biological_summary(clean_df).to_csv(TABLES_DIR / "clean_biological_summary_all_objects.csv", index=False)
    build_clean_review_manifest(clean_df).to_csv(TABLES_DIR / "clean_biological_review_manifest_all_objects.csv", index=False)
    build_manual_review_csv(build_clean_review_manifest(clean_df)).to_csv(TABLES_DIR / "manual_review_all_objects.csv", index=False)

    object_overlay_count = save_full_overlays(
        nuclei_df=nuclei_all,
        budneck_df=budneck_all,
        classification_df=classification_df,
        output_dir=FULL_OBJECT_OVERLAYS_DIR,
        include_rule_lines=False,
    )
    rule_overlay_count = save_full_overlays(
        nuclei_df=nuclei_all,
        budneck_df=budneck_all,
        classification_df=classification_df,
        output_dir=FULL_RULE_OVERLAYS_DIR,
        include_rule_lines=True,
    )
    summary_df = build_summary(
        all_gfp_df=all_gfp_raw,
        all_mcherry_df=all_mcherry_raw,
        pair_df=pair_all,
        classification_df=classification_df,
        clean_df=clean_df,
        object_overlay_count=object_overlay_count,
        rule_overlay_count=rule_overlay_count,
    )
    comparison_df = build_filtered_vs_all_comparison(clean_df, pair_all)
    summary_df.to_csv(TABLES_DIR / "rules_v5_all_objects_summary.csv", index=False)
    comparison_df.to_csv(TABLES_DIR / "filtered_v5_vs_all_objects_comparison.csv", index=False)

    print("All-object matching trial complete", flush=True)
    print(f"  all best-Z GFP objects used as nuclei: {len(all_gfp_raw)}", flush=True)
    print(f"  all best-Z mCherry objects used as bud necks: {len(all_mcherry_raw)}", flush=True)
    print(f"  pair candidates: {len(pair_all)}", flush=True)
    print("Clean biological class counts:", flush=True)
    print(clean_df["clean_class"].value_counts().to_string(), flush=True)
    print(f"  full object overlays: {object_overlay_count}", flush=True)
    print(f"  full rule overlays: {rule_overlay_count}", flush=True)
    print(f"Tables saved to: {TABLES_DIR}", flush=True)


if __name__ == "__main__":
    main()
