"""Build rules_v2_3 with adjacency-aware, bud-centered pair refinement."""

from __future__ import annotations

import sys
from pathlib import Path

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
    PAIR_IDEAL_DISTANCE_MAX,
    PAIR_IDEAL_DISTANCE_MIN,
    PAIR_MAX_DISTANCE,
    PAIR_REQUIRE_MUTUAL_TOP_K,
    PROJECTED_DIR,
    PROJECT_ROOT as CONFIG_PROJECT_ROOT,
    RULES_V2_1_TABLES_DIR,
    RULES_V2_3_DEBUG_CASES_DIR,
    RULES_V2_3_DIR,
    RULES_V2_3_OVERLAYS_DIR,
    RULES_V2_3_SAMPLED_REVIEW_DIR,
    RULES_V2_3_TABLES_DIR,
    RULES_V2_TABLES_DIR,
    USE_EXISTING_SAM_MASKS_FOR_ADJACENCY,
)
from my_sam_pipeline.io_utils import ensure_dir, read_tiff_stack  # noqa: E402
from my_sam_pipeline.rules_v2_3 import build_rules_v2_3, build_rules_v2_3_summary  # noqa: E402


def iter_conditions(projected_dir: Path) -> list[str]:
    return sorted(path.name[: -len("_Trans.tif")] for path in projected_dir.glob("*_Trans.tif"))


def load_existing_sam_masks(results_masks_dir: Path) -> dict[str, object]:
    """Load any previously saved SAM mask stacks without running new segmentation."""

    stacks: dict[str, object] = {}
    if not USE_EXISTING_SAM_MASKS_FOR_ADJACENCY or not results_masks_dir.exists():
        return stacks
    for path in sorted(results_masks_dir.glob("*_sam_masks.tif")):
        condition = path.name[: -len("_sam_masks.tif")]
        try:
            stacks[condition] = read_tiff_stack(path)
        except Exception as exc:  # pragma: no cover - defensive read path
            print(f"Skipping unreadable SAM masks for {condition}: {exc}")
    return stacks


def main() -> None:
    ensure_dir(RULES_V2_3_DIR)
    ensure_dir(RULES_V2_3_TABLES_DIR)
    ensure_dir(RULES_V2_3_OVERLAYS_DIR)
    ensure_dir(RULES_V2_3_DEBUG_CASES_DIR)
    ensure_dir(RULES_V2_3_SAMPLED_REVIEW_DIR)

    nuclei_v2 = pd.read_csv(RULES_V2_TABLES_DIR / "nucleus_candidates_v2.csv")
    pair_v2 = pd.read_csv(RULES_V2_TABLES_DIR / "pair_candidates_v2.csv")
    budneck_v1 = pd.read_csv(CONFIG_PROJECT_ROOT / "rule_discovery" / "tables" / "budneck_candidates.csv")
    cell_wall_v2_1 = pd.read_csv(RULES_V2_1_TABLES_DIR / "cell_wall_features.csv")
    pair_debug_v2_1 = pd.read_csv(RULES_V2_1_TABLES_DIR / "pair_assignment_debug.csv")
    class_v2_1 = pd.read_csv(RULES_V2_1_TABLES_DIR / "rules_v2_1_classification.csv")

    trans_stacks: dict[str, object] = {}
    for condition in iter_conditions(PROJECTED_DIR):
        trans_stacks[condition] = read_tiff_stack(PROJECTED_DIR / f"{condition}_Trans.tif")

    sam_mask_stacks = load_existing_sam_masks(CONFIG_PROJECT_ROOT / "results" / "masks")
    if sam_mask_stacks:
        print(f"Using existing SAM mask stacks for adjacency: {len(sam_mask_stacks)} conditions")
    else:
        print("No existing SAM mask stacks found inside Model/My; using Trans morphology fallback for adjacency")

    adjacency_df, geometry_df, pair_debug_df, classification_df = build_rules_v2_3(
        trans_stacks=trans_stacks,
        nuclei_v2_df=nuclei_v2,
        budneck_df=budneck_v1,
        pair_v2_df=pair_v2,
        cell_wall_df=cell_wall_v2_1,
        pair_debug_v2_1_df=pair_debug_v2_1,
        v2_1_classification_df=class_v2_1,
        sam_mask_stacks=sam_mask_stacks,
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
    summary_df = build_rules_v2_3_summary(
        v2_1_classification_df=class_v2_1,
        adjacency_df=adjacency_df,
        geometry_df=geometry_df,
        pair_debug_df=pair_debug_df,
        classification_df=classification_df,
    )

    adjacency_df.to_csv(RULES_V2_3_TABLES_DIR / "cell_adjacency_features.csv", index=False)
    geometry_df.to_csv(RULES_V2_3_TABLES_DIR / "nucleus_line_budneck_geometry.csv", index=False)
    pair_debug_df.to_csv(RULES_V2_3_TABLES_DIR / "pair_assignment_debug_v2_3.csv", index=False)
    classification_df.to_csv(RULES_V2_3_TABLES_DIR / "rules_v2_3_classification.csv", index=False)
    summary_df.to_csv(RULES_V2_3_TABLES_DIR / "rules_v2_3_summary.csv", index=False)

    counts = classification_df["final_class"].value_counts()
    print(f"mother_bud_pair: {int(counts.get('mother_bud_pair', 0))}")
    print(f"early_bud_pair: {int(counts.get('early_bud_pair', 0))}")
    print(f"single_cell: {int(counts.get('single_cell', 0))}")
    print(f"uncertain_pair: {int(counts.get('uncertain_pair', 0))}")
    print(f"rejected_nonadjacent_pair: {int(counts.get('rejected_nonadjacent_pair', 0))}")
    print(f"Wrote rules_v2_3 tables to: {RULES_V2_3_TABLES_DIR}")


if __name__ == "__main__":
    main()
