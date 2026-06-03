"""Build rules_v2_1 with explicit Trans cell-wall features and conflict-aware pairing."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    FAR_PAIR_EXTRA_SCORE_REQUIRED,
    PAIR_IDEAL_DISTANCE_MAX,
    PAIR_IDEAL_DISTANCE_MIN,
    PAIR_MAX_DISTANCE,
    PAIR_REQUIRE_MUTUAL_TOP_K,
    PROJECTED_DIR,
    RULE_DISCOVERY_DIR,
    RULES_V2_1_DEBUG_CASES_DIR,
    RULES_V2_1_DIR,
    RULES_V2_1_OVERLAYS_DIR,
    RULES_V2_1_SAMPLED_REVIEW_DIR,
    RULES_V2_1_TABLES_DIR,
)
from my_sam_pipeline.io_utils import ensure_dir, read_tiff_stack  # noqa: E402
from my_sam_pipeline.rules_v2_1 import (  # noqa: E402
    build_rules_v2_1_summary,
    classify_rules_v2_1,
    write_existing_feature_report,
)


def iter_conditions(projected_dir: Path) -> list[str]:
    return sorted(path.name[: -len("_Trans.tif")] for path in projected_dir.glob("*_Trans.tif"))


def main() -> None:
    ensure_dir(RULES_V2_1_DIR)
    ensure_dir(RULES_V2_1_TABLES_DIR)
    ensure_dir(RULES_V2_1_OVERLAYS_DIR)
    ensure_dir(RULES_V2_1_DEBUG_CASES_DIR)
    ensure_dir(RULES_V2_1_SAMPLED_REVIEW_DIR)

    report_path = RULES_V2_1_TABLES_DIR / "rules_v2_existing_feature_report.txt"
    write_existing_feature_report(report_path)

    nuclei_v1 = pd.read_csv(RULE_DISCOVERY_DIR / "tables" / "nucleus_candidates.csv")
    budneck_v1 = pd.read_csv(RULE_DISCOVERY_DIR / "tables" / "budneck_candidates.csv")
    nuclei_v2 = pd.read_csv(RULE_DISCOVERY_DIR / "rules_v2" / "tables" / "nucleus_candidates_v2.csv")
    pair_v2 = pd.read_csv(RULE_DISCOVERY_DIR / "rules_v2" / "tables" / "pair_candidates_v2.csv")
    v2_classification = pd.read_csv(RULE_DISCOVERY_DIR / "rules_v2" / "tables" / "rules_v2_classification.csv")

    gfp_stacks: dict[str, object] = {}
    trans_stacks: dict[str, object] = {}
    conditions = iter_conditions(PROJECTED_DIR)
    for condition in conditions:
        gfp_stacks[condition] = read_tiff_stack(PROJECTED_DIR / f"{condition}_GFP_projected.tif")
        trans_stacks[condition] = read_tiff_stack(PROJECTED_DIR / f"{condition}_Trans.tif")

    cell_wall_df, debug_df, conflict_df, missing_nucleus_df, classification_df = classify_rules_v2_1(
        trans_stacks=trans_stacks,
        gfp_stacks=gfp_stacks,
        nuclei_v1_df=nuclei_v1,
        nuclei_v2_df=nuclei_v2,
        budneck_df=budneck_v1,
        pair_v2_df=pair_v2,
        pair_require_mutual_top_k=PAIR_REQUIRE_MUTUAL_TOP_K,
        pair_ideal_distance_min=PAIR_IDEAL_DISTANCE_MIN,
        pair_ideal_distance_max=PAIR_IDEAL_DISTANCE_MAX,
        pair_max_distance=PAIR_MAX_DISTANCE,
        far_pair_extra_score_required=FAR_PAIR_EXTRA_SCORE_REQUIRED,
    )
    summary_df = build_rules_v2_1_summary(
        nuclei_v1_df=nuclei_v1,
        nuclei_v2_df=nuclei_v2,
        v2_classification_df=v2_classification,
        classification_df=classification_df,
        debug_df=debug_df,
        conflict_df=conflict_df,
    )

    cell_wall_df.to_csv(RULES_V2_1_TABLES_DIR / "cell_wall_features.csv", index=False)
    debug_df.to_csv(RULES_V2_1_TABLES_DIR / "pair_assignment_debug.csv", index=False)
    conflict_df.to_csv(RULES_V2_1_TABLES_DIR / "pair_conflicts_resolved.csv", index=False)
    missing_nucleus_df.to_csv(RULES_V2_1_TABLES_DIR / "missing_nucleus_diagnostics.csv", index=False)
    classification_df.to_csv(RULES_V2_1_TABLES_DIR / "rules_v2_1_classification.csv", index=False)
    summary_df.to_csv(RULES_V2_1_TABLES_DIR / "rules_v2_1_summary.csv", index=False)

    counts = classification_df["final_class"].value_counts()
    print(f"mother_bud_pair: {int(counts.get('mother_bud_pair', 0))}")
    print(f"early_bud_pair: {int(counts.get('early_bud_pair', 0))}")
    print(f"uncertain_pair: {int(counts.get('uncertain_pair', 0))}")
    print(f"single_cell: {int(counts.get('single_cell', 0))}")
    print(f"Wrote rules_v2_1 tables to: {RULES_V2_1_TABLES_DIR}")


if __name__ == "__main__":
    main()
