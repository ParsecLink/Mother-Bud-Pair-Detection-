"""Build rules_v2 features and classifications with improved nuclei and Trans morphology."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    PAIR_DISTANCE_MAX,
    PAIR_DISTANCE_MIN,
    PROJECTED_DIR,
    RULE_DISCOVERY_DIR,
    RULES_V2_DIR,
    RULES_V2_OVERLAYS_DIR,
    RULES_V2_SAMPLED_REVIEW_DIR,
    RULES_V2_TABLES_DIR,
)
from my_sam_pipeline.io_utils import ensure_dir, read_tiff_stack  # noqa: E402
from my_sam_pipeline.rule_features import build_pair_candidates  # noqa: E402
from my_sam_pipeline.rules_v2 import (  # noqa: E402
    build_nucleus_candidates_v2,
    build_rules_v2_summary,
    classify_rules_v2,
)


def iter_conditions(projected_dir: Path) -> list[str]:
    return sorted(path.name[: -len("_Trans.tif")] for path in projected_dir.glob("*_Trans.tif"))


def main() -> None:
    ensure_dir(RULES_V2_DIR)
    ensure_dir(RULES_V2_TABLES_DIR)
    ensure_dir(RULES_V2_OVERLAYS_DIR)
    ensure_dir(RULES_V2_SAMPLED_REVIEW_DIR)

    nuclei_v1 = pd.read_csv(RULE_DISCOVERY_DIR / "tables" / "nucleus_candidates.csv")
    budneck_v1 = pd.read_csv(RULE_DISCOVERY_DIR / "tables" / "budneck_candidates.csv")

    gfp_stacks: dict[str, object] = {}
    trans_stacks: dict[str, object] = {}
    conditions = iter_conditions(PROJECTED_DIR)
    print(f"Found {len(conditions)} projected conditions for rules_v2")
    for condition in conditions:
        gfp_stacks[condition] = read_tiff_stack(PROJECTED_DIR / f"{condition}_GFP_projected.tif")
        trans_stacks[condition] = read_tiff_stack(PROJECTED_DIR / f"{condition}_Trans.tif")

    nuclei_v2 = build_nucleus_candidates_v2(gfp_stacks=gfp_stacks, v1_nuclei=nuclei_v1)
    pair_candidates_v2 = build_pair_candidates(
        nucleus_candidates=nuclei_v2,
        budneck_candidates=budneck_v1,
        pair_distance_min=PAIR_DISTANCE_MIN,
        pair_distance_max=PAIR_DISTANCE_MAX,
    )
    morphology_df, classification_df = classify_rules_v2(
        trans_stacks=trans_stacks,
        nucleus_candidates_v2=nuclei_v2,
        budneck_candidates=budneck_v1,
        pair_candidates_v2=pair_candidates_v2,
    )
    summary_df = build_rules_v2_summary(
        classification_df=classification_df,
        nuclei_v2_df=nuclei_v2,
        nuclei_v1_df=nuclei_v1,
    )

    nuclei_out = RULES_V2_TABLES_DIR / "nucleus_candidates_v2.csv"
    pair_out = RULES_V2_TABLES_DIR / "pair_candidates_v2.csv"
    morph_out = RULES_V2_TABLES_DIR / "trans_morphology_features.csv"
    class_out = RULES_V2_TABLES_DIR / "rules_v2_classification.csv"
    summary_out = RULES_V2_TABLES_DIR / "rules_v2_summary.csv"
    nuclei_v2.to_csv(nuclei_out, index=False)
    pair_candidates_v2.to_csv(pair_out, index=False)
    morphology_df.to_csv(morph_out, index=False)
    classification_df.to_csv(class_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    class_counts = classification_df["rule_class"].value_counts()
    recovered_nuclei = int(nuclei_v2["recovered_vs_v1"].sum()) if not nuclei_v2.empty else 0
    print(f"V1 nuclei: {len(nuclei_v1)}")
    print(f"V2 nuclei: {len(nuclei_v2)}")
    print(f"Recovered nuclei vs v1: {recovered_nuclei}")
    print(f"mother_bud_pair: {int(class_counts.get('mother_bud_pair', 0))}")
    print(f"early_bud_pair: {int(class_counts.get('early_bud_pair', 0))}")
    print(f"single_cell: {int(class_counts.get('single_cell', 0))}")
    print(f"Wrote rules_v2 tables to: {RULES_V2_TABLES_DIR}")


if __name__ == "__main__":
    main()
