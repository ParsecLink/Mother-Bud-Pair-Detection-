"""Extract interpretable rule-discovery features from projected TIFF stacks."""

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
    RULE_FIGURES_DIR,
    RULE_OVERLAYS_DIR,
    RULE_PATCHES_DIR,
    RULE_TABLES_DIR,
)
from my_sam_pipeline.io_utils import ensure_dir, read_tiff_stack  # noqa: E402
from my_sam_pipeline.rule_features import (  # noqa: E402
    build_budneck_candidates,
    build_nucleus_candidates,
    build_pair_candidates,
    detect_channel_objects,
    suggest_thresholds,
)


def iter_conditions(projected_dir: Path) -> list[str]:
    conditions = []
    for path in sorted(projected_dir.glob("*_Trans.tif")):
        conditions.append(path.name[: -len("_Trans.tif")])
    return conditions


def main() -> None:
    ensure_dir(RULE_DISCOVERY_DIR)
    ensure_dir(RULE_TABLES_DIR)
    ensure_dir(RULE_FIGURES_DIR)
    ensure_dir(RULE_OVERLAYS_DIR)
    ensure_dir(RULE_PATCHES_DIR)

    fluorescence_rows: list[dict[str, object]] = []
    gfp_rows: list[dict[str, object]] = []
    mcherry_rows: list[dict[str, object]] = []

    conditions = iter_conditions(PROJECTED_DIR)
    print(f"Found {len(conditions)} projected conditions")
    for condition in conditions:
        gfp_stack = read_tiff_stack(PROJECTED_DIR / f"{condition}_GFP_projected.tif")
        mcherry_stack = read_tiff_stack(PROJECTED_DIR / f"{condition}_mCherry_projected.tif")
        frame_count = int(gfp_stack.shape[0])
        print(f"[{condition}] extracting GFP and mCherry objects from {frame_count} frames")
        for frame_index in range(frame_count):
            gfp_objects, _ = detect_channel_objects(gfp_stack[frame_index], "GFP", condition, frame_index)
            mcherry_objects, _ = detect_channel_objects(mcherry_stack[frame_index], "mCherry", condition, frame_index)
            fluorescence_rows.extend(gfp_objects)
            fluorescence_rows.extend(mcherry_objects)
            gfp_rows.extend(gfp_objects)
            mcherry_rows.extend(mcherry_objects)

    fluorescence_df = pd.DataFrame(fluorescence_rows)
    gfp_df = pd.DataFrame(gfp_rows)
    mcherry_df = pd.DataFrame(mcherry_rows)
    nucleus_df = build_nucleus_candidates(gfp_df)
    budneck_df = build_budneck_candidates(mcherry_df)
    pair_df = build_pair_candidates(
        nucleus_candidates=nucleus_df,
        budneck_candidates=budneck_df,
        pair_distance_min=PAIR_DISTANCE_MIN,
        pair_distance_max=PAIR_DISTANCE_MAX,
    )
    thresholds_df = suggest_thresholds(gfp_df, budneck_df, pair_df)

    fluorescence_out = RULE_TABLES_DIR / "fluorescence_objects.csv"
    nucleus_out = RULE_TABLES_DIR / "nucleus_candidates.csv"
    budneck_out = RULE_TABLES_DIR / "budneck_candidates.csv"
    pair_out = RULE_TABLES_DIR / "pair_candidates.csv"
    thresholds_out = RULE_TABLES_DIR / "suggested_thresholds.csv"

    fluorescence_df.to_csv(fluorescence_out, index=False)
    nucleus_df.to_csv(nucleus_out, index=False)
    budneck_df.to_csv(budneck_out, index=False)
    pair_df.to_csv(pair_out, index=False)
    thresholds_df.to_csv(thresholds_out, index=False)

    print(f"GFP objects: {len(gfp_df)}")
    print(f"mCherry objects: {len(mcherry_df)}")
    print(f"Nucleus candidates: {len(nucleus_df)}")
    print(f"Bud-neck candidates: {len(budneck_df)}")
    print(f"Pair candidates: {len(pair_df)}")
    print(f"Wrote tables to: {RULE_TABLES_DIR}")


if __name__ == "__main__":
    main()
