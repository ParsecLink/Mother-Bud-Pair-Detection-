"""Create overlays and manual-review patches for rules_v2_3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    MERGED_RGB_ENHANCED_DIR,
    RULES_V2_3_DEBUG_CASES_DIR,
    RULES_V2_3_OVERLAYS_DIR,
    RULES_V2_3_SAMPLED_REVIEW_DIR,
    RULES_V2_3_TABLES_DIR,
    RULES_V2_PATCH_SIZE,
)
from my_sam_pipeline.visualization import save_pair_patch, save_rules_v2_3_overlay  # noqa: E402


RNG = np.random.default_rng(23)


def background_png_path(condition: str, frame: int) -> Path:
    return MERGED_RGB_ENHANCED_DIR / f"{condition}_frame_{int(frame):03d}.png"


def pair_line_color(class_name: str) -> tuple[int, int, int]:
    return {
        "mother_bud_pair": (0, 255, 255),
        "early_bud_pair": (255, 165, 0),
        "rejected_nonadjacent_pair": (255, 75, 75),
        "uncertain_pair": (160, 160, 160),
    }.get(class_name, (180, 180, 180))


def sample_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if len(df) <= n:
        return df.copy()
    indices = RNG.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[sorted(indices)].copy()


def patch_center(row: pd.Series, nuclei_df: pd.DataFrame, budneck_df: pd.DataFrame) -> tuple[float, float]:
    points_y: list[float] = []
    points_x: list[float] = []
    for nucleus_col in ["nucleus_a_id", "nucleus_b_id"]:
        if pd.notna(row[nucleus_col]):
            nucleus_match = nuclei_df[nuclei_df["nucleus_id"] == int(row[nucleus_col])]
            if not nucleus_match.empty:
                points_y.append(float(nucleus_match.iloc[0]["centroid_y"]))
                points_x.append(float(nucleus_match.iloc[0]["centroid_x"]))
    if pd.notna(row.get("budneck_id", np.nan)):
        bud_match = budneck_df[budneck_df["budneck_id"] == int(row["budneck_id"])]
        if not bud_match.empty:
            points_y.append(float(bud_match.iloc[0]["centroid_y"]))
            points_x.append(float(bud_match.iloc[0]["centroid_x"]))
    if not points_y:
        return 128.0, 128.0
    return float(np.mean(points_y)), float(np.mean(points_x))


def build_overlay_frames(classification_df: pd.DataFrame) -> pd.DataFrame:
    interesting = classification_df[
        (classification_df["final_class"] != "single_cell")
        | classification_df["wrong_far_assignment_warning"].fillna(False).astype(bool)
        | classification_df["previous_v2_1_class"].notna()
        | classification_df["downgrade_reason"].fillna("").str.contains("line_misses_budneck")
    ].copy()
    return interesting[["condition", "frame"]].drop_duplicates().sort_values(["condition", "frame"])


def build_sample_manifest(classification_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    frames.append(classification_df[classification_df["final_class"] == "mother_bud_pair"].assign(sample_group="mother_bud_pair"))
    frames.append(classification_df[classification_df["final_class"] == "early_bud_pair"].assign(sample_group="early_bud_pair"))
    frames.append(sample_rows(classification_df[classification_df["final_class"] == "uncertain_pair"], 50).assign(sample_group="uncertain_pair"))
    frames.append(sample_rows(classification_df[classification_df["final_class"] == "rejected_nonadjacent_pair"], 50).assign(sample_group="rejected_nonadjacent_pair"))
    frames.append(sample_rows(classification_df[classification_df["wrong_far_assignment_warning"].fillna(False).astype(bool)], 50).assign(sample_group="wrong_far_assignment_warning"))
    frames.append(sample_rows(classification_df[classification_df["downgrade_reason"].fillna("").str.contains("line_misses_budneck")], 50).assign(sample_group="line_misses_budneck"))
    frames.append(sample_rows(classification_df[classification_df["previous_v2_1_class"].notna() & (classification_df["previous_v2_1_class"] != classification_df["final_class"])], 50).assign(sample_group="class_changed_from_v2_1"))
    manifest = pd.concat(frames, ignore_index=True, sort=False)
    return manifest


def main() -> None:
    for path in [RULES_V2_3_OVERLAYS_DIR, RULES_V2_3_DEBUG_CASES_DIR, RULES_V2_3_SAMPLED_REVIEW_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    nuclei_v2 = pd.read_csv(PROJECT_ROOT / "rule_discovery" / "rules_v2" / "tables" / "nucleus_candidates_v2.csv")
    budneck_v1 = pd.read_csv(PROJECT_ROOT / "rule_discovery" / "tables" / "budneck_candidates.csv")
    classification_df = pd.read_csv(RULES_V2_3_TABLES_DIR / "rules_v2_3_classification.csv")

    overlay_frames = build_overlay_frames(classification_df)
    overlay_count = 0
    for row in overlay_frames.itertuples(index=False):
        condition = str(row.condition)
        frame = int(row.frame)
        background = background_png_path(condition, frame)
        if not background.exists():
            continue
        nuclei_frame = nuclei_v2[(nuclei_v2["condition"] == condition) & (nuclei_v2["frame"] == frame)].copy()
        bud_frame = budneck_v1[(budneck_v1["condition"] == condition) & (budneck_v1["frame"] == frame)].copy()
        class_frame = classification_df[(classification_df["condition"] == condition) & (classification_df["frame"] == frame)].copy()
        class_frame = class_frame[
            (class_frame["final_class"] != "single_cell")
            | class_frame["wrong_far_assignment_warning"].fillna(False).astype(bool)
            | class_frame["downgrade_reason"].fillna("").str.contains("line_misses_budneck")
        ].copy()
        if len(class_frame) > 18:
            class_frame = class_frame.sort_values(
                ["final_class", "cells_are_adjacent", "nucleus_line_hits_budneck", "pair_score"],
                ascending=[True, False, False, False],
            ).head(18)
        save_rules_v2_3_overlay(
            background_png=background,
            output_path=RULES_V2_3_OVERLAYS_DIR / f"{condition}_frame_{frame:03d}.png",
            nuclei_df=nuclei_frame,
            budneck_df=bud_frame,
            classification_df=class_frame,
        )
        overlay_count += 1

    manifest = build_sample_manifest(classification_df)
    review_rows: list[dict[str, object]] = []
    patch_index = 1
    debug_subset = manifest[
        manifest["sample_group"].isin(["wrong_far_assignment_warning", "line_misses_budneck", "class_changed_from_v2_1"])
    ].copy()
    debug_subset = debug_subset.head(150)

    for save_dir, subset in [
        (RULES_V2_3_SAMPLED_REVIEW_DIR, manifest),
        (RULES_V2_3_DEBUG_CASES_DIR, debug_subset),
    ]:
        seen_patch_ids: set[str] = set()
        for row in subset.itertuples(index=False):
            condition = str(row.condition)
            frame = int(row.frame)
            background = background_png_path(condition, frame)
            if not background.exists():
                continue
            nuclei_frame = nuclei_v2[(nuclei_v2["condition"] == condition) & (nuclei_v2["frame"] == frame)].copy()
            bud_frame = budneck_v1[(budneck_v1["condition"] == condition) & (budneck_v1["frame"] == frame)].copy()
            nucleus_points: list[tuple[float, float, int]] = []
            for nucleus_col in ["nucleus_a_id", "nucleus_b_id"]:
                if pd.notna(getattr(row, nucleus_col)):
                    nucleus_match = nuclei_frame[nuclei_frame["nucleus_id"] == int(getattr(row, nucleus_col))]
                    if not nucleus_match.empty:
                        nucleus_points.append(
                            (
                                float(nucleus_match.iloc[0]["centroid_x"]),
                                float(nucleus_match.iloc[0]["centroid_y"]),
                                int(nucleus_match.iloc[0]["nucleus_id"]),
                            )
                        )
            budneck_info = None
            if pd.notna(row.budneck_id):
                bud_match = bud_frame[bud_frame["budneck_id"] == int(row.budneck_id)]
                if not bud_match.empty:
                    budneck_info = (
                        float(bud_match.iloc[0]["centroid_x"]),
                        float(bud_match.iloc[0]["centroid_y"]),
                        float(bud_match.iloc[0]["orientation_deg"]),
                        float(bud_match.iloc[0]["major_axis_length"]),
                        int(bud_match.iloc[0]["budneck_id"]),
                    )
            line = None
            if len(nucleus_points) == 2:
                line = (
                    (nucleus_points[0][1], nucleus_points[0][0]),
                    (nucleus_points[1][1], nucleus_points[1][0]),
                    pair_line_color(str(row.final_class)),
                )
            elif len(nucleus_points) == 1 and budneck_info is not None:
                line = (
                    (nucleus_points[0][1], nucleus_points[0][0]),
                    (budneck_info[1], budneck_info[0]),
                    pair_line_color(str(row.final_class)),
                )
            cy, cx = patch_center(pd.Series(row._asdict()), nuclei_frame, bud_frame)
            patch_id = f"{row.sample_group}_{patch_index:03d}_{condition}_frame_{frame:03d}"
            if patch_id in seen_patch_ids:
                continue
            seen_patch_ids.add(patch_id)
            save_pair_patch(
                background_png=background,
                output_path=save_dir / f"{patch_id}.png",
                nucleus_points=nucleus_points,
                budneck_info=budneck_info,
                patch_center_y=cy,
                patch_center_x=cx,
                patch_size=RULES_V2_PATCH_SIZE,
                pair_line=line,
                class_label=f"{row.final_class} | {row.sample_group}",
            )
            if save_dir == RULES_V2_3_SAMPLED_REVIEW_DIR:
                review_rows.append(
                    {
                        "condition": condition,
                        "frame": frame,
                        "candidate_id": int(row.candidate_id),
                        "previous_v2_1_class": row.previous_v2_1_class if pd.notna(row.previous_v2_1_class) else "",
                        "current_v2_3_class": str(row.final_class),
                        "budneck_id": int(row.budneck_id) if pd.notna(row.budneck_id) else "",
                        "nucleus_a_id": int(row.nucleus_a_id) if pd.notna(row.nucleus_a_id) else "",
                        "nucleus_b_id": int(row.nucleus_b_id) if pd.notna(row.nucleus_b_id) else "",
                        "cell_a_id": int(row.cell_a_id) if pd.notna(row.cell_a_id) else "",
                        "cell_b_id": int(row.cell_b_id) if pd.notna(row.cell_b_id) else "",
                        "cells_are_adjacent": bool(row.cells_are_adjacent) if pd.notna(row.cells_are_adjacent) else "",
                        "nucleus_line_hits_budneck": bool(row.nucleus_line_hits_budneck) if pd.notna(row.nucleus_line_hits_budneck) else "",
                        "budneck_projection_fraction": row.budneck_projection_fraction if pd.notna(row.budneck_projection_fraction) else "",
                        "budneck_distance_to_nucleus_line": row.budneck_distance_to_nucleus_line if pd.notna(row.budneck_distance_to_nucleus_line) else "",
                        "closer_adjacent_partner_exists": bool(row.closer_adjacent_partner_exists) if pd.notna(row.closer_adjacent_partner_exists) else "",
                        "wrong_far_assignment_warning": bool(row.wrong_far_assignment_warning) if pd.notna(row.wrong_far_assignment_warning) else "",
                        "decision_reason": str(row.decision_reason),
                        "manual_class": "",
                        "manual_correct_partner_ids": "",
                        "manual_notes": "",
                    }
                )
                patch_index += 1

    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(RULES_V2_3_TABLES_DIR / "manual_review_v2_3.csv", index=False)

    print(f"Saved {overlay_count} rules_v2_3 overlays")
    print(f"Saved {len(review_df)} sampled review patches")
    print(f"Wrote manual review CSV to: {RULES_V2_3_TABLES_DIR / 'manual_review_v2_3.csv'}")


if __name__ == "__main__":
    main()
