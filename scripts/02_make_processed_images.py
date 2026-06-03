#!/usr/bin/env python
"""Generate processed channel images, merged RGB images, and contact sheets."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_sam_pipeline.image_utils import fluor_display, trans_display  # noqa: E402
from my_sam_pipeline.io_utils import ensure_dir, read_tiff_stack  # noqa: E402
from my_sam_pipeline.visualization import (  # noqa: E402
    green_overlay,
    magenta_overlay,
    make_contact_sheet,
    merge_rgb,
    merge_rgb_enhanced,
    save_gray_png,
    save_rgb_png,
)


PIC_DIR = PROJECT_ROOT / "pic"
PROJECTED_DIR = PIC_DIR / "projected_tifs"
TRANS_DIR = PIC_DIR / "trans_only"
GFP_DIR = PIC_DIR / "gfp_only"
MCHERRY_DIR = PIC_DIR / "mcherry_only"
MERGED_DIR = PIC_DIR / "merged_rgb"
MERGED_ENHANCED_DIR = PIC_DIR / "merged_rgb_enhanced"
CONTACT_DIR = PIC_DIR / "contact_sheets"


def _iter_conditions() -> list[str]:
    return sorted(path.name.replace("_Trans.tif", "") for path in PROJECTED_DIR.glob("*_Trans.tif"))


def main() -> None:
    for directory in [TRANS_DIR, GFP_DIR, MCHERRY_DIR, MERGED_DIR, MERGED_ENHANCED_DIR, CONTACT_DIR]:
        ensure_dir(directory)

    processed_frames = 0
    for condition in _iter_conditions():
        gfp_stack = read_tiff_stack(PROJECTED_DIR / f"{condition}_GFP_projected.tif")
        mcherry_stack = read_tiff_stack(PROJECTED_DIR / f"{condition}_mCherry_projected.tif")
        trans_stack = read_tiff_stack(PROJECTED_DIR / f"{condition}_Trans.tif")

        contact_paths: list[Path] = []
        for frame_index in range(int(trans_stack.shape[0])):
            trans_frame = trans_display(trans_stack[frame_index])
            gfp_frame = fluor_display(gfp_stack[frame_index], gaussian_sigma=1.0, tophat_radius=25, gamma=0.82)
            mcherry_frame = fluor_display(mcherry_stack[frame_index], gaussian_sigma=1.0, tophat_radius=10, gamma=0.82)

            trans_path = TRANS_DIR / f"{condition}_frame_{frame_index:03d}.png"
            gfp_path = GFP_DIR / f"{condition}_frame_{frame_index:03d}.png"
            mcherry_path = MCHERRY_DIR / f"{condition}_frame_{frame_index:03d}.png"
            merged_path = MERGED_DIR / f"{condition}_frame_{frame_index:03d}.png"
            merged_enhanced_path = MERGED_ENHANCED_DIR / f"{condition}_frame_{frame_index:03d}.png"

            save_gray_png(trans_path, trans_frame)
            save_rgb_png(gfp_path, green_overlay(gfp_frame))
            save_rgb_png(mcherry_path, magenta_overlay(mcherry_frame))
            save_rgb_png(merged_path, merge_rgb(trans_frame, gfp_frame, mcherry_frame))
            save_rgb_png(merged_enhanced_path, merge_rgb_enhanced(trans_frame, gfp_frame, mcherry_frame))

            contact_paths.append(merged_enhanced_path)
            processed_frames += 1

        make_contact_sheet(contact_paths, CONTACT_DIR / f"{condition}_contact_sheet.png")
        print(f"[{condition}] processed {trans_stack.shape[0]} aligned frames")

    print(f"Processed {processed_frames} aligned frames total")


if __name__ == "__main__":
    main()
