#!/usr/bin/env python
"""Project fluorescence stacks so they align with Trans frames."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_sam_pipeline.image_utils import max_project_blocks  # noqa: E402
from my_sam_pipeline.io_utils import collect_condition_triplets, ensure_dir, read_tiff_stack, write_tiff_stack  # noqa: E402


IMAGE_DIR = PROJECT_ROOT.parent / "image"
PROJECTED_DIR = PROJECT_ROOT / "pic" / "projected_tifs"
PROJECTION_GROUP_SIZE = 10


def main() -> None:
    ensure_dir(PROJECTED_DIR)
    triplets = collect_condition_triplets(IMAGE_DIR, projection_group_size=PROJECTION_GROUP_SIZE)
    print(f"Found {len(triplets)} valid condition triplets")

    for triplet in triplets:
        condition = str(triplet["condition"])
        gfp_stack = read_tiff_stack(Path(triplet["gfp_path"]))
        mcherry_stack = read_tiff_stack(Path(triplet["mcherry_path"]))
        trans_stack = read_tiff_stack(Path(triplet["trans_path"]))

        gfp_projected = max_project_blocks(gfp_stack, PROJECTION_GROUP_SIZE)
        mcherry_projected = max_project_blocks(mcherry_stack, PROJECTION_GROUP_SIZE)

        write_tiff_stack(PROJECTED_DIR / f"{condition}_GFP_projected.tif", gfp_projected)
        write_tiff_stack(PROJECTED_DIR / f"{condition}_mCherry_projected.tif", mcherry_projected)
        write_tiff_stack(PROJECTED_DIR / f"{condition}_Trans.tif", trans_stack)
        print(f"[{condition}] saved projected TIFFs with {trans_stack.shape[0]} aligned frames")


if __name__ == "__main__":
    main()
