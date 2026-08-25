"""Create a small subset for comparing YeastSAM against current MobileSAM masks."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "v6_ml_detection" / "annotations" / "image_manifest.csv"
DEFAULT_REFERENCES = PROJECT_ROOT / "v6_ml_detection" / "references" / "source_tif_reference_manifest.csv"
DEFAULT_LABELS = (
    PROJECT_ROOT
    / "v6_ml_detection"
    / "pseudo_labels"
    / "labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv"
)
DEFAULT_SAM_OVERLAYS = PROJECT_ROOT / "v6_ml_detection" / "sam_masks" / "prompted_mobile_sam_from_source_tifs" / "overlays"
DEFAULT_OUTPUT = PROJECT_ROOT / "v6_ml_detection" / "benchmarks" / "yeastsam_subset"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def condition_from_image_id(image_id: str) -> str:
    return re.sub(r"_frame_\d+$", "", image_id)


def frame_index_from_image_id(image_id: str) -> int:
    match = re.search(r"_frame_(\d+)$", image_id)
    if not match:
        return 0
    return int(match.group(1))


def choose_first_middle_last(rows: list[dict[str, str]]) -> list[str]:
    ordered = sorted(rows, key=lambda row: frame_index_from_image_id(row["image_id"]))
    if not ordered:
        return []
    indexes = {0, len(ordered) // 2, len(ordered) - 1}
    return [ordered[index]["image_id"] for index in sorted(indexes)]


def relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def copy_if_exists(src: Path, dst: Path) -> str:
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return relpath(dst)


def build_subset(
    manifest_path: Path,
    references_path: Path,
    labels_path: Path,
    sam_overlays_dir: Path,
    output_dir: Path,
    max_images: int,
) -> None:
    manifest = read_csv(manifest_path)
    references = read_csv(references_path)
    labels = read_csv(labels_path)

    manifest_by_id = {row["image_id"]: row for row in manifest}
    labels_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in labels:
        labels_by_image[row["image_id"]].append(row)

    label_counts = Counter(row["image_id"] for row in labels)

    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        by_condition[condition_from_image_id(row["image_id"])].append(row)

    selected: list[str] = []
    seen: set[str] = set()

    for condition in sorted(by_condition):
        for image_id in choose_first_middle_last(by_condition[condition]):
            if image_id not in seen:
                selected.append(image_id)
                seen.add(image_id)

    for image_id, _count in label_counts.most_common(max_images):
        if image_id not in seen:
            selected.append(image_id)
            seen.add(image_id)
        if len(selected) >= max_images:
            break

    selected = selected[:max_images]

    trans_refs = {
        row["image_id"]: row
        for row in references
        if row.get("reference_name") == "Trans"
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    subset_rows: list[dict[str, object]] = []
    subset_label_rows: list[dict[str, str]] = []

    for image_id in selected:
        meta = manifest_by_id[image_id]
        rgb_src = PROJECT_ROOT / meta["image_path"]
        rgb_dst = output_dir / "rgb" / f"{image_id}.png"

        trans_row = trans_refs.get(image_id)
        trans_src = PROJECT_ROOT / trans_row["reference_path"] if trans_row else Path()
        trans_dst = output_dir / "trans" / f"{image_id}_Trans.png"

        overlay_src = sam_overlays_dir / f"{Path(meta['image_path']).stem}_mobile_sam_overlay.png"
        overlay_dst = output_dir / "current_mobilesam_overlays" / f"{image_id}_mobile_sam_overlay.png"

        rgb_out = copy_if_exists(rgb_src, rgb_dst)
        trans_out = copy_if_exists(trans_src, trans_dst) if trans_row else ""
        overlay_out = copy_if_exists(overlay_src, overlay_dst)

        subset_rows.append(
            {
                "image_id": image_id,
                "condition": condition_from_image_id(image_id),
                "split": meta.get("split", ""),
                "label_count": label_counts[image_id],
                "rgb_path": rgb_out,
                "trans_path": trans_out,
                "current_mobilesam_overlay_path": overlay_out,
            }
        )

        subset_label_rows.extend(labels_by_image.get(image_id, []))

    write_csv(
        output_dir / "subset_images.csv",
        subset_rows,
        [
            "image_id",
            "condition",
            "split",
            "label_count",
            "rgb_path",
            "trans_path",
            "current_mobilesam_overlay_path",
        ],
    )

    label_fields = ["image_id", "class_name", "x1", "y1", "x2", "y2", "source", "review_status", "notes"]
    write_csv(output_dir / "subset_labels.csv", subset_label_rows, label_fields)

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# YeastSAM Benchmark Subset",
                "",
                "This folder contains a small representative subset for comparing YeastSAM against the current prompted MobileSAM run.",
                "",
                "Use the `trans` images as the likely YeastSAM input because YeastSAM is designed around DIC / cell-boundary segmentation.",
                "",
                "Files:",
                "",
                "- `subset_images.csv`: selected images and copied paths",
                "- `subset_labels.csv`: current pseudo-label boxes for the selected images",
                "- `rgb`: current RGB composites",
                "- `trans`: Trans reference images",
                "- `current_mobilesam_overlays`: current MobileSAM mask overlays",
                "",
                "Suggested comparison:",
                "",
                "1. Run YeastSAM on the `trans` images.",
                "2. Compare YeastSAM masks with `current_mobilesam_overlays`.",
                "3. Score boundary correctness, mother-bud split/merge behavior, neighbor inclusion, and runtime.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(subset_rows)} benchmark images to {output_dir}")
    print(f"Wrote {len(subset_label_rows)} subset labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--sam-overlays", type=Path, default=DEFAULT_SAM_OVERLAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-images", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_subset(
        manifest_path=args.manifest,
        references_path=args.references,
        labels_path=args.labels,
        sam_overlays_dir=args.sam_overlays,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
