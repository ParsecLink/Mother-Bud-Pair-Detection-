"""Create balanced HTML review batches for multiple manual reviewers."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
GEOMETRY_LABELS = V6_DIR / "pseudo_labels" / "labels_pseudo_mobile_sam_from_source_tifs_mask_aware_geometry_flagged.csv"
CURRENT_LABELS = V6_DIR / "pseudo_labels" / "labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv"
DEFAULT_LABELS = GEOMETRY_LABELS if GEOMETRY_LABELS.exists() else CURRENT_LABELS
DEFAULT_REFERENCES = V6_DIR / "references" / "source_tif_reference_manifest.csv"
DEFAULT_OUTPUT = V6_DIR / "review" / "training_audit_batches"
DEFAULT_FROM_SCRATCH_OUTPUT = V6_DIR / "review" / "training_audit_batches_draw_from_scratch"
DEFAULT_CLASS_BUTTON_OUTPUT = V6_DIR / "review" / "training_audit_batches_draw_from_scratch_class_buttons"
LABEL_COLUMNS = ["image_id", "class_name", "x1", "y1", "x2", "y2", "source", "review_status", "notes"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def condition_from_image_id(image_id: str) -> str:
    if "_frame_" not in image_id:
        return image_id
    return image_id.rsplit("_frame_", 1)[0]


def load_builder_module():
    path = PROJECT_ROOT / "scripts" / "50_make_v6_label_review_app.py"
    spec = importlib.util.spec_from_file_location("v6_label_review_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_images(
    manifest: list[dict[str, str]],
    labels: list[dict[str, str]],
    reviewer_count: int,
) -> dict[int, list[str]]:
    labels_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in labels:
        labels_by_image[row["image_id"]].append(row)

    images_by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        images_by_condition[condition_from_image_id(row["image_id"])].append(row)

    assignments: dict[int, list[str]] = {idx: [] for idx in range(1, reviewer_count + 1)}
    reviewer_label_load = Counter({idx: 0 for idx in range(1, reviewer_count + 1)})
    reviewer_image_load = Counter({idx: 0 for idx in range(1, reviewer_count + 1)})
    reviewer_condition_load: dict[int, Counter[str]] = {
        idx: Counter() for idx in range(1, reviewer_count + 1)
    }

    for condition, rows in sorted(images_by_condition.items()):
        ordered = sorted(
            rows,
            key=lambda row: (len(labels_by_image[row["image_id"]]), row["image_id"]),
            reverse=True,
        )
        for row in ordered:
            image_id = row["image_id"]
            label_count = len(labels_by_image[image_id])
            reviewer = min(
                range(1, reviewer_count + 1),
                key=lambda idx: (
                    reviewer_condition_load[idx][condition],
                    reviewer_label_load[idx],
                    reviewer_image_load[idx],
                    idx,
                ),
            )
            assignments[reviewer].append(image_id)
            reviewer_label_load[reviewer] += label_count
            reviewer_image_load[reviewer] += 1
            reviewer_condition_load[reviewer][condition] += 1

    return assignments


def batch_readme(
    reviewer_id: str,
    image_count: int,
    label_count: int,
    from_scratch: bool,
    class_draw_buttons: bool,
) -> str:
    if from_scratch and class_draw_buttons:
        mode_text = f"""This batch starts with **no visible boxes**. Draw every correct object box manually.

This batch contains:

- Images: {image_count}
- Hidden seed pseudo-label boxes used only for balancing: {label_count}

## Review Instructions

1. Check every image in this batch.
2. Click `Draw single_cell`, `Draw early_bud_pair`, or `Draw mother_bud_pair`, then draw boxes of that class.
3. To stop drawing and select existing boxes, click the active draw button again.
4. If a drawn box is wrong, select it and click `Delete selected`.
5. If a class is wrong, select the box and choose the correct class in the right panel.
6. Use the `Reference` button and selector to compare RGB, Trans, GFP, and mCherry views.
7. Use the left search box or normal vertical scroll bar to navigate.
8. Click `Save` when pausing work. The saved progress stays in this browser for this batch.
9. Use `Undo` to reverse the most recent label-editing change.
"""
    elif from_scratch:
        mode_text = f"""This batch starts with **no visible boxes**. Draw every correct object box manually.

This batch contains:

- Images: {image_count}
- Hidden seed pseudo-label boxes used only for balancing: {label_count}

## Review Instructions

1. Check every image in this batch.
2. Choose the class beside `Draw box`, then draw each object box manually.
3. If a drawn box is wrong, select it and click `Delete selected`.
4. If a class is wrong, select the box and choose the correct class.
5. Use the `Reference` button and selector to compare RGB, Trans, GFP, and mCherry views.
6. Use the left search box or normal vertical scroll bar to navigate.
7. Click `Save` when pausing work. The saved progress stays in this browser for this batch.
8. Use `Undo` to reverse the most recent label-editing change.
"""
    else:
        mode_text = f"""This batch contains:

- Images: {image_count}
- Seed pseudo-label boxes: {label_count}

## Review Instructions

1. Check every image in this batch.
2. If all boxes on the current image are correct, click `Mark image reviewed`.
3. If a class is wrong, select the box and choose the correct class. It will become `reviewed`.
4. If a box is false positive, select it and click `Delete selected`.
5. If a box is missing or badly sized, use `Draw box` to add a corrected box, then delete the bad one if needed.
6. Use the `Reference` button and selector to compare RGB, Trans, GFP, and mCherry views.
7. Use the left search box or normal vertical scroll bar to navigate.
8. Click `Save` when pausing work. The saved progress stays in this browser for this batch.
9. Use `Undo` to reverse the most recent label-editing change.
"""
    return f"""# {reviewer_id} Training Audit Batch

Open `index.html` locally in Chrome or Edge. Do not use Google Drive preview.

{mode_text}

## Files To Send Back

When finished, click:

- `Export Audit CSV`
- `Export Reviewed CSV`

Send both CSV files back. If only one can be sent, send `labels_audit_export.csv`.
"""


def master_index(batch_rows: list[dict[str, Any]], from_scratch: bool, class_draw_buttons: bool) -> str:
    if from_scratch and class_draw_buttons:
        title = "Draw-From-Scratch Class-Button Training Audit Batches"
    else:
        title = "Draw-From-Scratch Training Audit Batches" if from_scratch else "Training Audit Batches"
    label_header = "Hidden Seed Boxes" if from_scratch else "Seed Boxes"
    description = (
        "These batches start with no visible boxes; reviewers draw all boxes manually with class-specific draw buttons."
        if from_scratch and class_draw_buttons
        else "These batches start with no visible boxes; reviewers draw all boxes manually."
        if from_scratch
        else "Open one group batch at a time. Google Drive preview will show HTML code; download and open locally."
    )
    items = "\n".join(
        (
            "<tr>"
            f"<td>{html.escape(str(row['reviewer_id']))}</td>"
            f"<td>{html.escape(str(row['image_count']))}</td>"
            f"<td>{html.escape(str(row['label_count']))}</td>"
            f"<td><a href=\"{html.escape(str(row['relative_index_html']).replace(chr(92), '/'))}\">open</a></td>"
            "</tr>"
        )
        for row in batch_rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; background: #111; color: #eee; margin: 24px; }}
a {{ color: #8fd3ff; }}
table {{ border-collapse: collapse; min-width: 680px; }}
td, th {{ border: 1px solid #333; padding: 8px 10px; }}
th {{ background: #222; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>{description}</p>
<table>
<thead><tr><th>Reviewer</th><th>Images</th><th>{label_header}</th><th>HTML</th></tr></thead>
<tbody>
{items}
</tbody>
</table>
</body>
</html>
"""


def create_batches(
    manifest_path: Path,
    labels_path: Path,
    references_path: Path,
    output_dir: Path,
    reviewer_count: int,
    from_scratch: bool,
    class_draw_buttons: bool,
) -> None:
    manifest = read_csv(manifest_path)
    labels = read_csv(labels_path)
    references = read_csv(references_path)

    assignments = split_images(manifest, labels, reviewer_count)
    manifest_by_id = {row["image_id"]: row for row in manifest}
    labels_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    references_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in labels:
        labels_by_image[row["image_id"]].append(row)
    for row in references:
        references_by_image[row["image_id"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    builder = load_builder_module()
    batch_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    for reviewer_idx in range(1, reviewer_count + 1):
        reviewer_id = f"Group{reviewer_idx}"
        reviewer_dir = output_dir / reviewer_id
        returned_dir = reviewer_dir / "returned"
        returned_dir.mkdir(parents=True, exist_ok=True)
        (returned_dir / "PUT_RETURNED_CSV_FILES_HERE.txt").write_text(
            "Put this group's returned labels_audit_export.csv and labels_reviewed_only_export.csv here.\n",
            encoding="utf-8",
        )

        image_ids = assignments[reviewer_idx]
        batch_manifest = [manifest_by_id[image_id] for image_id in image_ids]
        batch_labels = [label for image_id in image_ids for label in labels_by_image.get(image_id, [])]
        editor_labels = [] if from_scratch else batch_labels
        batch_references = [ref for image_id in image_ids for ref in references_by_image.get(image_id, [])]

        manifest_out = reviewer_dir / "manifest.csv"
        labels_out = reviewer_dir / "labels_seed.csv"
        references_out = reviewer_dir / "references.csv"
        index_out = reviewer_dir / "index.html"

        write_csv(manifest_out, batch_manifest, list(manifest[0].keys()))
        write_csv(labels_out, editor_labels, LABEL_COLUMNS)
        write_csv(references_out, batch_references, list(references[0].keys()))
        (reviewer_dir / "README.md").write_text(
            batch_readme(reviewer_id, len(batch_manifest), len(batch_labels), from_scratch, class_draw_buttons),
            encoding="utf-8",
        )

        builder_args = [
            "--manifest",
            str(manifest_out),
            "--labels",
            str(labels_out),
            "--references",
            str(references_out),
            "--output",
            str(index_out),
        ]
        if class_draw_buttons:
            builder_args.append("--class-draw-buttons")
        builder.main(builder_args)

        class_counts = Counter(row["class_name"] for row in batch_labels)
        batch_rows.append(
            {
                "reviewer_id": reviewer_id,
                "image_count": len(batch_manifest),
                "label_count": len(batch_labels),
                "single_cell": class_counts["single_cell"],
                "early_bud_pair": class_counts["early_bud_pair"],
                "mother_bud_pair": class_counts["mother_bud_pair"],
                "relative_index_html": str(index_out.relative_to(output_dir)),
                "returned_dir": str(returned_dir.relative_to(PROJECT_ROOT)),
            }
        )

        for image_id in image_ids:
            image_labels = labels_by_image.get(image_id, [])
            image_class_counts = Counter(row["class_name"] for row in image_labels)
            assignment_rows.append(
                {
                    "reviewer_id": reviewer_id,
                    "image_id": image_id,
                    "condition": condition_from_image_id(image_id),
                    "label_count": len(image_labels),
                    "single_cell": image_class_counts["single_cell"],
                    "early_bud_pair": image_class_counts["early_bud_pair"],
                    "mother_bud_pair": image_class_counts["mother_bud_pair"],
                }
            )

    write_csv(
        output_dir / "batch_summary.csv",
        batch_rows,
        [
            "reviewer_id",
            "image_count",
            "label_count",
            "single_cell",
            "early_bud_pair",
            "mother_bud_pair",
            "relative_index_html",
            "returned_dir",
        ],
    )
    write_csv(
        output_dir / "batch_assignments.csv",
        assignment_rows,
        [
            "reviewer_id",
            "image_id",
            "condition",
            "label_count",
            "single_cell",
            "early_bud_pair",
            "mother_bud_pair",
        ],
    )
    (output_dir / "index.html").write_text(master_index(batch_rows, from_scratch, class_draw_buttons), encoding="utf-8")
    if from_scratch:
        title = "# Draw-From-Scratch Class-Button Training Audit Batches" if class_draw_buttons else "# Draw-From-Scratch Training Audit Batches"
        root_name = (
            "training_audit_batches_draw_from_scratch_class_buttons"
            if class_draw_buttons
            else "training_audit_batches_draw_from_scratch"
        )
        output_name = (
            "labels_training_reviewed_draw_from_scratch_class_buttons.csv"
            if class_draw_buttons
            else "labels_training_reviewed_draw_from_scratch.csv"
        )
        report_name = (
            "training_audit_merge_summary_draw_from_scratch_class_buttons.json"
            if class_draw_buttons
            else "training_audit_merge_summary_draw_from_scratch.json"
        )
        readme_text = f"""{title}

Five manual review batches are in `Group1` through `Group5`.

Each group should open its own `index.html`, draw all correct boxes manually, then send back:

- `labels_audit_export.csv`
- `labels_reviewed_only_export.csv`

The pseudo-labels are not visible in this version. They were used only to balance the workload across groups.

Returned CSV files can be placed in each group's `returned` folder. Then run:

```powershell
python scripts\\60_merge_v6_training_audit_exports.py --batch-root v6_ml_detection\\review\\{root_name} --output v6_ml_detection\\annotations\\{output_name} --report v6_ml_detection\\reports\\{report_name}
```

The merged reviewed labels become the draw-from-scratch ground-truth CSV.

To score the current rule/SAM pseudo-labels against this independent ground truth, run:

```powershell
python scripts\\61_score_v6_rules_against_ground_truth.py --ground-truth v6_ml_detection\\annotations\\{output_name}
```

This writes the accuracy report under `v6_ml_detection\\reports`.
"""
    else:
        readme_text = """# Training Audit Batches

Five manual review batches are in `Group1` through `Group5`.

Each group should open its own `index.html`, review all assigned images, then send back:

- `labels_audit_export.csv`
- `labels_reviewed_only_export.csv`

Returned CSV files can be placed in each group's `returned` folder. Then run:

```powershell
python scripts\\60_merge_v6_training_audit_exports.py
```

The merged reviewed labels become the training-label CSV.
"""
    (output_dir / "README.md").write_text(
        readme_text,
        encoding="utf-8",
    )

    if from_scratch and class_draw_buttons:
        mode = "draw-from-scratch class-button training audit batches"
    else:
        mode = "draw-from-scratch training audit batches" if from_scratch else "training audit batches"
    print(f"Wrote {reviewer_count} {mode} to {output_dir}")
    for row in batch_rows:
        print(
            f"  {row['reviewer_id']}: {row['image_count']} images, "
            f"{row['label_count']} labels"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviewer-count", type=int, default=5)
    parser.add_argument("--from-scratch", action="store_true", help="Start each editor with no visible seed boxes.")
    parser.add_argument("--class-draw-buttons", action="store_true", help="Use one draw button per class.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if args.from_scratch and output_dir == DEFAULT_OUTPUT:
        output_dir = DEFAULT_CLASS_BUTTON_OUTPUT if args.class_draw_buttons else DEFAULT_FROM_SCRATCH_OUTPUT
    create_batches(
        manifest_path=args.manifest,
        labels_path=args.labels,
        references_path=args.references,
        output_dir=output_dir,
        reviewer_count=args.reviewer_count,
        from_scratch=args.from_scratch,
        class_draw_buttons=args.class_draw_buttons,
    )


if __name__ == "__main__":
    main()
