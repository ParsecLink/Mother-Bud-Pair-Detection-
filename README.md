# Ajit's Project: YeastPair

## Overview

YeastPair is a work-in-progress microscopy analysis project for detecting biological units in budding yeast images. The current pipeline combines GFP nucleus signals and mCherry bud-neck signals to assign one of three classes (`single_cell`, `early_bud_pair`, or `mother_bud_pair`), then uses prompted MobileSAM masks to size the final bounding boxes.

The repository contains research code, portable manifests, the current compact pseudo-label output, and validation summaries. It does not contain the source microscopy TIFFs, generated image collections, model weights, review applications, or a trained detector.

## Current Status

The current source-TIF run contains 311 aligned images and 1,528 pseudo-label boxes. All 1,528 rows pass the implemented schema and coordinate checks, but every row remains `needs_review`. These counts verify file consistency, not biological accuracy.

No final ML model has been trained. Manual audit exports have been returned for Groups 2, 4, and 5, but the aggregate checks mark all three returns as not yet usable for a final accuracy claim. See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for the evidence, limitations, and handoff details.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `scripts/40_...py` through `scripts/63_...py` | Current v6 preparation, pseudo-labeling, review, training-export, and evaluation utilities |
| `v6_ml_detection/annotations/` | Image manifest and annotation schema files |
| `v6_ml_detection/pseudo_labels/` | Current compact pseudo-label CSV and diagnostics |
| `v6_ml_detection/reports/` | Current validation and rule-geometry summaries |
| `v6_ml_detection/references/` | Portable reference manifests; generated reference PNGs are local-only |
| `v6_ml_detection/sam_masks/README.md` | Expected mask naming and format |
| `sam_pipeline/`, `scripts/01_...py` through `scripts/32_...py` | Legacy rule-development code; not the current production workflow |
| `result_pic/RGB/` | Small legacy image sample already in repository history |
| `docs/PROJECT_STATUS.md` | Detailed project status and collaborator handoff |
| `requirements-current.txt` | Direct packages and versions verified in the current local environment |

## Data

The current run expects paired GFP, mCherry, and transmitted-light (`Trans`) TIFF stacks. `scripts/54_prepare_v6_images_from_source_tifs.py` discovers channel triplets below a source root, aligns frames, writes 256 x 256 RGB/reference PNGs, and produces the image and reference manifests.

Source TIFFs and generated PNG collections are intentionally excluded from Git because they are research data and large reproducible artifacts. An authorized collaborator should place local source data under `data/source_tifs/` or pass another location with `--source-root`. Do not commit raw data without confirming its distribution rights.

The repository includes:

- `v6_ml_detection/annotations/image_manifest.csv` with 311 derived-image records;
- `v6_ml_detection/references/source_tif_reference_manifest.csv` with the channel-reference mapping;
- the current 1,528-row pseudo-label CSV and compact diagnostics/reports.

It excludes raw TIFFs, 311 RGB images, 933 generated reference PNGs, SAM instance masks and overlays, embedded-image HTML review apps, audit zip packages, third-party source trees, model weights, and returned collaborator files.

## Methods and Workflow

The current pipeline runs in this order:

1. Convert paired source TIFF stacks into aligned RGB and channel-reference PNGs and create manifests.
2. Detect green nucleus and red/magenta bud-neck candidates.
3. Apply geometric rules to assign `single_cell`, `early_bud_pair`, or `mother_bud_pair`.
4. Prompt MobileSAM using the rule boxes and signal centroids.
5. Use mask boundaries or mask unions for final box size, with a dot-geometry fallback when mask matching fails.
6. Validate the label schema, class names, image references, and coordinates.
7. Generate browser-based review batches for independent manual correction.
8. Merge accepted manual exports, score rule labels against ground truth, export YOLO data, and train/evaluate a detector. The accepted-label and model-training portions of this step are not complete.

Centroids determine biological assignment and prompts; mask boundaries determine box size. Pseudo-labels are bootstrap labels, not ground truth.

## Setup

The current local run was verified with Python 3.12.13 and the package versions in `requirements-current.txt`. MobileSAM was installed from commit `f706ad9c4eb7f219c00d9050e46328518ffb65d2` of `ChaoningZhang/MobileSAM`.

```powershell
py -3.12 -m venv .venv_sam
.\.venv_sam\Scripts\python.exe -m pip install --upgrade pip
.\.venv_sam\Scripts\python.exe -m pip install -r requirements-current.txt
git clone https://github.com/ChaoningZhang/MobileSAM.git external/MobileSAM
git -C external/MobileSAM checkout f706ad9c4eb7f219c00d9050e46328518ffb65d2
.\.venv_sam\Scripts\python.exe -m pip install -e external/MobileSAM
New-Item -ItemType Directory -Force models
Copy-Item external/MobileSAM/weights/mobile_sam.pt models/mobile_sam.pt
```

`ultralytics` is optional for `scripts/43_train_v6_yolo.py` and `scripts/47_predict_v6_yolo.py`. It was not installed in the verified environment, so this repository does not claim a tested `ultralytics` version.

## Running the Project

Prepare images and manifests from authorized local source data:

```powershell
.\.venv_sam\Scripts\python.exe scripts/54_prepare_v6_images_from_source_tifs.py --source-root data/source_tifs
```

Generate prompted masks and current pseudo-labels:

```powershell
.\.venv_sam\Scripts\python.exe scripts/52_run_prompted_mobile_sam_masks.py --output-dir v6_ml_detection/sam_masks/prompted_mobile_sam_from_source_tifs
.\.venv_sam\Scripts\python.exe scripts/51_generate_v6_mask_aware_pseudo_labels.py --cell-mask-dir v6_ml_detection/sam_masks/prompted_mobile_sam_from_source_tifs/instances --output-name labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv --diagnostics-name mobile_sam_from_source_tifs_mask_aware_diagnostics.csv
```

Validate labels and reproduce the compact rule-geometry reports:

```powershell
.\.venv_sam\Scripts\python.exe scripts/48_validate_v6_labels.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv --report v6_ml_detection/reports/mobile_sam_from_source_tifs_mask_aware_validation_summary.json --fail-on-error
.\.venv_sam\Scripts\python.exe scripts/57_report_v6_rule_geometry_stats.py
```

Create manual review material and merge completed returns:

```powershell
.\.venv_sam\Scripts\python.exe scripts/50_make_v6_label_review_app.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv --references v6_ml_detection/references/source_tif_reference_manifest.csv --output v6_ml_detection/review/audit_label_editor/index.html
.\.venv_sam\Scripts\python.exe scripts/59_create_v6_training_audit_batches.py --from-scratch --class-draw-buttons
.\.venv_sam\Scripts\python.exe scripts/60_merge_v6_training_audit_exports.py
```

Evaluate reviewed labels and, only after review is accepted, export/train the detector:

```powershell
.\.venv_sam\Scripts\python.exe scripts/61_score_v6_rules_against_ground_truth.py
.\.venv_sam\Scripts\python.exe scripts/41_export_v6_yolo_dataset.py
.\.venv_sam\Scripts\python.exe scripts/43_train_v6_yolo.py
.\.venv_sam\Scripts\python.exe scripts/47_predict_v6_yolo.py --model v6_ml_detection/runs/yolo_v6_cells/weights/best.pt --hide-text
```

Generate visual review outputs:

```powershell
.\.venv_sam\Scripts\python.exe scripts/45_render_v6_csv_label_overlays.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv --only-with-labels --hide-text
.\.venv_sam\Scripts\python.exe scripts/49_make_v6_review_pack.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv
```

## Current Results

Reproduced and structurally validated:

- 311 images in the current manifest;
- 1,528 valid pseudo-label rows across all 311 images;
- class counts: 869 `mother_bud_pair`, 520 `early_bud_pair`, and 139 `single_cell`;
- 1,527 boxes derived from SAM mask boundaries/unions and one dot fallback, according to the current diagnostics;
- zero errors from `scripts/48_validate_v6_labels.py`.

Preliminary only:

- rule-geometry summaries describe the current candidates but are not ground-truth-optimized thresholds;
- returned manual audit comparisons exist for Groups 2, 4, and 5, but their current aggregate status is not usable for a final accuracy value.

Not yet available:

- accepted ground-truth training labels;
- a trained YOLO model, evaluation metrics, or final figures;
- a completed YeastSAM benchmark.

## Limitations and Known Issues

- Raw data and model weights are required to reproduce the pipeline from the beginning and are intentionally absent.
- All current pseudo-labels still require biological review.
- The three returned group audits need reconciliation before their metrics can be treated as verified.
- `ultralytics` training and inference have not been validated in the current environment.
- The legacy `scripts/01`-`32` code imports a `my_sam_pipeline` package and several modules not present in the current tracked tree; it is retained as historical research context, not a reproducible pipeline.
- The repository has no formal test suite; validation currently relies on syntax checks, CLI smoke checks, and data validators.

## Next Steps

1. Resolve incomplete/invalid returned audits and produce one accepted `labels_training_reviewed.csv`.
2. Score current rules against that independent ground truth and review failure cases by condition/class.
3. Confirm the PI's biological label definition and whether the final target is boxes or masks.
4. Train and evaluate the first detector only after labels are accepted.
5. Benchmark YeastSAM against MobileSAM on the prepared subset before changing the production mask source.

## Collaboration Notes

Start with [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md). Keep raw TIFFs, weights, generated images, masks, HTML review bundles, and returned collaborator exports outside Git. When returning an audit batch, provide both `labels_audit_export.csv` and `labels_reviewed_only_export.csv`; do not treat a batch as complete until the checker reports it usable.
