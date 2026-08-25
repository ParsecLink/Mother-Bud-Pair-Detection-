# v6 ML Detection Pipeline

Current run entry point:

```text
..\_CURRENT\README.md
```

Current labels:

```text
pseudo_labels\labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv
```

Current audit editor:

```text
review\audit_label_editor\index.html
```

The current run uses all source TIF folders, prompted MobileSAM masks, and final boxes from mask boundary / mask union. YeastSAM is cloned under `external\YeastSAM` for benchmarking, but it has not replaced MobileSAM yet.

Goal: train an object detector that outputs one class and one box per biological unit:

- `single_cell`
- `mother_bud_pair`
- `early_bud_pair`

This is different from the v5 rule pipeline. v5 is useful for bootstrapping review labels, but v6 should learn the final output directly: class + bounding box.

## Annotation Format

Use `v6_ml_detection/annotations/labels.csv` as one row per object box:

```text
image_id,class_name,x1,y1,x2,y2,source,review_status,notes
```

Coordinates are pixel coordinates in the RGB image:

- `x1,y1`: top-left corner
- `x2,y2`: bottom-right corner
- `class_name`: one of the three names in `classes.txt`
- `review_status`: use `reviewed` for training-ready labels
- `source`: for example `manual`, `v5_pseudo_label`, or `corrected_v5`

## Workflow

1. Build the current image manifest and empty label template:

```powershell
python scripts/40_make_v6_annotation_manifest.py
```

2. Add or review boxes in:

```text
v6_ml_detection/annotations/labels.csv
```

Optional bootstrap: generate rough RGB pseudo-labels for review:

```powershell
python scripts/44_generate_v6_pseudo_labels_from_rgb.py
python scripts/45_render_v6_csv_label_overlays.py --only-with-labels --hide-text
python scripts/48_validate_v6_labels.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_rgb.csv
python scripts/49_make_v6_review_pack.py
python scripts/50_make_v6_label_review_app.py
```

Those rows are written as `needs_review`. Copy corrected rows into `annotations/labels.csv` and change `review_status` to `reviewed` before training.
The interactive review app is written to `v6_ml_detection/review/label_editor/index.html`; use its exported CSV as the reviewed label source.

Mask-aware bootstrap: keep the same green nucleus / magenta bud-neck class rules, but size boxes from matched whole-cell masks when possible:

```powershell
python scripts/51_generate_v6_mask_aware_pseudo_labels.py
python scripts/45_render_v6_csv_label_overlays.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mask_aware.csv --output-dir v6_ml_detection/overlays/mask_aware_boxes_only --only-with-labels --hide-text
python scripts/48_validate_v6_labels.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mask_aware.csv
python scripts/49_make_v6_review_pack.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mask_aware.csv --output-dir v6_ml_detection/review/mask_aware
python scripts/50_make_v6_label_review_app.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mask_aware.csv --output v6_ml_detection/review/audit_label_editor/index.html
```

If CellSAM, YeastSAM, or manual instance masks are available, pass them with `--cell-mask-dir path\to\masks`. Otherwise the script searches for transmitted/DIC/brightfield-like images, then falls back to an RGB-derived transmitted proxy. The `notes` field records `box_source=mask_union` or `box_source=fallback_dot`.

Prompted MobileSAM bootstrap: generate one SAM mask per rule object, then size v6 boxes from those SAM mask boundaries:

```powershell
# one-time local setup from the repository root
python -m venv .venv_sam
git clone https://github.com/ChaoningZhang/MobileSAM.git external/MobileSAM
.venv_sam\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv_sam\Scripts\python.exe -m pip install numpy pillow opencv-python timm
.venv_sam\Scripts\python.exe -m pip install -e external/MobileSAM
copy external\MobileSAM\weights\mobile_sam.pt models\mobile_sam.pt

# run prompted SAM masks and rebuild the audit editor
.venv_sam\Scripts\python.exe scripts\52_run_prompted_mobile_sam_masks.py --output-dir v6_ml_detection\sam_masks\prompted_mobile_sam
python scripts/51_generate_v6_mask_aware_pseudo_labels.py --cell-mask-dir v6_ml_detection/sam_masks/prompted_mobile_sam/instances --output-name labels_pseudo_mobile_sam_mask_aware.csv --diagnostics-name mobile_sam_mask_aware_diagnostics.csv
python scripts/48_validate_v6_labels.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mobile_sam_mask_aware.csv
python scripts/45_render_v6_csv_label_overlays.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mobile_sam_mask_aware.csv --output-dir v6_ml_detection/overlays/mobile_sam_mask_aware_boxes_only --only-with-labels --hide-text
python scripts/50_make_v6_label_review_app.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mobile_sam_mask_aware.csv --output v6_ml_detection/review/audit_label_editor/index.html
```

TIF references in the audit editor: convert TIFF frames to browser-friendly PNGs and pair them to matching image IDs. The editor will show available references side-by-side with the RGB image and draw the same boxes on both views.

```powershell
python scripts/53_extract_tif_references_for_v6_editor.py "_ess dense_GFP_projected.tif"
python scripts/50_make_v6_label_review_app.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_mobile_sam_mask_aware.csv --references v6_ml_detection/references/tif_reference_manifest.csv --output v6_ml_detection/review/audit_label_editor/index.html
```

The expected naming pattern is `<condition>_<channel>.tif` or `<condition>_<channel>_projected.tif`, with frames matched to image IDs like `<condition>_frame_000` after replacing spaces with underscores.

If the v5 box/SAM trial has already been run, import those boxes into the same v6 label schema:

```powershell
python scripts/46_import_v5_boxes_to_v6_labels.py
```

By default this writes `v6_ml_detection/annotations/labels_from_v5.csv` with `review_status=needs_review`.

For a quick experimental export using the unreviewed pseudo-labels:

```powershell
python scripts/41_export_v6_yolo_dataset.py --labels v6_ml_detection/pseudo_labels/labels_pseudo_rgb.csv --include-unreviewed --output-dir v6_ml_detection/datasets/yolo_v6_pseudo_rgb
```

3. Export YOLO-format training data:

```powershell
python scripts/48_validate_v6_labels.py --fail-on-error
python scripts/41_export_v6_yolo_dataset.py
```

4. Train a detector when `ultralytics` is available:

```powershell
python scripts/43_train_v6_yolo.py
```

5. Render prediction or label overlays:

```powershell
python scripts/42_render_v6_yolo_overlays.py --labels-dir v6_ml_detection/datasets/yolo_v6/labels/train --images-dir v6_ml_detection/datasets/yolo_v6/images/train
```

6. Run a trained model over the current manifest and draw final prediction boxes:

```powershell
python scripts/47_predict_v6_yolo.py --model v6_ml_detection/runs/yolo_v6_cells/weights/best.pt --hide-text
```

## Why YOLO Format

YOLO is a good first target because it is simple, fast to iterate, and stores exactly what this task needs: class IDs plus boxes. The dataset can later be converted to COCO if a different detector performs better.
