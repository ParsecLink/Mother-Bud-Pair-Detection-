# Project Status and Handoff

Status date: 2026-08-25

## 1. Project Objective

YeastPair is an image-analysis research project for detecting and classifying budding-yeast biological units from multi-channel microscopy. The intended model output is one class and one bounding box per unit:

- `single_cell`
- `early_bud_pair`
- `mother_bud_pair`

The current repository is a work in progress. It contains a rule-based and MobileSAM-assisted pseudo-label pipeline, not a completed ML model.

## 2. Current Research Question

Can GFP nucleus signals and mCherry bud-neck signals assign biologically meaningful unit classes while a segmentation model supplies reliable whole-cell boundaries for box sizing? The immediate practical question is whether the current pseudo-labels are accurate enough, after correction, to train a detector.

Questions still requiring the project owner's or the PI's decision are listed at the end of this document.

## 3. Component Status

| Component | Status | Main Files | Evidence or Output | Remaining Work |
| --- | --- | --- | --- | --- |
| Source-TIF conversion | Completed for the current local run | `scripts/54_prepare_v6_images_from_source_tifs.py` | 311-row `v6_ml_detection/annotations/image_manifest.csv`; 933-row reference manifest | Reproduction requires excluded source TIFFs |
| Biological rule assignment | Completed as pseudo-label bootstrap | `scripts/44_generate_v6_pseudo_labels_from_rgb.py`, `scripts/51_generate_v6_mask_aware_pseudo_labels.py` | 1,528 current label rows | Validate rules against accepted independent ground truth |
| Prompted MobileSAM masks | Completed for the current local run | `scripts/52_run_prompted_mobile_sam_masks.py` | Diagnostics report 1,527 mask-derived boxes and one fallback | Masks and weight are local-only; visually audit boundary failures |
| Label validation | Completed | `scripts/48_validate_v6_labels.py` | 1,528 valid rows, zero reported errors | Validation checks structure/coordinates, not biological correctness |
| Geometry analysis | Completed descriptively | `scripts/57_report_v6_rule_geometry_stats.py` | `v6_ml_detection/reports/rule_geometry_stats.json` | Select thresholds only after ground-truth comparison |
| Manual review batches | Partially returned | `scripts/59_create_v6_training_audit_batches.py`, `scripts/63_check_returned_group_results.py` | Compact audit artifacts for Groups 2, 4, and 5 under `results/manual_audits/` | Correct incomplete/invalid returns; Groups 1 and 3 are not present |
| Reviewed-label merge | Script ready; accepted output not available | `scripts/60_merge_v6_training_audit_exports.py` | No accepted `labels_training_reviewed.csv` | Complete and approve reviewer returns |
| Rule accuracy evaluation | Script ready; final evaluation not completed | `scripts/56_score_v6_audit_accuracy.py`, `scripts/61_score_v6_rules_against_ground_truth.py` | Existing group comparisons are preliminary and marked unusable | Run after ground truth is accepted |
| YOLO dataset export | Script ready | `scripts/41_export_v6_yolo_dataset.py` | No reviewed training dataset in repository | Export only accepted reviewed labels |
| Detector training/inference | Not completed | `scripts/43_train_v6_yolo.py`, `scripts/47_predict_v6_yolo.py` | No trained weights or verified metrics | Choose/install `ultralytics`, train, validate, document |
| YeastSAM benchmark | Planned, not run | `scripts/55_make_yeastsam_benchmark_subset.py` | 30-image local subset prepared; local weights exist | Run fair side-by-side comparison with MobileSAM |

## 4. Dataset Status, Availability, and Restrictions

The current manifest describes 311 256 x 256 RGB images derived from paired GFP, mCherry, and transmitted-light TIFF stacks. The reference manifest contains 933 rows, corresponding to three channel views per image.

The source TIFFs are outside this repository and were not assessed as redistributable. They must remain in an authorized local location such as `data/source_tifs/`; this path is ignored by Git. Generated RGB/reference PNGs are also excluded because they total many files and can be regenerated from authorized source data.

No human-subject identifiers or PHI were observed in the portable code and compact outputs reviewed for this handoff. The data are nevertheless unpublished research material, so repository visibility must be confirmed before uploading new content.

Local-only material includes:

| Category | Local Evidence | Approximate Size | Version-Control Decision |
| --- | --- | ---: | --- |
| Python environment | `.venv_sam/` | 1.8 GiB | Ignore; recreate from `requirements-current.txt` plus MobileSAM |
| YeastSAM weights/archive | `models/yeastsam/` | 760 MiB | Ignore; model assets, including files above GitHub's 100 MB limit |
| MobileSAM checkpoint | `models/mobile_sam.pt` | 38.8 MiB | Ignore; retrieve from the upstream MobileSAM checkout |
| Current RGB/reference PNGs | `result_pic/from_source_tifs/` | 69.8 MiB | Ignore; derived from excluded source TIFFs |
| SAM masks and overlays | `v6_ml_detection/sam_masks/` | 41.7 MiB | Ignore except format README; derived artifacts |
| Review applications/packages | `v6_ml_detection/review/` | about 1.1 GiB | Ignore; self-contained embedded images, duplicates, and zip packages |
| YeastSAM benchmark subset | `v6_ml_detection/benchmarks/` | 9.8 MiB | Ignore; duplicated images and overlays |
| Third-party repositories | `external/` | 181 MiB | Ignore; clone upstream sources instead of vendoring nested Git repositories |
| Local convenience bundle | `_CURRENT/` | 0.8 MiB | Ignore; duplicates canonical files and contains local launch shortcuts |

## 5. Preprocessing Completed

For the current run, `scripts/54_prepare_v6_images_from_source_tifs.py`:

1. discovers related GFP, mCherry, and Trans TIFF stacks;
2. aligns frames by condition and index;
3. produces RGB images and browser-friendly channel references;
4. assigns stable image IDs and deterministic train/test/validation splits;
5. writes the image and reference manifests.

The compact manifests are present. Full reproduction from raw inputs cannot occur from the repository alone because the TIFFs and generated PNGs are intentionally excluded.

## 6. Modeling and Analysis Completed

The current pseudo-label method detects green nucleus candidates and magenta/red bud-neck candidates. Geometric relationships assign the three biological classes. Candidate centroids are used for assignment and MobileSAM prompting; the MobileSAM mask boundary or union supplies the final box extent.

The current canonical label file contains 1,528 objects:

| Class | Count |
| --- | ---: |
| `mother_bud_pair` | 869 |
| `early_bud_pair` | 520 |
| `single_cell` | 139 |
| **Total** | **1,528** |

All are pseudo-labels with `review_status=needs_review`. These counts are not accuracy measurements.

## 7. Evaluation Completed

Structural validation completed successfully: all 1,528 current rows reference manifest images, use allowed classes, and pass the implemented box checks. The rule-geometry report is descriptive and explicitly states that its candidate thresholds are not ground-truth optimized.

Returned manual comparisons exist for Groups 2, 4, and 5, with compact artifacts included under `results/manual_audits/`. Every group remains `usable=False`, so its precision/recall values must not be cited as final performance. Group 2 covers all 63 assigned images but contains two invalid manual-label boxes with `y1=-0.94`. Group 4 is missing `alphafactor_arrest_frame_026`; Group 5 is missing `training_2_frame_036`.

## 8. Verified Results and Producing Files

| Result or Figure | Producing Script/Notebook | Input | Reproducible Now? | Notes |
| --- | --- | --- | --- | --- |
| 311-image manifest | `scripts/54_prepare_v6_images_from_source_tifs.py` | Excluded GFP/mCherry/Trans TIFFs | No, not from repository alone | Existing manifest is included |
| 933 channel-reference rows | `scripts/54_prepare_v6_images_from_source_tifs.py` | Excluded TIFFs | No, not from repository alone | Existing reference manifest is included |
| 1,528 current pseudo-labels | `scripts/52_run_prompted_mobile_sam_masks.py`, `scripts/51_generate_v6_mask_aware_pseudo_labels.py` | Generated RGB images, MobileSAM checkpoint | No, not from repository alone | Compact current CSV and diagnostics are included |
| Zero structural validation errors | `scripts/48_validate_v6_labels.py` | Included manifest and current pseudo-label CSV | Yes | This does not measure biological accuracy |
| Rule geometry statistics | `scripts/57_report_v6_rule_geometry_stats.py` | Generated RGB images and manifest | No, not from repository alone | Existing compact reports are included |
| Interactive audit editor | `scripts/50_make_v6_label_review_app.py` | Images, references, labels | No, not from repository alone | 94 MiB embedded-image HTML is excluded |
| Five reviewer batches | `scripts/59_create_v6_training_audit_batches.py` | Images, references, labels | No, not from repository alone | Large local HTML/zip packages are excluded |
| Rule-vs-manual comparison | `scripts/62_make_group_result_rule_comparison_html.py`, `scripts/63_check_returned_group_results.py` | Returned reviewer CSVs and local images | No | Current aggregates are preliminary and not usable |
| Trained detector metrics | `scripts/43_train_v6_yolo.py`, `scripts/47_predict_v6_yolo.py` | Accepted labels and optional `ultralytics` environment | No | Training has not occurred |

No notebooks were found in the main project tree. Notebooks under `external/MobileSAM` are third-party materials and are not part of this repository.

## 9. Work in Progress

- independent draw-from-scratch annotation and reconciliation;
- conversion of accepted audits into `labels_training_reviewed.csv`;
- ground-truth rule accuracy scoring and failure analysis;
- choice between box and segmentation-mask prediction targets;
- first detector training and evaluation;
- YeastSAM versus MobileSAM boundary benchmark.

## 10. Known Bugs and Technical Problems

- The current pipeline has no automated unit/integration test suite. CLI help, Python compilation, and data validators are the available checks.
- `scripts/43_train_v6_yolo.py` and `scripts/47_predict_v6_yolo.py` require optional `ultralytics`, which was not present in the verified environment.
- Full current-run reproduction depends on excluded source data, third-party code, and model weights.

## 11. Important Methodological Decisions

| Decision | Current Rationale | Decision Status |
| --- | --- | --- |
| Three class labels | Matches the intended biological-unit output | Active, but PI definition should be reconfirmed |
| Centroids for assignment/prompts | GFP and mCherry signal geometry encodes object relationships | Active |
| Mask boundaries for box extent | Improves on boxes sized only from fluorescent dots | Active |
| MobileSAM as current mask source | Current end-to-end run exists and produced masks | Active baseline |
| Manual review before training | Pseudo-label validation cannot establish biological correctness | Required |
| YOLO as first detector format | Direct class-plus-box representation and existing scripts | Planned; no training evidence yet |
| YeastSAM as benchmark | Yeast-specific segmentation may improve mother-bud boundaries | Planned, not accepted as replacement |

## 12. Approaches Tried but Not Retained as Current

- Early `rules_v2`, `rules_v2_1`, and `rules_v2_3` pipelines explored nucleus, bud-neck, and transmitted-light geometry; they are not part of the current tree.
- v5 rolling-ball and all-object matching scripts were later trials; they are not part of the current tree.
- RGB-only/dot-sized pseudo-label files are preserved locally as older experiments; they are not the current canonical labels.
- The current method replaced dot-only box sizing with MobileSAM mask boundary/union sizing. One current object still used the dot fallback.
- YeastSAM is not a failed approach: its repository and weights exist locally, but no benchmark result was found.

## 13. Reproducibility Status

The compact validation result is reproducible from included files. Source-to-label reproduction is only partially portable because raw data, generated images, masks, model weights, and MobileSAM source are intentionally excluded. `requirements-current.txt` records verified direct package versions, and the README pins the MobileSAM commit used locally.

The existing 311-image/1,528-label run should be treated as preserved evidence. It cannot be claimed as independently reproduced until an authorized collaborator reruns the documented pipeline from the same source TIFFs and compares outputs.

## 14. Immediate Next Steps

1. Correct or re-review the two invalid Group 2 boxes, and resolve the missing Group 4 and Group 5 images.
2. Obtain or complete Groups 1 and 3 if a five-group audit is still the intended protocol.
3. Merge only accepted exports and review the merged class/box distribution.
4. Run `scripts/61_score_v6_rules_against_ground_truth.py` and inspect errors by class and condition.
5. Resolve the PI questions below before changing labels or training targets.
6. Install and record a tested `ultralytics` version, then train a baseline model from accepted labels.
7. Run the prepared YeastSAM subset benchmark and document whether it changes box-boundary quality.

## 15. Longer-Term Possibilities

- predict segmentation masks if boundary quality matters more than box-only detection;
- add condition-stratified evaluation and held-out test governance;
- calibrate biological rule thresholds against independent annotations;
- compare MobileSAM, YeastSAM, and detector-derived boxes on the same accepted test set;
- package a small, legally redistributable sample dataset if the lab approves one.

## 16. Open Decisions

1. Should the final model predict bounding boxes or segmentation masks?
2. Does each label represent a whole cell, a mother-bud unit, or only the nucleus/bud-neck event?
3. For `early_bud_pair`, is one nucleus plus a bud-neck signal sufficient, or is visible bud morphology required?
4. When fluorescence channels and transmitted-light morphology disagree, which evidence defines the label?
5. How many independently reviewed labels are required before training or reporting accuracy?
6. Should all five group audits be completed, and who adjudicates disagreements?
7. Is the current public GitHub repository appropriate for this unpublished work, or must it be made private before this handoff commit is pushed?
