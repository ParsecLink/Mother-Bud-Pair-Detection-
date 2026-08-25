# Preliminary Manual Audit Results

This directory contains compact returned annotations and rule-comparison tables for Groups 2, 4, and 5. These results are research artifacts, not accepted ground truth or final accuracy measurements.

| Group | Expected images | Covered images | Missing image | Status |
| --- | ---: | ---: | --- | --- |
| Group 2 | 63 | 63 | None | Incomplete (`usable=False`; two `bad_box` rows with `y1=-0.94`) |
| Group 4 | 62 | 61 | `alphafactor_arrest_frame_026` | Incomplete (`usable=False`) |
| Group 5 | 62 | 61 | `training_2_frame_036` | Incomplete (`usable=False`) |

Each group directory includes the full audit export, the reviewed-only labels, per-image metrics, pairwise comparison rows, matched rows, and one compact summary row. Duplicate reviewed exports, local-path JSON files, embedded-image comparison HTML, and original zip packages are intentionally excluded.
