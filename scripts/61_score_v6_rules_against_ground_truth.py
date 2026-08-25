"""Score v6 pseudo-label rules against independent manual ground truth.

Use this for draw-from-scratch review exports. Those exports do not contain the
original pseudo-label IDs, so accuracy must be measured by matching boxes by IoU.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_PREDICTIONS = V6_DIR / "pseudo_labels" / "labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv"
DEFAULT_GROUND_TRUTH = V6_DIR / "annotations" / "labels_training_reviewed.csv"
DEFAULT_REPORT = V6_DIR / "reports" / "rules_vs_scratch_ground_truth_summary.json"
DEFAULT_MATCHES = V6_DIR / "reports" / "rules_vs_scratch_ground_truth_matches.csv"
CLASS_NAMES = {"single_cell", "early_bud_pair", "mother_bud_pair"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (as_float(row, "x1"), as_float(row, "y1"), as_float(row, "x2"), as_float(row, "y2"))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def valid_label(row: dict[str, str], allow_needs_review: bool = False) -> bool:
    if row.get("class_name", "") not in CLASS_NAMES:
        return False
    if row.get("review_status", "") == "reject":
        return False
    if not allow_needs_review and row.get("review_status", "") == "needs_review":
        return False
    try:
        x1, y1, x2, y2 = box(row)
    except ValueError:
        return False
    return x2 > x1 and y2 > y1


def add_row_ids(rows: list[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    with_ids: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        updated = dict(row)
        updated["_row_id"] = f"{prefix}{idx}"
        with_ids.append(updated)
    return with_ids


def group_by_image(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("image_id", "")].append(row)
    return grouped


def greedy_matches(
    predictions: list[dict[str, str]],
    ground_truth: list[dict[str, str]],
    iou_threshold: float,
    require_same_class: bool,
) -> list[tuple[dict[str, str], dict[str, str], float]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truth):
            if require_same_class and pred.get("class_name") != gt.get("class_name"):
                continue
            overlap = iou(box(pred), box(gt))
            if overlap >= iou_threshold:
                candidates.append((overlap, pred_idx, gt_idx))
    candidates.sort(reverse=True, key=lambda item: item[0])

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[dict[str, str], dict[str, str], float]] = []
    for overlap, pred_idx, gt_idx in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((predictions[pred_idx], ground_truth[gt_idx], overlap))
    return matches


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 4)


def class_metrics(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
    }


def score(
    predictions_path: Path,
    ground_truth_path: Path,
    report_path: Path,
    matches_path: Path,
    iou_threshold: float,
) -> dict[str, Any]:
    predictions = add_row_ids(
        [row for row in read_csv(predictions_path) if valid_label(row, allow_needs_review=True)],
        "pred_",
    )
    ground_truth = add_row_ids(
        [row for row in read_csv(ground_truth_path) if valid_label(row, allow_needs_review=False)],
        "gt_",
    )

    pred_by_image = group_by_image(predictions)
    gt_by_image = group_by_image(ground_truth)
    image_ids = sorted(set(pred_by_image) | set(gt_by_image))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    localization_matches_total = 0
    localization_class_correct = 0
    iou_values: list[float] = []
    per_class = {class_name: Counter() for class_name in sorted(CLASS_NAMES)}
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    image_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []

    for image_id in image_ids:
        preds = pred_by_image.get(image_id, [])
        gts = gt_by_image.get(image_id, [])
        class_aware_matches = greedy_matches(preds, gts, iou_threshold, require_same_class=True)
        localization_matches = greedy_matches(preds, gts, iou_threshold, require_same_class=False)

        matched_pred_ids = {pred["_row_id"] for pred, _, _ in class_aware_matches}
        matched_gt_ids = {gt["_row_id"] for _, gt, _ in class_aware_matches}
        tp = len(class_aware_matches)
        fp = len(preds) - tp
        fn = len(gts) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for pred, gt, overlap in class_aware_matches:
            class_name = pred["class_name"]
            per_class[class_name]["tp"] += 1
            iou_values.append(overlap)
            match_rows.append(
                {
                    "match_type": "tp_class_and_box",
                    "image_id": image_id,
                    "pred_row_id": pred["_row_id"],
                    "gt_row_id": gt["_row_id"],
                    "pred_class": pred["class_name"],
                    "gt_class": gt["class_name"],
                    "iou": f"{overlap:.4f}",
                    "pred_box": ",".join(f"{value:.2f}" for value in box(pred)),
                    "gt_box": ",".join(f"{value:.2f}" for value in box(gt)),
                }
            )

        for pred in preds:
            if pred["_row_id"] not in matched_pred_ids:
                per_class[pred["class_name"]]["fp"] += 1
        for gt in gts:
            if gt["_row_id"] not in matched_gt_ids:
                per_class[gt["class_name"]]["fn"] += 1

        localization_matches_total += len(localization_matches)
        for pred, gt, _ in localization_matches:
            confusion[gt["class_name"]][pred["class_name"]] += 1
            if pred["class_name"] == gt["class_name"]:
                localization_class_correct += 1

        image_precision = ratio(tp, tp + fp)
        image_recall = ratio(tp, tp + fn)
        image_rows.append(
            {
                "image_id": image_id,
                "prediction_count": len(preds),
                "ground_truth_count": len(gts),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": image_precision,
                "recall": image_recall,
                "f1": f1(image_precision, image_recall),
                "box_only_matches": len(localization_matches),
            }
        )

    precision = ratio(total_tp, total_tp + total_fp)
    recall = ratio(total_tp, total_tp + total_fn)
    iou_sorted = sorted(iou_values)

    def percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        index = round((len(values) - 1) * p)
        return round(values[index], 4)

    report: dict[str, Any] = {
        "predictions_path": str(predictions_path),
        "ground_truth_path": str(ground_truth_path),
        "iou_threshold": iou_threshold,
        "image_count": len(image_ids),
        "prediction_labels": len(predictions),
        "ground_truth_labels": len(ground_truth),
        "class_aware_detection": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        },
        "box_localization_ignoring_class": {
            "matched_boxes": localization_matches_total,
            "matched_box_recall_vs_ground_truth": ratio(localization_matches_total, len(ground_truth)),
            "matched_box_precision_vs_predictions": ratio(localization_matches_total, len(predictions)),
            "class_accuracy_on_box_matches": ratio(localization_class_correct, localization_matches_total),
        },
        "matched_iou": {
            "count": len(iou_values),
            "median": percentile(iou_sorted, 0.5),
            "p10": percentile(iou_sorted, 0.1),
            "p90": percentile(iou_sorted, 0.9),
        },
        "per_class": {
            class_name: class_metrics(
                counts["tp"],
                counts["fp"],
                counts["fn"],
            )
            for class_name, counts in sorted(per_class.items())
        },
        "confusion_on_box_matches": {
            gt_class: dict(pred_counts)
            for gt_class, pred_counts in sorted(confusion.items())
        },
        "note": (
            "Use this report for draw-from-scratch review data. Class-aware TP requires "
            "same class and IoU >= threshold. Box-localization metrics ignore class."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(
        matches_path,
        match_rows,
        ["match_type", "image_id", "pred_row_id", "gt_row_id", "pred_class", "gt_class", "iou", "pred_box", "gt_box"],
    )
    write_csv(
        matches_path.with_name(matches_path.stem + "_by_image.csv"),
        image_rows,
        ["image_id", "prediction_count", "ground_truth_count", "tp", "fp", "fn", "precision", "recall", "f1", "box_only_matches"],
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = score(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        report_path=args.report,
        matches_path=args.matches,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote report to {args.report}")
    print(f"Wrote matched boxes to {args.matches}")
    print(f"Wrote per-image summary to {args.matches.with_name(args.matches.stem + '_by_image.csv')}")


if __name__ == "__main__":
    main()
