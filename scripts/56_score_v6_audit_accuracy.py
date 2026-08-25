"""Score pseudo-label accuracy from an audit editor export.

The audit export should come from the audit editor's "Export Audit CSV" button.
It includes rejected labels and label_id values, which makes it possible to
compare the original pseudo labels against human-reviewed outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = (
    PROJECT_ROOT
    / "v6_ml_detection"
    / "pseudo_labels"
    / "labels_pseudo_mobile_sam_from_source_tifs_mask_aware.csv"
)
DEFAULT_REPORT = PROJECT_ROOT / "v6_ml_detection" / "reports" / "audit_accuracy_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    if union <= 0:
        return 0.0
    return intersection / union


def add_label_ids(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    with_ids: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        updated = dict(row)
        updated["label_id"] = str(idx)
        with_ids.append(updated)
    return with_ids


def is_manual_label(row: dict[str, str], original_count: int) -> bool:
    label_id = row.get("label_id", "")
    if label_id.isdigit() and int(label_id) > original_count:
        return True
    return row.get("source", "") == "manual_review_app"


def score(original_path: Path, audit_path: Path, report_path: Path, iou_threshold: float) -> dict[str, object]:
    original = add_label_ids(read_csv(original_path))
    audit = read_csv(audit_path)
    original_by_id = {row["label_id"]: row for row in original}
    original_count = len(original)

    audited_original = []
    manual_reviewed = []
    unreviewed_rows = 0

    for row in audit:
        status = row.get("review_status", "")
        if status == "needs_review":
            unreviewed_rows += 1
            continue
        if is_manual_label(row, original_count):
            if status == "reviewed":
                manual_reviewed.append(row)
            continue
        original_row = original_by_id.get(str(row.get("label_id", "")))
        if original_row:
            audited_original.append((original_row, row))

    reviewed_original = [(orig, aud) for orig, aud in audited_original if aud.get("review_status") == "reviewed"]
    rejected_original = [(orig, aud) for orig, aud in audited_original if aud.get("review_status") == "reject"]

    correct = 0
    class_changed = 0
    box_changed = 0
    reviewed_same_class = 0
    reviewed_box_ok = 0
    per_class: dict[str, Counter[str]] = defaultdict(Counter)

    for orig, aud in audited_original:
        original_class = orig.get("class_name", "")
        audit_class = aud.get("class_name", "")
        status = aud.get("review_status", "")
        box_iou = iou(box(orig), box(aud))
        class_ok = original_class == audit_class
        box_ok = box_iou >= iou_threshold

        per_class[original_class]["audited"] += 1
        if status == "reject":
            per_class[original_class]["rejected"] += 1
            continue

        if class_ok:
            reviewed_same_class += 1
        else:
            class_changed += 1
            per_class[original_class]["class_changed"] += 1

        if box_ok:
            reviewed_box_ok += 1
        else:
            box_changed += 1
            per_class[original_class]["box_changed"] += 1

        if class_ok and box_ok:
            correct += 1
            per_class[original_class]["correct"] += 1

    audited_count = len(audited_original)
    reviewed_count = len(reviewed_original)
    rejected_count = len(rejected_original)
    manual_count = len(manual_reviewed)

    def ratio(numerator: int, denominator: int) -> float | None:
        if denominator == 0:
            return None
        return round(numerator / denominator, 4)

    summary: dict[str, object] = {
        "original_labels_path": str(original_path),
        "audit_export_path": str(audit_path),
        "iou_threshold": iou_threshold,
        "original_total_labels": original_count,
        "audit_total_rows": len(audit),
        "unreviewed_rows_in_audit_export": unreviewed_rows,
        "audited_original_labels": audited_count,
        "reviewed_original_labels": reviewed_count,
        "rejected_original_labels": rejected_count,
        "manual_reviewed_labels_added": manual_count,
        "correct_original_labels": correct,
        "pseudo_label_accuracy_on_audited_originals": ratio(correct, audited_count),
        "class_changed_original_labels": class_changed,
        "class_accuracy_on_reviewed_originals": ratio(reviewed_same_class, reviewed_count),
        "box_changed_original_labels": box_changed,
        "box_accuracy_on_reviewed_originals": ratio(reviewed_box_ok, reviewed_count),
        "reject_rate_on_audited_originals": ratio(rejected_count, audited_count),
        "manual_add_rate_vs_reviewed_ground_truth": ratio(manual_count, reviewed_count + manual_count),
        "per_original_class": {
            class_name: {
                "audited": counts["audited"],
                "correct": counts["correct"],
                "rejected": counts["rejected"],
                "class_changed": counts["class_changed"],
                "box_changed": counts["box_changed"],
                "accuracy": ratio(counts["correct"], counts["audited"]),
            }
            for class_name, counts in sorted(per_class.items())
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="CSV from Export Audit CSV")
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = score(
        original_path=args.original,
        audit_path=args.audit,
        report_path=args.report,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
