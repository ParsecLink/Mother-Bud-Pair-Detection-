"""Build a visual rule-vs-manual comparison report for one review group."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results" / "Group5"
DEFAULT_GROUP_DIR = V6_DIR / "review" / "training_audit_batches_draw_from_scratch_class_buttons" / "Group5"
DEFAULT_RULE_LABELS = V6_DIR / "pseudo_labels" / "labels_pseudo_mobile_sam_from_source_tifs_mask_aware_geometry_flagged.csv"
CLASS_NAMES = ["single_cell", "early_bud_pair", "mother_bud_pair"]
CLASS_COLORS = {
    "single_cell": "#00d7ff",
    "early_bud_pair": "#ff58ce",
    "mother_bud_pair": "#ffd84a",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latest_export(result_dir: Path, stem: str) -> Path:
    candidates = sorted(result_dir.glob(f"{stem}*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {stem}*.csv found in {result_dir}")
    return candidates[0]


def data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def as_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))


def width(row: dict[str, Any]) -> float:
    x1, _, x2, _ = box(row)
    return x2 - x1


def height(row: dict[str, Any]) -> float:
    _, y1, _, y2 = box(row)
    return y2 - y1


def area(row: dict[str, Any]) -> float:
    return width(row) * height(row)


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


def valid_label(row: dict[str, str], expected_sizes: dict[str, tuple[int, int]], allow_reject: bool = False) -> tuple[bool, str]:
    image_id = row.get("image_id", "")
    if image_id not in expected_sizes:
        return False, "unexpected_image"
    if row.get("class_name", "") not in CLASS_NAMES:
        return False, "bad_class"
    status = row.get("review_status", "")
    if status == "reject" and not allow_reject:
        return False, "reject"
    if status not in {"reviewed", "needs_review", "reject", ""}:
        return False, "bad_status"
    try:
        x1, y1, x2, y2 = (as_float(row, key) for key in ("x1", "y1", "x2", "y2"))
    except ValueError:
        return False, "bad_box_parse"
    max_w, max_h = expected_sizes[image_id]
    if not (0 <= x1 < x2 <= max_w and 0 <= y1 < y2 <= max_h):
        return False, "bad_box"
    return True, ""


def normalize_row(row: dict[str, str], row_id: str, source_kind: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "image_id": row.get("image_id", ""),
        "class_name": row.get("class_name", ""),
        "x1": as_float(row, "x1"),
        "y1": as_float(row, "y1"),
        "x2": as_float(row, "x2"),
        "y2": as_float(row, "y2"),
        "source": row.get("source", ""),
        "review_status": row.get("review_status", ""),
        "notes": row.get("notes", ""),
        "source_kind": source_kind,
        "match_status": "unmatched",
        "matched_id": "",
        "matched_iou": None,
        "localization_status": "unmatched",
        "localization_id": "",
        "localization_iou": None,
    }


def group_by_image(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["image_id"]].append(row)
    return grouped


def greedy_matches(
    rules: list[dict[str, Any]],
    manual: list[dict[str, Any]],
    iou_threshold: float,
    require_same_class: bool,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    candidates: list[tuple[float, int, int]] = []
    for rule_idx, rule in enumerate(rules):
        for manual_idx, gt in enumerate(manual):
            if require_same_class and rule["class_name"] != gt["class_name"]:
                continue
            overlap = iou(box(rule), box(gt))
            if overlap >= iou_threshold:
                candidates.append((overlap, rule_idx, manual_idx))
    candidates.sort(reverse=True, key=lambda item: item[0])

    used_rule: set[int] = set()
    used_manual: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for overlap, rule_idx, manual_idx in candidates:
        if rule_idx in used_rule or manual_idx in used_manual:
            continue
        used_rule.add(rule_idx)
        used_manual.add(manual_idx)
        matches.append((rules[rule_idx], manual[manual_idx], overlap))
    return matches


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 4)


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def box_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "width_px": stats([width(row) for row in rows]),
        "height_px": stats([height(row) for row in rows]),
        "area_px2": stats([area(row) for row in rows]),
    }


def per_class_box_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        class_name: box_stats([row for row in rows if row["class_name"] == class_name])
        for class_name in CLASS_NAMES
    }


def class_count(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["class_name"] for row in rows)
    return {class_name: counts[class_name] for class_name in CLASS_NAMES}


def group_name_from_dir(group_dir: Path) -> str:
    return group_dir.name or "Group"


def report_prefix(group_name: str) -> str:
    return group_name.lower().replace(" ", "_") + "_rule_vs_manual"


def class_metrics(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1(precision, recall)}


def flatten_summary_for_csv(summary: dict[str, Any]) -> dict[str, Any]:
    group_acc = summary["class_aware_accuracy_on_group_assignment"]
    covered_acc = summary["class_aware_accuracy_on_covered_images_only"]
    box_loc = summary["box_localization_ignoring_class"]
    pair_size = summary["localized_pair_box_size_difference_rule_minus_manual"]
    manual_size = summary["box_size_manual_all"]
    rule_size = summary["box_size_rule_all_group"]
    row: dict[str, Any] = {
        "group_name": summary["group_name"],
        "usable": summary["usable"],
        "expected_images": summary["expected_images"],
        "manual_covered_images": summary["manual_covered_images"],
        "missing_manual_images": ";".join(summary["missing_manual_images"]),
        "manual_rows": summary["manual_rows"],
        "audit_rows": summary["audit_rows"],
        "manual_single_cell": summary["manual_class_counts"]["single_cell"],
        "manual_early_bud_pair": summary["manual_class_counts"]["early_bud_pair"],
        "manual_mother_bud_pair": summary["manual_class_counts"]["mother_bud_pair"],
        "rule_single_cell": summary["rule_class_counts_on_covered_images"]["single_cell"],
        "rule_early_bud_pair": summary["rule_class_counts_on_covered_images"]["early_bud_pair"],
        "rule_mother_bud_pair": summary["rule_class_counts_on_covered_images"]["mother_bud_pair"],
        "diff_manual_minus_rule_single_cell": summary["count_difference_manual_minus_rule_covered"]["single_cell"],
        "diff_manual_minus_rule_early_bud_pair": summary["count_difference_manual_minus_rule_covered"]["early_bud_pair"],
        "diff_manual_minus_rule_mother_bud_pair": summary["count_difference_manual_minus_rule_covered"]["mother_bud_pair"],
        "assignment_precision": group_acc["precision"],
        "assignment_recall": group_acc["recall"],
        "assignment_f1": group_acc["f1"],
        "covered_precision": covered_acc["precision"],
        "covered_recall": covered_acc["recall"],
        "covered_f1": covered_acc["f1"],
        "box_only_matched_boxes": box_loc["matched_boxes"],
        "box_only_precision_covered": box_loc["precision_vs_rule_boxes_on_covered_images"],
        "box_only_recall_manual": box_loc["recall_vs_manual_boxes"],
        "box_only_class_accuracy": box_loc["class_accuracy_on_localized_matches"],
        "manual_median_width_px": manual_size["width_px"]["median"],
        "manual_median_height_px": manual_size["height_px"]["median"],
        "manual_median_area_px2": manual_size["area_px2"]["median"],
        "rule_median_width_px": rule_size["width_px"]["median"],
        "rule_median_height_px": rule_size["height_px"]["median"],
        "rule_median_area_px2": rule_size["area_px2"]["median"],
        "matched_median_width_diff_rule_minus_manual_px": pair_size["width_diff_px"]["median"],
        "matched_median_height_diff_rule_minus_manual_px": pair_size["height_diff_px"]["median"],
        "matched_median_area_ratio_rule_over_manual": pair_size["area_ratio_rule_over_manual"]["median"],
    }
    for class_name in CLASS_NAMES:
        metrics = summary["per_class_accuracy"][class_name]
        row[f"{class_name}_precision"] = metrics["precision"]
        row[f"{class_name}_recall"] = metrics["recall"]
        row[f"{class_name}_f1"] = metrics["f1"]
    return row


def read_references(group_dir: Path) -> dict[str, list[dict[str, Any]]]:
    ref_path = group_dir / "references.csv"
    if not ref_path.exists():
        return {}
    refs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(ref_path):
        image_id = row.get("image_id", "")
        ref_rel = row.get("reference_path", "")
        if not image_id or not ref_rel:
            continue
        path = PROJECT_ROOT / ref_rel
        if not path.exists():
            continue
        title = Path(row.get("source_tif", "") or ref_rel).stem
        refs[image_id].append(
            {
                "name": row.get("reference_name", "reference"),
                "title": title,
                "width": int(float(row.get("width", 0) or 0)),
                "height": int(float(row.get("height", 0) or 0)),
                "data_uri": data_uri(path),
            }
        )
    return refs


def build_report(
    result_dir: Path,
    group_dir: Path,
    rule_labels_path: Path,
    manual_csv: Path | None,
    output_html: Path,
    iou_threshold: float,
) -> dict[str, Any]:
    group_name = group_name_from_dir(group_dir)
    prefix = report_prefix(group_name)
    manual_path = manual_csv or latest_export(result_dir, "labels_reviewed_only_export")
    audit_path = latest_export(result_dir, "labels_audit_export")
    reviewed_path = latest_export(result_dir, "labels_reviewed_export")
    manifest = read_csv(group_dir / "manifest.csv")
    expected_ids = [row["image_id"] for row in manifest]
    expected_set = set(expected_ids)
    expected_sizes = {row["image_id"]: (int(row["width"]), int(row["height"])) for row in manifest}

    raw_manual = read_csv(manual_path)
    raw_audit = read_csv(audit_path)
    invalid_manual: list[dict[str, Any]] = []
    invalid_audit: list[dict[str, Any]] = []

    manual: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_manual, start=2):
        ok, reason = valid_label(row, expected_sizes)
        if ok:
            manual.append(normalize_row(row, f"manual_{idx}", "manual"))
        else:
            invalid_manual.append({"line": idx, "reason": reason, "row": row})

    audit_rows_kept = 0
    for idx, row in enumerate(raw_audit, start=2):
        ok, reason = valid_label(row, expected_sizes, allow_reject=True)
        if ok:
            audit_rows_kept += 1
        else:
            invalid_audit.append({"line": idx, "reason": reason, "row": row})

    rule_raw = [row for row in read_csv(rule_labels_path) if row.get("image_id", "") in expected_set]
    rules_all: list[dict[str, Any]] = []
    for idx, row in enumerate(rule_raw, start=1):
        ok, _ = valid_label(row, expected_sizes, allow_reject=True)
        if ok:
            rules_all.append(normalize_row(row, f"rule_{idx}", "rule"))

    manual_image_ids = {row["image_id"] for row in manual}
    missing_images = [image_id for image_id in expected_ids if image_id not in manual_image_ids]
    covered_ids = set(expected_ids) - set(missing_images)
    rules_covered = [row for row in rules_all if row["image_id"] in covered_ids]

    manual_by_image = group_by_image(manual)
    rule_by_image = group_by_image(rules_all)
    per_class = {class_name: Counter() for class_name in CLASS_NAMES}
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    total_tp = total_fp = total_fn = 0
    total_loc_matches = 0
    loc_class_correct = 0
    loc_pair_stats: list[dict[str, Any]] = []
    class_match_iou_values: list[float] = []
    image_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for image_id in expected_ids:
        rules = rule_by_image.get(image_id, [])
        gts = manual_by_image.get(image_id, [])
        class_matches = greedy_matches(rules, gts, iou_threshold, require_same_class=True)
        loc_matches = greedy_matches(rules, gts, iou_threshold, require_same_class=False)

        matched_rule_ids = {rule["id"] for rule, _, _ in class_matches}
        matched_manual_ids = {gt["id"] for _, gt, _ in class_matches}
        for rule, gt, overlap in class_matches:
            rule["match_status"] = "tp"
            rule["matched_id"] = gt["id"]
            rule["matched_iou"] = round(overlap, 4)
            gt["match_status"] = "tp"
            gt["matched_id"] = rule["id"]
            gt["matched_iou"] = round(overlap, 4)
            per_class[rule["class_name"]]["tp"] += 1
            class_match_iou_values.append(overlap)
            match_rows.append(
                {
                    "image_id": image_id,
                    "match_type": "same_class_iou_match",
                    "rule_id": rule["id"],
                    "manual_id": gt["id"],
                    "rule_class": rule["class_name"],
                    "manual_class": gt["class_name"],
                    "iou": f"{overlap:.4f}",
                    "rule_box": ",".join(f"{value:.2f}" for value in box(rule)),
                    "manual_box": ",".join(f"{value:.2f}" for value in box(gt)),
                    "rule_area_px2": f"{area(rule):.2f}",
                    "manual_area_px2": f"{area(gt):.2f}",
                    "area_ratio_rule_over_manual": f"{area(rule) / area(gt):.4f}" if area(gt) else "",
                }
            )

        loc_matched_rule_ids: set[str] = set()
        loc_matched_manual_ids: set[str] = set()
        for rule, gt, overlap in loc_matches:
            loc_matched_rule_ids.add(rule["id"])
            loc_matched_manual_ids.add(gt["id"])
            rule["localization_status"] = "matched"
            rule["localization_id"] = gt["id"]
            rule["localization_iou"] = round(overlap, 4)
            gt["localization_status"] = "matched"
            gt["localization_id"] = rule["id"]
            gt["localization_iou"] = round(overlap, 4)
            confusion[gt["class_name"]][rule["class_name"]] += 1
            total_loc_matches += 1
            if gt["class_name"] == rule["class_name"]:
                loc_class_correct += 1
            loc_pair_stats.append(
                {
                    "rule_width": width(rule),
                    "manual_width": width(gt),
                    "rule_height": height(rule),
                    "manual_height": height(gt),
                    "rule_area": area(rule),
                    "manual_area": area(gt),
                    "area_ratio": area(rule) / area(gt) if area(gt) else None,
                    "width_diff": width(rule) - width(gt),
                    "height_diff": height(rule) - height(gt),
                    "area_diff": area(rule) - area(gt),
                    "same_class": gt["class_name"] == rule["class_name"],
                }
            )
            comparison_rows.append(
                {
                    "image_id": image_id,
                    "comparison_type": "same_class_box_match" if gt["class_name"] == rule["class_name"] else "class_mismatch_box_match",
                    "rule_id": rule["id"],
                    "manual_id": gt["id"],
                    "rule_class": rule["class_name"],
                    "manual_class": gt["class_name"],
                    "iou": f"{overlap:.4f}",
                    "rule_box": ",".join(f"{value:.2f}" for value in box(rule)),
                    "manual_box": ",".join(f"{value:.2f}" for value in box(gt)),
                    "rule_area_px2": f"{area(rule):.2f}",
                    "manual_area_px2": f"{area(gt):.2f}",
                    "area_ratio_rule_over_manual": f"{area(rule) / area(gt):.4f}" if area(gt) else "",
                    "notes": "",
                }
            )

        for rule in rules:
            if rule["id"] not in loc_matched_rule_ids:
                comparison_rows.append(
                    {
                        "image_id": image_id,
                        "comparison_type": "rule_only_or_bad_box",
                        "rule_id": rule["id"],
                        "manual_id": "",
                        "rule_class": rule["class_name"],
                        "manual_class": "",
                        "iou": "",
                        "rule_box": ",".join(f"{value:.2f}" for value in box(rule)),
                        "manual_box": "",
                        "rule_area_px2": f"{area(rule):.2f}",
                        "manual_area_px2": "",
                        "area_ratio_rule_over_manual": "",
                        "notes": "No manual box matched this rule box at IoU threshold.",
                    }
                )
        for gt in gts:
            if gt["id"] not in loc_matched_manual_ids:
                comparison_rows.append(
                    {
                        "image_id": image_id,
                        "comparison_type": "manual_only_missed_by_rule",
                        "rule_id": "",
                        "manual_id": gt["id"],
                        "rule_class": "",
                        "manual_class": gt["class_name"],
                        "iou": "",
                        "rule_box": "",
                        "manual_box": ",".join(f"{value:.2f}" for value in box(gt)),
                        "rule_area_px2": "",
                        "manual_area_px2": f"{area(gt):.2f}",
                        "area_ratio_rule_over_manual": "",
                        "notes": "No rule box matched this manual box at IoU threshold.",
                    }
                )

        tp = len(class_matches)
        fp = len(rules) - tp
        fn = len(gts) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for rule in rules:
            if rule["id"] not in matched_rule_ids:
                per_class[rule["class_name"]]["fp"] += 1
        for gt in gts:
            if gt["id"] not in matched_manual_ids:
                per_class[gt["class_name"]]["fn"] += 1

        image_precision = ratio(tp, tp + fp)
        image_recall = ratio(tp, tp + fn)
        image_rows.append(
            {
                "image_id": image_id,
                "manual_count": len(gts),
                "rule_count": len(rules),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": image_precision,
                "recall": image_recall,
                "f1": f1(image_precision, image_recall),
                "box_only_matches": len(loc_matches),
                "missing_manual_review": image_id in missing_images,
            }
        )

    precision = ratio(total_tp, total_tp + total_fp)
    recall = ratio(total_tp, total_tp + total_fn)
    covered_fp = sum(1 for row in rules_covered if row["match_status"] != "tp")
    covered_precision = ratio(total_tp, total_tp + covered_fp)
    covered_recall = recall
    pair_area_ratios = [row["area_ratio"] for row in loc_pair_stats if row["area_ratio"] is not None]

    summary: dict[str, Any] = {
        "group_name": group_name,
        "result_dir": str(result_dir),
        "manual_csv": str(manual_path),
        "audit_csv": str(audit_path),
        "reviewed_csv": str(reviewed_path),
        "rule_labels": str(rule_labels_path),
        "iou_threshold": iou_threshold,
        "usable": len(invalid_manual) == 0 and len(missing_images) == 0,
        "expected_images": len(expected_ids),
        "manual_covered_images": len(manual_image_ids & expected_set),
        "missing_manual_images": missing_images,
        "manual_rows": len(manual),
        "audit_rows": audit_rows_kept,
        "invalid_manual_rows": invalid_manual[:20],
        "invalid_audit_rows": invalid_audit[:20],
        "manual_class_counts": class_count(manual),
        "rule_class_counts_all_group": class_count(rules_all),
        "rule_class_counts_all_group5": class_count(rules_all),
        "rule_class_counts_on_covered_images": class_count(rules_covered),
        "count_difference_manual_minus_rule_covered": {
            class_name: class_count(manual)[class_name] - class_count(rules_covered)[class_name]
            for class_name in CLASS_NAMES
        },
        "class_aware_accuracy_on_group_assignment": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        },
        "class_aware_accuracy_on_group5_assignment": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        },
        "class_aware_accuracy_on_covered_images_only": {
            "tp": total_tp,
            "fp": covered_fp,
            "fn": total_fn,
            "precision": covered_precision,
            "recall": covered_recall,
            "f1": f1(covered_precision, covered_recall),
        },
        "box_localization_ignoring_class": {
            "matched_boxes": total_loc_matches,
            "precision_vs_rule_boxes": ratio(total_loc_matches, len(rules_all)),
            "precision_vs_rule_boxes_on_covered_images": ratio(total_loc_matches, len(rules_covered)),
            "recall_vs_manual_boxes": ratio(total_loc_matches, len(manual)),
            "class_accuracy_on_localized_matches": ratio(loc_class_correct, total_loc_matches),
        },
        "per_class_accuracy": {
            class_name: class_metrics(
                per_class[class_name]["tp"],
                per_class[class_name]["fp"],
                per_class[class_name]["fn"],
            )
            for class_name in CLASS_NAMES
        },
        "confusion_on_box_localization_matches_manual_rows_rule_columns": {
            manual_class: {rule_class: counts[rule_class] for rule_class in CLASS_NAMES}
            for manual_class, counts in sorted(confusion.items())
        },
        "box_size_manual_all": box_stats(manual),
        "box_size_rule_all_group": box_stats(rules_all),
        "box_size_rule_all_group5": box_stats(rules_all),
        "box_size_manual_by_class": per_class_box_stats(manual),
        "box_size_rule_by_class": per_class_box_stats(rules_all),
        "localized_pair_box_size_difference_rule_minus_manual": {
            "matched_pair_count": len(loc_pair_stats),
            "width_diff_px": stats([row["width_diff"] for row in loc_pair_stats]),
            "height_diff_px": stats([row["height_diff"] for row in loc_pair_stats]),
            "area_diff_px2": stats([row["area_diff"] for row in loc_pair_stats]),
            "area_ratio_rule_over_manual": stats(pair_area_ratios),
        },
        "class_match_iou": stats(class_match_iou_values),
    }

    report_json = result_dir / f"{prefix}_summary.json"
    report_summary_csv = result_dir / f"{prefix}_summary.csv"
    report_matches = result_dir / f"{prefix}_matches.csv"
    report_comparison = result_dir / f"{prefix}_comparison.csv"
    report_by_image = result_dir / f"{prefix}_by_image.csv"
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flattened = flatten_summary_for_csv(summary)
    write_csv(report_summary_csv, [flattened], list(flattened.keys()))
    write_csv(
        report_matches,
        match_rows,
        [
            "image_id",
            "match_type",
            "rule_id",
            "manual_id",
            "rule_class",
            "manual_class",
            "iou",
            "rule_box",
            "manual_box",
            "rule_area_px2",
            "manual_area_px2",
            "area_ratio_rule_over_manual",
        ],
    )
    write_csv(
        report_by_image,
        image_rows,
        ["image_id", "manual_count", "rule_count", "tp", "fp", "fn", "precision", "recall", "f1", "box_only_matches", "missing_manual_review"],
    )
    write_csv(
        report_comparison,
        comparison_rows,
        [
            "image_id",
            "comparison_type",
            "rule_id",
            "manual_id",
            "rule_class",
            "manual_class",
            "iou",
            "rule_box",
            "manual_box",
            "rule_area_px2",
            "manual_area_px2",
            "area_ratio_rule_over_manual",
            "notes",
        ],
    )

    refs_by_image = read_references(group_dir)
    images: list[dict[str, Any]] = []
    rule_grouped = group_by_image(rules_all)
    manual_grouped = group_by_image(manual)
    image_metric_by_id = {row["image_id"]: row for row in image_rows}
    for row in manifest:
        image_id = row["image_id"]
        image_path = PROJECT_ROOT / row["image_path"]
        images.append(
            {
                "image_id": image_id,
                "split": row.get("split", ""),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "data_uri": data_uri(image_path),
                "references": refs_by_image.get(image_id, []),
                "manual": manual_grouped.get(image_id, []),
                "rule": rule_grouped.get(image_id, []),
                "metrics": image_metric_by_id.get(image_id, {}),
            }
        )

    html_payload = json.dumps({"summary": summary, "images": images, "classColors": CLASS_COLORS}, separators=(",", ":"))
    output_html.write_text(render_html(html_payload, summary), encoding="utf-8")
    return summary


def render_html(payload: str, summary: dict[str, Any]) -> str:
    title = f"{summary['group_name']} Rule vs Manual Check"
    usable_text = "Complete" if summary["usable"] else "Needs attention"
    escaped_manual = html.escape(Path(summary["manual_csv"]).name)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Arial, sans-serif; background: #101214; color: #eee; }}
.app {{ display: grid; grid-template-columns: 300px minmax(0, 1fr) 360px; height: 100vh; overflow: hidden; }}
aside, .panel {{ background: #191c1f; border-color: #333940; border-style: solid; }}
aside {{ border-width: 0 1px 0 0; display: grid; grid-template-rows: auto auto minmax(0, 1fr); min-height: 0; }}
.panel {{ border-width: 0 0 0 1px; padding: 12px; overflow: auto; }}
.summary {{ padding: 10px; border-bottom: 1px solid #333940; font-size: 13px; line-height: 1.45; }}
.tools {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 10px; background: #171a1d; border-bottom: 1px solid #333940; }}
button, select, input {{ background: #262b30; color: #eee; border: 1px solid #49515a; border-radius: 4px; padding: 7px; }}
button {{ cursor: pointer; }}
button.active {{ outline: 2px solid #fff; }}
label {{ display: inline-flex; align-items: center; gap: 6px; }}
#imageList {{ overflow-y: auto; min-height: 0; }}
.thumb {{ padding: 8px 10px; border-bottom: 1px solid #2a3036; cursor: pointer; display: grid; gap: 4px; }}
.thumb.active {{ background: #30363d; }}
.thumb.missing {{ border-left: 4px solid #ff6961; }}
.thumb small {{ color: #aab2bb; }}
main {{ min-width: 0; min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); overflow: hidden; }}
.viewer-wrap {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; padding: 12px; overflow: auto; align-items: start; }}
.viewer {{ display: grid; gap: 6px; justify-items: center; min-width: 0; }}
.viewer-title {{ font-size: 12px; color: #b7c0ca; text-align: center; min-height: 14px; overflow-wrap: anywhere; }}
canvas {{ image-rendering: pixelated; background: #000; max-width: 100%; max-height: calc(100vh - 150px); }}
.metric {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; border-bottom: 1px solid #30363d; padding: 6px 0; }}
.warn {{ color: #ffb0aa; }}
.ok {{ color: #b8f7bf; }}
.legend {{ display: grid; gap: 6px; margin: 8px 0 14px; }}
.swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; border: 1px solid #fff8; vertical-align: -1px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin: 8px 0 16px; }}
td, th {{ border: 1px solid #333940; padding: 5px; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
</style>
</head>
<body>
<div class="app">
  <aside>
    <div class="summary">
      <b>{title}</b><br>
      Manual CSV: {escaped_manual}<br>
      Status: <span class="{('ok' if summary['usable'] else 'warn')}">{usable_text}</span><br>
      Images covered: {summary['manual_covered_images']} / {summary['expected_images']}<br>
      Manual boxes: {summary['manual_rows']} | Rule boxes: {sum(summary['rule_class_counts_all_group'].values())}
    </div>
    <div class="summary">
      <input id="search" type="search" placeholder="Search image/class/status" style="width:100%;">
    </div>
    <div id="imageList"></div>
  </aside>
  <main>
    <div class="tools">
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
      <label><input id="manualToggle" type="checkbox" checked> Manual solid</label>
      <label><input id="ruleToggle" type="checkbox" checked> Rule dashed</label>
      <span>Right view</span>
      <select id="backgroundSelect"></select>
      <span id="imageTitle"></span>
    </div>
    <div class="viewer-wrap">
      <div class="viewer">
        <div class="viewer-title" id="mainTitle">RGB</div>
        <canvas id="canvas" width="256" height="256"></canvas>
      </div>
      <div class="viewer">
        <div class="viewer-title" id="refTitle">Reference</div>
        <canvas id="refCanvas" width="256" height="256"></canvas>
      </div>
    </div>
  </main>
  <section class="panel">
    <h3>Summary</h3>
    <div class="legend">
      <div><span class="swatch" style="background:#00d7ff"></span>single_cell</div>
      <div><span class="swatch" style="background:#ff58ce"></span>early_bud_pair</div>
      <div><span class="swatch" style="background:#ffd84a"></span>mother_bud_pair</div>
      <div>Manual = solid line. Rule = dashed line.</div>
    </div>
    <div id="summaryMetrics"></div>
    <h3>Current Image</h3>
    <div id="imageMetrics"></div>
    <h3>Boxes</h3>
    <div id="boxList"></div>
  </section>
</div>
<script>
const DATA = {payload};
let current = 0;
let query = '';
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const refCanvas = document.getElementById('refCanvas');
const refCtx = refCanvas.getContext('2d');
const cache = new Map();

function loadImage(src) {{
  if (cache.has(src)) return Promise.resolve(cache.get(src));
  return new Promise(resolve => {{
    const img = new Image();
    img.onload = () => {{ cache.set(src, img); resolve(img); }};
    img.src = src;
  }});
}}
function colorFor(cls) {{ return DATA.classColors[cls] || '#fff'; }}
function filteredIndices() {{
  const q = query.trim().toLowerCase();
  if (!q) return DATA.images.map((_, i) => i);
  return DATA.images.map((img, i) => [img, i]).filter(([img]) => {{
    const text = [img.image_id, img.split, ...img.manual.map(b => b.class_name), ...img.rule.map(b => b.class_name), img.metrics.missing_manual_review ? 'missing' : ''].join(' ').toLowerCase();
    return text.includes(q);
  }}).map(([, i]) => i);
}}
function go(index) {{
  current = Math.max(0, Math.min(DATA.images.length - 1, index));
  render();
}}
function drawRect(targetCtx, box, mode, scaleX = 1, scaleY = 1) {{
  targetCtx.save();
  targetCtx.strokeStyle = colorFor(box.class_name);
  targetCtx.lineWidth = mode === 'manual' ? 3 : 2;
  targetCtx.setLineDash(mode === 'rule' ? [7, 4] : []);
  targetCtx.strokeRect(box.x1 * scaleX, box.y1 * scaleY, (box.x2 - box.x1) * scaleX, (box.y2 - box.y1) * scaleY);
  targetCtx.restore();
}}
function drawLabel(targetCtx, box, mode, scaleX = 1, scaleY = 1) {{
  return;
}}
async function renderCanvases() {{
  const img = DATA.images[current];
  const rgb = await loadImage(img.data_uri);
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(rgb, 0, 0, canvas.width, canvas.height);
  const scaleX = 1;
  const scaleY = 1;
  if (document.getElementById('manualToggle').checked) {{
    for (const box of img.manual) drawRect(ctx, box, 'manual', scaleX, scaleY);
  }}
  if (document.getElementById('ruleToggle').checked) {{
    for (const box of img.rule) drawRect(ctx, box, 'rule', scaleX, scaleY);
  }}
  document.getElementById('mainTitle').textContent = 'RGB fixed';

  const bgValue = document.getElementById('backgroundSelect').value || 'Trans';
  let ref = {{name: 'RGB', title: 'RGB', width: img.width, height: img.height, data_uri: img.data_uri}};
  if (bgValue !== 'RGB') ref = img.references.find(r => r.name === bgValue) || img.references.find(r => r.name === 'Trans') || img.references[0] || ref;
  const refImg = await loadImage(ref.data_uri);
  refCanvas.width = ref.width || img.width;
  refCanvas.height = ref.height || img.height;
  refCtx.drawImage(refImg, 0, 0, refCanvas.width, refCanvas.height);
  const rsx = refCanvas.width / img.width;
  const rsy = refCanvas.height / img.height;
  if (document.getElementById('manualToggle').checked) for (const box of img.manual) drawRect(refCtx, box, 'manual', rsx, rsy);
  if (document.getElementById('ruleToggle').checked) for (const box of img.rule) drawRect(refCtx, box, 'rule', rsx, rsy);
  document.getElementById('refTitle').textContent = ref.title || ref.name || 'Reference';
}}
function metricRow(name, value, cls = '') {{
  return `<div class="metric"><span>${{name}}</span><b class="${{cls}}">${{value ?? ''}}</b></div>`;
}}
function tableFromMetrics(title, rows) {{
  return `<h4>${{title}}</h4><table><thead><tr><th>Class</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${{r.className}}</td><td>${{r.tp}}</td><td>${{r.fp}}</td><td>${{r.fn}}</td><td>${{r.precision ?? ''}}</td><td>${{r.recall ?? ''}}</td><td>${{r.f1 ?? ''}}</td></tr>`).join('') +
    '</tbody></table>';
}}
function renderSummaryMetrics() {{
  const s = DATA.summary;
  const acc = s.class_aware_accuracy_on_group_assignment;
  const loc = s.box_localization_ignoring_class;
  const rows = Object.entries(s.per_class_accuracy).map(([className, m]) => ({{className, ...m}}));
  document.getElementById('summaryMetrics').innerHTML =
    metricRow('Complete/usable', s.usable ? 'yes' : 'no', s.usable ? 'ok' : 'warn') +
    metricRow('Missing images', s.missing_manual_images.length ? s.missing_manual_images.join(', ') : 'none', s.missing_manual_images.length ? 'warn' : 'ok') +
    metricRow('Class-aware precision', acc.precision) +
    metricRow('Class-aware recall', acc.recall) +
    metricRow('Class-aware F1', acc.f1) +
    metricRow('Box-only matched boxes', loc.matched_boxes) +
    metricRow('Class accuracy on localized boxes', loc.class_accuracy_on_localized_matches) +
    tableFromMetrics('Per-Class Rule Accuracy', rows);
}}
function renderImageList() {{
  const indices = filteredIndices();
  document.getElementById('imageList').innerHTML = indices.map(i => {{
    const img = DATA.images[i];
    const m = img.metrics;
    return `<div class="thumb ${{i === current ? 'active' : ''}} ${{m.missing_manual_review ? 'missing' : ''}}" onclick="go(${{i}})">
      <b>${{img.image_id}}</b>
      <small>${{img.split}} | manual ${{m.manual_count || 0}} | rule ${{m.rule_count || 0}} | TP ${{m.tp || 0}} FP ${{m.fp || 0}} FN ${{m.fn || 0}}</small>
    </div>`;
  }}).join('');
}}
function renderImageMetrics() {{
  const img = DATA.images[current];
  const m = img.metrics;
  const cls = m.missing_manual_review ? 'warn' : '';
  document.getElementById('imageTitle').textContent = `${{img.image_id}} (${{current + 1}}/${{DATA.images.length}})`;
  document.getElementById('imageMetrics').innerHTML =
    metricRow('Missing manual review', m.missing_manual_review ? 'yes' : 'no', cls) +
    metricRow('Manual boxes', m.manual_count) +
    metricRow('Rule boxes', m.rule_count) +
    metricRow('TP / FP / FN', `${{m.tp}} / ${{m.fp}} / ${{m.fn}}`) +
    metricRow('Precision', m.precision) +
    metricRow('Recall', m.recall) +
    metricRow('F1', m.f1) +
    metricRow('Box-only matches', m.box_only_matches);
  const boxes = [
    ...img.manual.map(b => ({{kind: 'Manual', ...b}})),
    ...img.rule.map(b => ({{kind: 'Rule', ...b}})),
  ];
  document.getElementById('boxList').innerHTML = boxes.map(b =>
    `<div class="metric"><span><span class="swatch" style="background:${{colorFor(b.class_name)}}"></span>${{b.kind}} ${{b.class_name}}<br><small>${{Math.round(b.x1)}},${{Math.round(b.y1)}}-${{Math.round(b.x2)}},${{Math.round(b.y2)}} | match ${{b.match_status}} | loc ${{b.localization_status}}</small></span><b>${{b.matched_iou ?? b.localization_iou ?? ''}}</b></div>`
  ).join('');
}}
function renderBackgroundOptions() {{
  const img = DATA.images[current];
  const select = document.getElementById('backgroundSelect');
  const old = select.value;
  const options = ['RGB', ...img.references.map(ref => ref.name)];
  select.innerHTML = options.map(name => `<option>${{name}}</option>`).join('');
  const preferred = options.includes('Trans') ? 'Trans' : options[0];
  select.value = options.includes(old) ? old : preferred;
}}
async function render() {{
  renderBackgroundOptions();
  renderImageList();
  renderSummaryMetrics();
  renderImageMetrics();
  await renderCanvases();
}}
document.getElementById('prevBtn').onclick = () => go(current - 1);
document.getElementById('nextBtn').onclick = () => go(current + 1);
document.getElementById('search').oninput = e => {{ query = e.target.value; renderImageList(); }};
for (const id of ['manualToggle', 'ruleToggle', 'backgroundSelect']) {{
  document.getElementById(id).onchange = () => renderCanvases();
}}
render();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--rule-labels", type=Path, default=DEFAULT_RULE_LABELS)
    parser.add_argument("--manual-csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_name = group_name_from_dir(args.group_dir)
    output = args.output or (args.result_dir / f"{report_prefix(group_name)}_check.html")
    summary = build_report(
        result_dir=args.result_dir,
        group_dir=args.group_dir,
        rule_labels_path=args.rule_labels,
        manual_csv=args.manual_csv,
        output_html=output,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote HTML report to {output}")


if __name__ == "__main__":
    main()
