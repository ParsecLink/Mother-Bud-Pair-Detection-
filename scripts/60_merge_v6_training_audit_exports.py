"""Merge returned manual audit CSV files into a training-label CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_BATCH_ROOT = V6_DIR / "review" / "training_audit_batches"
DEFAULT_OUTPUT = V6_DIR / "annotations" / "labels_training_reviewed.csv"
DEFAULT_REPORT = V6_DIR / "reports" / "training_audit_merge_summary.json"
LABEL_COLUMNS = ["image_id", "class_name", "x1", "y1", "x2", "y2", "source", "review_status", "notes"]
CLASS_NAMES = {"single_cell", "mother_bud_pair", "early_bud_pair"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def discover_exports(batch_root: Path) -> list[Path]:
    patterns = [
        "Group*/returned/*audit*.csv",
        "Group*/returned/*reviewed*.csv",
        "student_*/returned/*audit*.csv",
        "student_*/returned/*reviewed*.csv",
        "returned_exports/*.csv",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(batch_root.glob(pattern))
    return sorted(set(paths))


def reviewer_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("Group") or part.startswith("student_"):
            return part
    return path.stem


def is_training_row(row: dict[str, str]) -> bool:
    if row.get("class_name", "") not in CLASS_NAMES:
        return False
    status = row.get("review_status", "")
    if status == "reject":
        return False
    if "label_id" in row:
        return status == "reviewed"
    return status in {"reviewed", ""}


def clean_row(row: dict[str, str], source_file: Path) -> dict[str, str]:
    cleaned = {key: row.get(key, "") for key in LABEL_COLUMNS}
    cleaned["review_status"] = "reviewed"
    if not cleaned["source"]:
        cleaned["source"] = "manual_training_audit"
    suffix = f"merged_from={reviewer_from_path(source_file)}:{source_file.name}"
    cleaned["notes"] = f"{cleaned['notes']}; {suffix}" if cleaned["notes"] else suffix
    for key in ("x1", "y1", "x2", "y2"):
        cleaned[key] = f"{float(cleaned[key]):.2f}"
    return cleaned


def merge_exports(batch_root: Path, output_path: Path, report_path: Path, allow_empty: bool) -> dict[str, object]:
    exports = discover_exports(batch_root)
    if not exports and not allow_empty:
        raise FileNotFoundError(
            f"No returned CSV files found under {batch_root}. "
            "Put each group's exported CSV in GroupX/returned first, "
            "or rerun with --allow-empty for a dry run."
        )
    merged: list[dict[str, str]] = []
    seen_label_keys: set[tuple[str, str, str, str, str, str]] = set()
    skipped = 0
    duplicate_rows = 0
    rows_by_file: dict[str, dict[str, int]] = {}

    for path in exports:
        rows = read_csv(path)
        kept_for_file = 0
        skipped_for_file = 0
        duplicates_for_file = 0
        for row in rows:
            if is_training_row(row):
                cleaned = clean_row(row, path)
                label_key = (
                    cleaned["image_id"],
                    cleaned["class_name"],
                    cleaned["x1"],
                    cleaned["y1"],
                    cleaned["x2"],
                    cleaned["y2"],
                )
                if label_key in seen_label_keys:
                    duplicate_rows += 1
                    duplicates_for_file += 1
                    continue
                seen_label_keys.add(label_key)
                merged.append(cleaned)
                kept_for_file += 1
            else:
                skipped += 1
                skipped_for_file += 1
        rows_by_file[str(path.relative_to(PROJECT_ROOT))] = {
            "input_rows": len(rows),
            "kept_reviewed_rows": kept_for_file,
            "skipped_rows": skipped_for_file,
            "duplicate_rows": duplicates_for_file,
        }

    write_csv(output_path, merged)
    class_counts = Counter(row["class_name"] for row in merged)
    image_count = len({row["image_id"] for row in merged})
    report = {
        "batch_root": str(batch_root),
        "output_path": str(output_path),
        "export_file_count": len(exports),
        "merged_reviewed_labels": len(merged),
        "skipped_rows": skipped,
        "duplicate_rows": duplicate_rows,
        "image_count": image_count,
        "class_counts": dict(class_counts),
        "files": rows_by_file,
        "note": "Rows are included only when review_status is reviewed. Rejected and needs_review rows are excluded.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = merge_exports(args.batch_root, args.output, args.report, allow_empty=args.allow_empty)
    print(json.dumps(report, indent=2))
    print(f"Wrote merged training labels to {args.output}")
    print(f"Wrote merge report to {args.report}")


if __name__ == "__main__":
    main()
