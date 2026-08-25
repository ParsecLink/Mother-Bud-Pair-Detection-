"""Check returned review CSV folders and compare them against v6 rule labels."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_BATCH_ROOT = V6_DIR / "review" / "training_audit_batches_draw_from_scratch_class_buttons"
DEFAULT_RULE_LABELS = V6_DIR / "pseudo_labels" / "labels_pseudo_mobile_sam_from_source_tifs_mask_aware_geometry_flagged.csv"
DEFAULT_AGGREGATE = V6_DIR / "reports" / "returned_group_rule_comparison_summary.csv"


def load_checker_module():
    path = PROJECT_ROOT / "scripts" / "62_make_group_result_rule_comparison_html.py"
    spec = importlib.util.spec_from_file_location("group_result_checker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_group_number(path: Path) -> int | None:
    text = path.name.lower()
    patterns = [
        r"group[_\-\s]*(\d+)",
        r"student[_\-\s]*(\d+)",
        r"g[_\-\s]*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def has_returned_csvs(path: Path) -> bool:
    return any(path.glob("labels_reviewed_only_export*.csv")) or any(path.glob("labels_audit_export*.csv"))


def discover_result_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        path for path in results_root.iterdir()
        if path.is_dir() and infer_group_number(path) is not None and has_returned_csvs(path)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, action="append", help="Returned CSV folder. Can be passed multiple times.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Root folder to scan for Group*_Result folders.")
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--rule-labels", type=Path, default=DEFAULT_RULE_LABELS)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checker = load_checker_module()
    result_dirs = args.result_dir if args.result_dir else discover_result_dirs(args.results_root)
    if not result_dirs:
        raise FileNotFoundError(
            f"No returned result folders found. Pass --result-dir or place Group*_Result folders under {args.results_root}."
        )

    aggregate_rows: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        group_number = infer_group_number(result_dir)
        if group_number is None:
            raise ValueError(f"Could not infer group number from {result_dir}")
        group_dir = args.batch_root / f"Group{group_number}"
        if not group_dir.exists():
            raise FileNotFoundError(f"Expected group assignment folder does not exist: {group_dir}")
        output_html = result_dir / f"group{group_number}_rule_vs_manual_check.html"
        summary = checker.build_report(
            result_dir=result_dir,
            group_dir=group_dir,
            rule_labels_path=args.rule_labels,
            manual_csv=None,
            output_html=output_html,
            iou_threshold=args.iou_threshold,
        )
        flattened = checker.flatten_summary_for_csv(summary)
        flattened["result_dir"] = str(result_dir)
        flattened["html_report"] = str(output_html)
        aggregate_rows.append(flattened)
        print(f"Checked Group{group_number}: {result_dir}")
        print(f"  HTML: {output_html}")
        print(f"  Usable: {summary['usable']} | covered {summary['manual_covered_images']}/{summary['expected_images']}")

    write_csv(args.aggregate_output, aggregate_rows)
    print(f"Wrote aggregate summary to {args.aggregate_output}")


if __name__ == "__main__":
    main()
