#!/usr/bin/env python
"""Train a YOLO detector for v6 when ultralytics is installed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "v6_ml_detection" / "datasets" / "yolo_v6" / "data.yaml"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "v6_ml_detection" / "runs")
    parser.add_argument("--name", default="yolo_v6_cells")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.data.exists():
        raise FileNotFoundError(f"YOLO data file not found. Run scripts/41_export_v6_yolo_dataset.py first: {args.data}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed in this Python environment. "
            "Install it in your training environment, then rerun this script."
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
