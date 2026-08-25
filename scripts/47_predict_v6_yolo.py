#!/usr/bin/env python
"""Run a trained YOLO detector and export v6 prediction CSV plus overlays."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_CLASSES = V6_DIR / "classes.txt"
DEFAULT_OUTPUT_DIR = V6_DIR / "predictions"
PREDICTION_COLUMNS = ["image_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "model_path", "source_image"]
CLASS_COLORS = {
    "single_cell": (0, 255, 255),
    "mother_bud_pair": (255, 210, 0),
    "early_bud_pair": (255, 80, 220),
}


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_predictions(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def draw_overlay(image_path: Path, rows: list[dict[str, object]], output_path: Path, hide_text: bool) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in rows:
        class_name = str(row["class_name"])
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        if hide_text:
            continue
        label = f"{class_name} {float(row['confidence']):.2f}"
        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to trained YOLO .pt weights.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--hide-text", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.model.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed in this Python environment. "
            "Install it in your training environment, then rerun prediction."
        ) from exc

    classes = read_classes(args.classes)
    model = YOLO(str(args.model))
    manifest = read_manifest(args.manifest)
    all_rows: list[dict[str, object]] = []
    overlay_dir = args.output_dir / "overlays"

    for meta in manifest:
        image_path = PROJECT_ROOT / meta["image_path"]
        result = model.predict(source=str(image_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        image_rows: list[dict[str, object]] = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            for box, class_id, score in zip(xyxy, cls, conf):
                idx = int(class_id)
                class_name = classes[idx] if 0 <= idx < len(classes) else str(idx)
                row = {
                    "image_id": meta["image_id"],
                    "class_name": class_name,
                    "confidence": round(float(score), 5),
                    "x1": round(float(box[0]), 2),
                    "y1": round(float(box[1]), 2),
                    "x2": round(float(box[2]), 2),
                    "y2": round(float(box[3]), 2),
                    "model_path": str(args.model),
                    "source_image": meta["image_path"],
                }
                image_rows.append(row)
                all_rows.append(row)
        draw_overlay(image_path, image_rows, overlay_dir / f"{Path(meta['image_path']).stem}_prediction_overlay.png", hide_text=bool(args.hide_text))

    output_csv = args.output_dir / "predictions.csv"
    write_predictions(all_rows, output_csv)
    print(f"Wrote {len(all_rows)} predictions to {output_csv}")
    print(f"Prediction overlays: {overlay_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
