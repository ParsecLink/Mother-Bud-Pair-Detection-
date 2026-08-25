#!/usr/bin/env python
"""Extract TIFF frames as reference PNGs for the v6 audit editor."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_OUTPUT_DIR = V6_DIR / "references" / "tif_frames"
DEFAULT_MANIFEST = V6_DIR / "references" / "tif_reference_manifest.csv"
REFERENCE_COLUMNS = ["image_id", "reference_name", "reference_path", "width", "height", "source_tif", "frame_index"]


def sanitize_filename(value: str) -> str:
    value = value.replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "reference"


def infer_condition_channel(path: Path) -> tuple[str, str]:
    stem = path.stem
    known_suffixes = [
        ("_GFP_projected", "GFP projected"),
        ("_mCherry_projected", "mCherry projected"),
        ("_Trans", "Trans"),
        ("_DIC", "DIC"),
        ("_Brightfield", "Brightfield"),
        ("_brightfield", "Brightfield"),
    ]
    for suffix, channel in known_suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], channel
    return stem, "TIFF"


def image_id_for_frame(condition: str, frame_index: int) -> str:
    return f"{condition}_frame_{frame_index:03d}".replace(" ", "_")


def normalize_to_uint8(frame: Image.Image) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.8))
    if hi <= lo:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def iter_tifs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.rglob("*.tif")))
            found.extend(sorted(path.rglob("*.tiff")))
    return sorted(set(found))


def extract_tif(path: Path, output_dir: Path) -> list[dict[str, object]]:
    condition, channel = infer_condition_channel(path)
    safe_condition = sanitize_filename(condition)
    safe_channel = sanitize_filename(channel)
    rows: list[dict[str, object]] = []
    with Image.open(path) as image:
        for frame_index, frame in enumerate(ImageSequence.Iterator(image)):
            frame_u8 = normalize_to_uint8(frame)
            output_path = output_dir / f"{safe_condition}_{safe_channel}_frame_{frame_index:03d}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame_u8, mode="L").save(output_path)
            rows.append(
                {
                    "image_id": image_id_for_frame(condition, frame_index),
                    "reference_name": channel,
                    "reference_path": str(output_path.relative_to(PROJECT_ROOT)),
                    "width": int(frame_u8.shape[1]),
                    "height": int(frame_u8.shape[0]),
                    "source_tif": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
                    "frame_index": frame_index,
                }
            )
    return rows


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFERENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="TIFF files or directories. Defaults to current project root TIFFs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    search_paths = args.paths or [
        PROJECT_ROOT,
        PROJECT_ROOT / "pic" / "projected_tifs",
        PROJECT_ROOT / "result_pic" / "projected_tifs",
    ]
    tif_paths = iter_tifs([path.resolve() for path in search_paths if path.exists()])
    rows: list[dict[str, object]] = []
    for path in tif_paths:
        extracted = extract_tif(path, args.output_dir.resolve())
        rows.extend(extracted)
        print(f"Extracted {len(extracted)} reference frames from {path}")
    write_manifest(rows, args.manifest.resolve())
    print(f"Wrote {len(rows)} reference rows to {args.manifest}")
    print(f"Reference PNGs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
