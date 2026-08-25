#!/usr/bin/env python
"""Prepare v6 RGB images and references from source GFP/mCherry/Trans TIFFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "source_tifs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "result_pic" / "from_source_tifs"
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_REFERENCES = V6_DIR / "references" / "source_tif_reference_manifest.csv"


def split_for_image(image_id: str, train_percent: int = 70, val_percent: int = 15) -> str:
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + val_percent:
        return "val"
    return "test"


def safe_name(value: str) -> str:
    value = value.strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    value = re.sub(r"\s+", " ", value)
    return value


def image_id_from_stem(stem: str) -> str:
    return stem.replace(" ", "_")


def channel_and_condition(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    if stem.startswith("MAX_") or stem.lower() == "composite":
        return None

    patterns = [
        (r"^(?P<condition>.+?)[ _]GFP_center$", "GFP"),
        (r"^(?P<condition>.+?)[ _]mCherry_center$", "mCherry"),
        (r"^(?P<condition>.+?)[ _]Trans_center$", "Trans"),
        (r"^(?P<condition>.+?)_GFP_center$", "GFP"),
        (r"^(?P<condition>.+?)_mCherry_center$", "mCherry"),
        (r"^(?P<condition>.+?)_Trans_center$", "Trans"),
    ]
    for pattern, channel in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match:
            return channel, safe_name(match.group("condition"))
    return None


def discover_triplets(source_root: Path) -> dict[str, dict[str, Path]]:
    triplets: dict[str, dict[str, Path]] = {}
    for path in sorted(source_root.rglob("*.tif")):
        parsed = channel_and_condition(path)
        if parsed is None:
            continue
        channel, condition = parsed
        triplets.setdefault(condition, {})[channel] = path
    return {condition: channels for condition, channels in triplets.items() if {"GFP", "mCherry", "Trans"} <= set(channels)}


def read_tiff_stack(path: Path) -> np.ndarray:
    frames: list[np.ndarray] = []
    with Image.open(path) as image:
        for frame in ImageSequence.Iterator(image):
            arr = np.asarray(frame, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            frames.append(arr)
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return np.stack(frames, axis=0)


def project_to_trans_frames(stack: np.ndarray, trans_frame_count: int, channel: str, condition: str) -> np.ndarray:
    frame_count = int(stack.shape[0])
    if frame_count == trans_frame_count:
        return stack
    if frame_count % trans_frame_count != 0:
        raise ValueError(
            f"{condition} {channel}: {frame_count} frames cannot align to {trans_frame_count} Trans frames"
        )
    block_size = frame_count // trans_frame_count
    return stack.reshape(trans_frame_count, block_size, stack.shape[1], stack.shape[2]).max(axis=1)


def normalize_uint8(image: np.ndarray, low: float = 1.0, high: float = 99.8, gamma: float = 1.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(arr, low))
    hi = float(np.percentile(arr, high))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        scaled = np.power(scaled, gamma)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def merge_rgb(trans: np.ndarray, gfp: np.ndarray, mcherry: np.ndarray) -> np.ndarray:
    trans_u8 = normalize_uint8(trans, low=1.0, high=99.8, gamma=1.0)
    gfp_u8 = normalize_uint8(gfp, low=1.0, high=99.7, gamma=0.82)
    mcherry_u8 = normalize_uint8(mcherry, low=1.0, high=99.7, gamma=0.82)
    rgb = np.zeros((*trans_u8.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(0.42 * trans_u8 + 1.00 * mcherry_u8, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(0.42 * trans_u8 + 1.00 * gfp_u8, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(0.42 * trans_u8 + 1.00 * mcherry_u8, 0, 255).astype(np.uint8)
    return rgb


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def write_manifest(rows: list[dict[str, object]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def prepare_condition(
    condition: str,
    paths: dict[str, Path],
    source_root: Path,
    output_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trans_stack = read_tiff_stack(paths["Trans"])
    gfp_stack = project_to_trans_frames(read_tiff_stack(paths["GFP"]), trans_stack.shape[0], "GFP", condition)
    mcherry_stack = project_to_trans_frames(read_tiff_stack(paths["mCherry"]), trans_stack.shape[0], "mCherry", condition)

    image_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    for frame_index in range(int(trans_stack.shape[0])):
        stem = f"{condition}_frame_{frame_index:03d}"
        image_id = image_id_from_stem(stem)
        rgb_path = output_root / "RGB" / f"{stem}.png"
        trans_path = output_root / "references" / "Trans" / f"{stem}_Trans.png"
        gfp_path = output_root / "references" / "GFP_projected" / f"{stem}_GFP_projected.png"
        mcherry_path = output_root / "references" / "mCherry_projected" / f"{stem}_mCherry_projected.png"

        rgb = merge_rgb(trans_stack[frame_index], gfp_stack[frame_index], mcherry_stack[frame_index])
        trans_u8 = normalize_uint8(trans_stack[frame_index], low=1.0, high=99.8)
        gfp_u8 = normalize_uint8(gfp_stack[frame_index], low=1.0, high=99.7, gamma=0.82)
        mcherry_u8 = normalize_uint8(mcherry_stack[frame_index], low=1.0, high=99.7, gamma=0.82)
        write_png(rgb_path, rgb)
        write_png(trans_path, trans_u8)
        write_png(gfp_path, gfp_u8)
        write_png(mcherry_path, mcherry_u8)

        image_rows.append(
            {
                "image_id": image_id,
                "image_path": str(rgb_path.relative_to(PROJECT_ROOT)),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "split": split_for_image(image_id),
                "annotation_status": "unreviewed",
                "notes": f"source_tifs={paths['GFP'].name};{paths['mCherry'].name};{paths['Trans'].name}",
            }
        )
        for ref_name, ref_path, source_tif in [
            ("Trans", trans_path, paths["Trans"]),
            ("GFP projected", gfp_path, paths["GFP"]),
            ("mCherry projected", mcherry_path, paths["mCherry"]),
        ]:
            reference_rows.append(
                {
                    "image_id": image_id,
                    "reference_name": ref_name,
                    "reference_path": str(ref_path.relative_to(PROJECT_ROOT)),
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "source_tif": str(source_tif.relative_to(source_root)),
                    "frame_index": frame_index,
                }
            )
    return image_rows, reference_rows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    triplets = discover_triplets(source_root)
    if not triplets:
        raise RuntimeError(f"No GFP/mCherry/Trans triplets found under {source_root}")

    all_images: list[dict[str, object]] = []
    all_references: list[dict[str, object]] = []
    for condition, paths in sorted(triplets.items()):
        image_rows, reference_rows = prepare_condition(condition, paths, source_root, output_root)
        all_images.extend(image_rows)
        all_references.extend(reference_rows)
        print(f"{condition}: {len(image_rows)} aligned frames")

    write_manifest(
        all_images,
        args.manifest.resolve(),
        ["image_id", "image_path", "width", "height", "split", "annotation_status", "notes"],
    )
    write_manifest(
        all_references,
        args.references.resolve(),
        ["image_id", "reference_name", "reference_path", "width", "height", "source_tif", "frame_index"],
    )
    print(f"Wrote {len(all_images)} RGB images under {output_root / 'RGB'}")
    print(f"Wrote image manifest to {args.manifest}")
    print(f"Wrote {len(all_references)} reference rows to {args.references}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
