"""I/O helpers for preprocessing QC."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_tiff_stack(path: Path) -> np.ndarray:
    """Read a TIFF stack into a numpy array."""

    return np.asarray(tifffile.imread(path))


def write_tiff_stack(path: Path, data: np.ndarray) -> None:
    """Write a numpy array to TIFF."""

    ensure_dir(path.parent)
    tifffile.imwrite(path, np.asarray(data))


def _strip_max_prefix(stem: str) -> str:
    if stem.lower().startswith("max_"):
        return stem[4:]
    return stem


def parse_condition_and_channel(filename: str) -> tuple[str, str]:
    """Infer condition and channel name from a raw image filename."""

    stem = _strip_max_prefix(Path(filename).stem)
    lower = stem.lower()
    if lower == "composite":
        return "Composite", "Composite"

    channel_patterns = [
        ("mCherry", r"[_ ]mcherry_center$"),
        ("GFP", r"[_ ]gfp_center$"),
        ("Trans", r"[_ ]trans_center$"),
    ]
    for channel, pattern in channel_patterns:
        if re.search(pattern, lower):
            condition = re.sub(pattern, "", stem, flags=re.IGNORECASE).rstrip(" _").lstrip(" ")
            return condition, channel
    return stem, "Unknown"


def scan_image_files(image_dir: Path) -> list[dict[str, object]]:
    """Collect basic metadata for TIFF files in the image directory."""

    rows: list[dict[str, object]] = []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in TIFF_SUFFIXES:
            continue
        array = read_tiff_stack(path)
        condition, channel = parse_condition_and_channel(path.name)
        frame_count = int(array.shape[0]) if array.ndim >= 3 else 1
        rows.append(
            {
                "path": path,
                "filename": path.name,
                "condition": condition,
                "channel": channel,
                "shape": tuple(int(v) for v in array.shape),
                "dtype": str(array.dtype),
                "frame_count": frame_count,
            }
        )
    return rows


def collect_condition_triplets(image_dir: Path, projection_group_size: int = 10) -> list[dict[str, object]]:
    """Find valid GFP/mCherry/Trans triplets from the raw image directory."""

    rows = scan_image_files(image_dir)
    by_condition: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)

    triplets: list[dict[str, object]] = []
    for condition in sorted(by_condition):
        if condition == "Composite":
            continue
        group = by_condition[condition]
        trans_rows = [row for row in group if row["channel"] == "Trans"]
        if len(trans_rows) != 1:
            continue
        trans_row = trans_rows[0]
        trans_frames = int(trans_row["frame_count"])

        def _pick(channel: str) -> dict[str, object] | None:
            matches = [row for row in group if row["channel"] == channel and int(row["frame_count"]) == projection_group_size * trans_frames]
            if len(matches) == 1:
                return matches[0]
            non_max = [row for row in matches if not str(row["filename"]).startswith("MAX_")]
            if len(non_max) == 1:
                return non_max[0]
            return None

        gfp_row = _pick("GFP")
        mcherry_row = _pick("mCherry")
        if gfp_row is None or mcherry_row is None:
            continue

        triplets.append(
            {
                "condition": condition,
                "gfp_path": Path(gfp_row["path"]),
                "mcherry_path": Path(mcherry_row["path"]),
                "trans_path": Path(trans_row["path"]),
                "trans_frames": trans_frames,
            }
        )
    return triplets
