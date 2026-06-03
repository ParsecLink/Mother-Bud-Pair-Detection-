"""Image preprocessing helpers for visualization."""

from __future__ import annotations

import numpy as np
from skimage import filters, morphology


def max_project_blocks(stack: np.ndarray, block_size: int) -> np.ndarray:
    """Max-project every block of frames in a 3D stack."""

    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {stack.shape}")
    frame_count = int(stack.shape[0])
    if frame_count % block_size != 0:
        raise ValueError(f"Frame count {frame_count} is not divisible by {block_size}")
    reshaped = stack.reshape(frame_count // block_size, block_size, stack.shape[1], stack.shape[2])
    return reshaped.max(axis=1)


def percentile_normalize(image: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    """Normalize image intensities to [0, 1] using percentile stretch."""

    image = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(image, low))
    hi = float(np.percentile(image, high))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    scaled = (image - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def background_correct(image: np.ndarray, gaussian_sigma: float, tophat_radius: int) -> np.ndarray:
    """Apply light smoothing and white top-hat background correction."""

    normalized = percentile_normalize(image, low=0.5, high=99.9)
    smoothed = filters.gaussian(normalized, sigma=gaussian_sigma, preserve_range=True)
    corrected = morphology.white_tophat(smoothed, morphology.disk(tophat_radius))
    return np.asarray(corrected, dtype=np.float32)


def trans_display(image: np.ndarray) -> np.ndarray:
    """Prepare Trans as a contrast-stretched grayscale background."""

    return percentile_normalize(image, low=1.0, high=99.8)


def fluor_display(image: np.ndarray, gaussian_sigma: float, tophat_radius: int, gamma: float = 0.85) -> np.ndarray:
    """Prepare fluorescence as enhanced but not over-saturated display data."""

    corrected = background_correct(image, gaussian_sigma=gaussian_sigma, tophat_radius=tophat_radius)
    stretched = percentile_normalize(corrected, low=1.0, high=99.7)
    return np.power(np.clip(stretched, 0.0, 1.0), gamma)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float [0, 1] image to uint8."""

    return np.clip(np.asarray(image, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
