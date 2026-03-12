"""Estimate per-object area from an AODC density map.

Pure-function module with no torch/model dependencies — easy to unit test.
"""

from typing import List, Tuple

import numpy as np
from scipy.ndimage import label


def estimate_object_areas(
    density_map: np.ndarray,
    peak_indices: List[Tuple[int, int]],
    orig_h: int,
    orig_w: int,
    half_max_ratio: float = 0.5,
) -> List[float]:
    """Estimate the area (in original-image pixels²) of the object at each peak.

    For each peak, computes the connected region where the density exceeds
    ``peak_value * half_max_ratio`` and converts that region's pixel count
    from density-map space to original-image space.

    Args:
        density_map: 2-D array ``[H_den, W_den]`` (non-negative values).
        peak_indices: List of ``(row, col)`` positions in *density-map* coords.
        orig_h: Original image height (pixels).
        orig_w: Original image width (pixels).
        half_max_ratio: Fraction of peak value used as the region threshold.

    Returns:
        List of estimated areas (one per peak) in original-image pixel² units.
    """
    den_h, den_w = density_map.shape
    pixel_scale = (orig_h / den_h) * (orig_w / den_w)  # area of 1 density pixel in orig space

    areas: List[float] = []
    for row, col in peak_indices:
        peak_val = density_map[row, col]
        if peak_val <= 0:
            areas.append(0.0)
            continue

        threshold = peak_val * half_max_ratio
        binary = density_map >= threshold
        labeled, _ = label(binary)
        region_label = labeled[row, col]
        if region_label == 0:
            areas.append(0.0)
            continue

        region_area_den = float((labeled == region_label).sum())
        areas.append(region_area_den * pixel_scale)

    return areas
