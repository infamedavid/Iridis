import cv2
import numpy as np

from .color_analysis import clamp01


def _majority_region_values(values: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    region_values = {}
    for region_id in np.unique(labels):
        if region_id <= 0:
            continue
        sel = labels == region_id
        if not np.any(sel):
            continue
        region_values[int(region_id)] = float(np.mean(values[sel]) > threshold)
    return region_values


def _remove_tiny_components(binary_map: np.ndarray, min_area: int, mask: np.ndarray) -> np.ndarray:
    work = ((binary_map > 0.5) & (mask > 0.5)).astype(np.uint8)
    comp_count, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    cleaned = np.zeros_like(work, dtype=np.uint8)
    for comp_id in range(1, comp_count):
        if stats[comp_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == comp_id] = 1
    return cleaned.astype(np.float32)


def stabilize_binary_map(
    value_map: np.ndarray,
    region_id_map: np.ndarray,
    work_mask: np.ndarray,
    threshold: float,
    min_area_px: int = 24,
    region_mix: float = 0.70,
) -> np.ndarray:
    """
    Optional region-level cleanup pass for metallic/roughness binarization stability.
    Keeps behavior conservative by blending original values with refined region guidance.
    """
    if np.count_nonzero(work_mask > 0.5) < 16:
        return clamp01(value_map * work_mask)

    binary = (value_map > threshold).astype(np.float32)
    region_majority = _majority_region_values(value_map, region_id_map, threshold=threshold)
    if region_majority:
        region_map = np.zeros_like(value_map, dtype=np.float32)
        for region_id, region_value in region_majority.items():
            region_map[region_id_map == region_id] = region_value
        binary = binary * (1.0 - region_mix) + region_map * region_mix

    binary = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)
    binary = (binary > 0.5).astype(np.float32)
    binary = _remove_tiny_components(binary, min_area=min_area_px, mask=work_mask)
    binary = cv2.medianBlur((binary * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    return clamp01(binary) * work_mask
