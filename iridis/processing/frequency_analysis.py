import cv2
import numpy as np

from .color_analysis import clamp01


def _odd_kernel_from_scale(size: int, scale: float, minimum: int) -> int:
    k = max(minimum, int(size * scale))
    if k % 2 == 0:
        k += 1
    return k


def compute_frequency_maps(gray: np.ndarray, enable_heavier_relief: bool = False) -> dict:
    height, width = gray.shape[:2]
    mindim = min(width, height)

    illum_k = _odd_kernel_from_scale(mindim, 0.06, 7)
    mid_k = _odd_kernel_from_scale(mindim, 0.02, 5)
    local_k = _odd_kernel_from_scale(mindim, 0.012, 5)

    illum_low = cv2.GaussianBlur(gray, (illum_k, illum_k), 0)
    mid_tone = cv2.GaussianBlur(gray, (mid_k, mid_k), 0)
    detail_high = gray - mid_tone
    detail_abs = np.abs(detail_high)
    if detail_abs.max() > 0.0:
        detail_abs = detail_abs / detail_abs.max()

    local_mean = cv2.GaussianBlur(gray, (local_k, local_k), 0)
    local_sq_mean = cv2.GaussianBlur(gray * gray, (local_k, local_k), 0)
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    local_contrast = np.sqrt(local_var)
    if local_contrast.max() > 0.0:
        local_contrast = local_contrast / local_contrast.max()

    maps = {
        "illum_low": clamp01(illum_low.astype(np.float32)),
        "mid_tone": clamp01(mid_tone.astype(np.float32)),
        "detail_high": detail_high.astype(np.float32),
        "detail_abs": clamp01(detail_abs.astype(np.float32)),
        "local_contrast_map": clamp01(local_contrast.astype(np.float32)),
    }

    if enable_heavier_relief:
        mid_structure = mid_tone - illum_low
        relief_base = gray - illum_low
        maps["mid_structure_map"] = mid_structure.astype(np.float32)
        maps["relief_base_map"] = relief_base.astype(np.float32)

    return maps
