import cv2
import numpy as np

from .color_analysis import clamp01


def _odd_kernel_from_scale(size: int, scale: float, minimum: int) -> int:
    k = max(minimum, int(size * scale))
    if k % 2 == 0:
        k += 1
    return k


def _robust_normalize_signed(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    valid = mask > 0.001 if mask is not None else np.ones_like(x, dtype=bool)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)

    p5 = float(np.percentile(x[valid], 5))
    p95 = float(np.percentile(x[valid], 95))
    span = max(1e-6, p95 - p5)
    y = (x - p5) / span
    y = (y - 0.5) * 2.0
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def compute_frequency_maps(gray: np.ndarray, work_mask: np.ndarray | None = None) -> dict:
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

    # Shared relief basis:
    # mid-scale structure comes from illumination-compensated luminance residual,
    # then mixed with a softer band-pass component.
    illum_safe = np.maximum(illum_low, 1e-4)
    rel_excess = (gray - illum_low) / illum_safe
    rel_excess = _robust_normalize_signed(rel_excess, work_mask)

    mid_residual = mid_tone - illum_low
    mid_residual = _robust_normalize_signed(mid_residual, work_mask)

    mid_structure_map = np.clip(rel_excess * 0.65 + mid_residual * 0.35, -1.0, 1.0).astype(np.float32)
    relief_base_map = np.clip(mid_structure_map + detail_high * 0.28, -1.0, 1.0).astype(np.float32)

    return {
        "illum_low": clamp01(illum_low.astype(np.float32)),
        "mid_tone": clamp01(mid_tone.astype(np.float32)),
        "detail_high": detail_high.astype(np.float32),
        "detail_abs": clamp01(detail_abs.astype(np.float32)),
        "local_contrast_map": clamp01(local_contrast.astype(np.float32)),
        "mid_structure_map": mid_structure_map,
        "relief_base_map": relief_base_map,
    }
