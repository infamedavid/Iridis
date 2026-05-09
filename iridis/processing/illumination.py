import cv2
import numpy as np


def stabilize_lighting_for_analysis(
    rgb: np.ndarray,
    work_mask: np.ndarray | None = None,
    strength: float = 0.65,
) -> dict:
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    gray = (
        0.2126 * rgb[:, :, 0] +
        0.7152 * rgb[:, :, 1] +
        0.0722 * rgb[:, :, 2]
    ).astype(np.float32)

    if work_mask is not None:
        valid = work_mask > 0.5
    else:
        valid = np.ones(gray.shape, dtype=bool)

    if strength <= 0.0 or np.count_nonzero(valid) < 16:
        return {
            "analysis_rgb": rgb.copy(),
            "illumination_field": np.ones(gray.shape, dtype=np.float32),
            "lighting_highlight_mask": np.zeros(gray.shape, dtype=np.float32),
        }

    h, w = gray.shape
    sigma = max(h, w) * 0.04
    sigma = max(8.0, float(sigma))
    illum = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)

    illum_safe = np.maximum(illum, 1e-4)
    illum_median = float(np.median(illum_safe[valid]))
    illum_norm = illum_safe / max(illum_median, 1e-4)
    illum_norm = np.clip(illum_norm, 0.45, 2.20).astype(np.float32)

    corrected = rgb / illum_norm[:, :, None]
    corrected = np.clip(corrected, 0.0, 1.0).astype(np.float32)

    rgb_max = np.max(rgb, axis=2)
    rgb_min = np.min(rgb, axis=2)
    sat = (rgb_max - rgb_min) / np.maximum(rgb_max, 1e-4)

    p95 = np.percentile(gray[valid], 95)
    highlight_luma = np.maximum(float(p95), 0.78)

    highlight_seed = np.clip((gray - highlight_luma) / 0.18, 0.0, 1.0)
    low_sat = np.clip((0.45 - sat) / 0.45, 0.0, 1.0)
    highlight_mask = highlight_seed * low_sat
    highlight_mask = cv2.GaussianBlur(highlight_mask.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)
    highlight_mask = np.clip(highlight_mask, 0.0, 1.0).astype(np.float32)

    corrected = corrected * (1.0 - highlight_mask[:, :, None]) + rgb * highlight_mask[:, :, None]

    strength = float(np.clip(strength, 0.0, 1.0))
    analysis_rgb = rgb * (1.0 - strength) + corrected * strength
    analysis_rgb = np.clip(analysis_rgb, 0.0, 1.0).astype(np.float32)

    if work_mask is not None:
        m = np.clip(work_mask, 0.0, 1.0).astype(np.float32)
        analysis_rgb = rgb * (1.0 - m[:, :, None]) + analysis_rgb * m[:, :, None]

    return {
        "analysis_rgb": analysis_rgb.astype(np.float32),
        "illumination_field": illum_norm.astype(np.float32),
        "lighting_highlight_mask": highlight_mask.astype(np.float32),
    }
