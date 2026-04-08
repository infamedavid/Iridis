import cv2
import numpy as np

from .color_analysis import clamp01


def generate_normal_map(common: dict, eff: dict) -> np.ndarray:
    mask = common["work_mask"]
    detail = common["detail_high"]
    local_contrast = common["local_contrast_map"]
    uv_distance = common["uv_distance_map"]
    border_falloff = common["border_falloff_map"]
    use_enhanced_relief = bool(common.get("enhanced_relief_enabled", False))
    relief_control_mask = common.get("relief_control_mask")

    detail_weight = eff["normal_detail_weight"]
    mid_weight = eff["normal_mid_weight"]
    smoothing = eff["normal_smoothing"]

    local_detail_weight = detail_weight
    local_mid_weight = mid_weight
    if relief_control_mask is not None:
        centered_mask = (relief_control_mask - 0.5) * 2.0
        local_detail_weight = np.clip(
            1.0 + centered_mask * (detail_weight - 1.0),
            0.0,
            None,
        ).astype(np.float32)
        local_mid_weight = np.clip(
            1.0 + centered_mask * (mid_weight - 1.0),
            0.0,
            None,
        ).astype(np.float32)

    # Denoise detail before deriving normals to avoid crunchy micro-noise.
    detail_smooth = cv2.GaussianBlur(detail, (3, 3), 0)
    detail_hp = detail - detail_smooth
    detail_clean = detail_smooth * 0.85 + detail_hp * 0.25

    if use_enhanced_relief:
        mid_structure = common["mid_structure_map"]
        relief_base = common["relief_base_map"]
        relief = (
            detail_clean * local_detail_weight * 0.9 +
            mid_structure * 0.48 * local_mid_weight +
            relief_base * 0.12
        )
    else:
        mid = common["mid_tone"] - np.mean(common["mid_tone"])
        relief = detail_clean * local_detail_weight + mid * 0.32 * local_mid_weight
    blur_amount = max(1, int(1 + smoothing * 8))
    if blur_amount % 2 == 0:
        blur_amount += 1
    relief = cv2.GaussianBlur(relief, (blur_amount, blur_amount), 0)

    dx = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(relief, cv2.CV_32F, 0, 1, ksize=3)

    detail_gate = clamp01(local_contrast * 0.75 + np.abs(detail_clean) * 0.65)
    if np.isscalar(local_detail_weight) and np.isscalar(local_mid_weight):
        strength = max(0.001, local_detail_weight * 1.35 + local_mid_weight * 0.55)
    else:
        strength = np.clip(local_detail_weight * 1.35 + local_mid_weight * 0.55, 0.001, None)
    strength_map = strength * (0.35 + detail_gate * 0.65)
    nx = -dx * strength_map
    ny = -dy * strength_map
    if eff["normal_format"] == "DIRECTX":
        ny = -ny
    nz = np.ones_like(nx)

    length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= length
    ny /= length
    nz /= length

    normal = np.stack([
        nx * 0.5 + 0.5,
        ny * 0.5 + 0.5,
        nz * 0.5 + 0.5,
    ], axis=2)

    neutral = np.dstack([
        np.full_like(mask, 0.5),
        np.full_like(mask, 0.5),
        np.full_like(mask, 1.0),
    ])

    fade = clamp01(np.clip(uv_distance * 2.4, 0.0, 1.0) * border_falloff)
    normal = neutral * (1.0 - fade[:, :, None]) + normal * fade[:, :, None]
    normal = neutral * (1.0 - mask[:, :, None]) + normal * mask[:, :, None]
    return clamp01(normal)
