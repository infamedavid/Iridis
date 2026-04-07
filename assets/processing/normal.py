import cv2
import numpy as np

from .color_analysis import clamp01


def generate_normal_map(common: dict, eff: dict) -> np.ndarray:
    mask = common["work_mask"]
    detail = common["detail_high"]
    mid = common["mid_tone"] - np.mean(common["mid_tone"])
    uv_distance = common["uv_distance_map"]
    border_falloff = common["border_falloff_map"]

    detail_weight = eff["normal_detail_weight"]
    mid_weight = eff["normal_mid_weight"]
    smoothing = eff["normal_smoothing"]

    relief = detail * detail_weight + mid * 0.30 * mid_weight
    blur_amount = max(1, int(1 + smoothing * 8))
    if blur_amount % 2 == 0:
        blur_amount += 1
    relief = cv2.GaussianBlur(relief, (blur_amount, blur_amount), 0)

    dx = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(relief, cv2.CV_32F, 0, 1, ksize=3)

    strength = max(0.001, detail_weight * 1.6 + mid_weight * 0.5)
    nx = -dx * strength
    ny = -dy * strength
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

    fade = clamp01(np.clip(uv_distance * 2.0, 0.0, 1.0) * border_falloff)
    normal = neutral * (1.0 - fade[:, :, None]) + normal * fade[:, :, None]
    normal = neutral * (1.0 - mask[:, :, None]) + normal * mask[:, :, None]
    return clamp01(normal)
