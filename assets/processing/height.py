import cv2
import numpy as np

from .color_analysis import clamp01


def generate_height_map(common: dict, eff: dict) -> np.ndarray:
    mask = common["work_mask"]
    mid_structure = common["mid_structure_map"]
    relief_base = common["relief_base_map"]
    detail = common["detail_high"]
    cavity = common["cavity_map"]
    dark_residue = common["dark_residue_map"]
    local_contrast = common["local_contrast_map"]
    region_id_map = common["region_id_map"]
    border_falloff = common["border_falloff_map"]

    detail_soft = cv2.GaussianBlur(detail, (3, 3), 0)
    detail_gate = clamp01(local_contrast * 0.70 + cavity * 0.30)

    height = (
        relief_base * (0.55 * eff["height_macro_weight"]) +
        mid_structure * (0.35 * eff["height_macro_weight"]) +
        detail_soft * detail_gate * 0.10 * eff["height_detail_weight"] +
        cavity * 0.22 +
        dark_residue * 0.12
    )

    # Region coherence to reduce speckled high-frequency garbage.
    region_mean = np.zeros_like(height, dtype=np.float32)
    for region_id in np.unique(region_id_map):
        sel = region_id_map == region_id
        if np.any(sel):
            region_mean[sel] = float(np.mean(height[sel]))
    height = height * 0.78 + region_mean * 0.22

    height -= height.min()
    if height.max() > 0.0:
        height /= height.max()

    blur_amount = max(1, int(1 + eff["height_smoothing"] * 8))
    if blur_amount % 2 == 0:
        blur_amount += 1
    height = cv2.GaussianBlur(height, (blur_amount, blur_amount), 0)
    height = clamp01((height - 0.5) * eff["height_contrast"] + 0.5)
    height *= border_falloff
    return height * mask
