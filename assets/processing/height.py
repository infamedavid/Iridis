import cv2
import numpy as np

from .color_analysis import clamp01


def generate_height_map(common: dict, eff: dict) -> np.ndarray:
    mask = common["work_mask"]
    mid = common["mid_tone"]
    detail = common["detail_high"]
    cavity = common["cavity_map"]
    dark_residue = common["dark_residue_map"]
    border_falloff = common["border_falloff_map"]

    height = (
        mid * eff["height_macro_weight"] +
        detail * 0.18 * eff["height_detail_weight"] +
        cavity * 0.18 +
        dark_residue * 0.10
    )

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
