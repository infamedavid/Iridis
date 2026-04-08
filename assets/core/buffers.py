import numpy as np

from ..processing.color_analysis import (
    compute_neutrality_map,
    rgb_to_gray,
    rgb_to_hsv,
    rgb_to_lab,
)
from ..processing.frequency_analysis import compute_frequency_maps
from ..processing.structure_analysis import compute_structure_maps
from ..processing.region_analysis import compute_region_maps


def build_common_buffers(rgb: np.ndarray, alpha: np.ndarray, work_mask: np.ndarray) -> dict:
    gray = rgb_to_gray(rgb)
    hsv_h, hsv_s, hsv_v = rgb_to_hsv(rgb)
    lab_l, lab_a, lab_b = rgb_to_lab(rgb)
    neutrality_map = compute_neutrality_map(hsv_s, lab_a, lab_b)

    common = {
        "src_rgb": rgb.astype(np.float32),
        "src_alpha": alpha.astype(np.float32),
        "work_mask": work_mask.astype(np.float32),
        "masked_rgb": rgb.astype(np.float32) * work_mask[:, :, None],
        "gray": gray,
        "lab_l": lab_l.astype(np.float32),
        "lab_a": lab_a.astype(np.float32),
        "lab_b": lab_b.astype(np.float32),
        "hsv_h": hsv_h.astype(np.float32),
        "hsv_s": hsv_s.astype(np.float32),
        "hsv_v": hsv_v.astype(np.float32),
        "neutrality_map": neutrality_map.astype(np.float32),
    }

    common.update(compute_frequency_maps(gray))
    common.update(
        compute_structure_maps(
            gray,
            common["hsv_s"],
            common["lab_a"],
            common["lab_b"],
            common["local_contrast_map"],
            work_mask,
        )
    )
    common.update(compute_region_maps(common))
    return common
