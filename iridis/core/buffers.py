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
from ..processing.illumination import stabilize_lighting_for_analysis


def build_common_buffers(
    rgb: np.ndarray,
    alpha: np.ndarray,
    work_mask: np.ndarray,
    enable_heavier_relief: bool = False,
    enable_lighting_stabilization: bool = False,
    lighting_stabilization_strength: float = 0.65,
) -> dict:
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
        "enhanced_relief_enabled": bool(enable_heavier_relief),
    }

    common.update(
        compute_frequency_maps(
            gray,
            enable_heavier_relief=enable_heavier_relief,
        )
    )
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

    if not enable_lighting_stabilization:
        common["analysis_enabled"] = False
        return common

    analysis = stabilize_lighting_for_analysis(
        rgb,
        work_mask,
        lighting_stabilization_strength,
    )
    analysis_rgb = analysis["analysis_rgb"]
    common["analysis_enabled"] = True
    common["analysis_rgb"] = analysis_rgb
    common["illumination_field"] = analysis["illumination_field"]
    common["lighting_highlight_mask"] = analysis["lighting_highlight_mask"]

    analysis_gray = rgb_to_gray(analysis_rgb)
    analysis_hsv_h, analysis_hsv_s, analysis_hsv_v = rgb_to_hsv(analysis_rgb)
    analysis_lab_l, analysis_lab_a, analysis_lab_b = rgb_to_lab(analysis_rgb)
    analysis_neutrality_map = compute_neutrality_map(
        analysis_hsv_s,
        analysis_lab_a,
        analysis_lab_b,
    )

    common["analysis_gray"] = analysis_gray.astype(np.float32)
    common["analysis_hsv_h"] = analysis_hsv_h.astype(np.float32)
    common["analysis_hsv_s"] = analysis_hsv_s.astype(np.float32)
    common["analysis_hsv_v"] = analysis_hsv_v.astype(np.float32)
    common["analysis_lab_l"] = analysis_lab_l.astype(np.float32)
    common["analysis_lab_a"] = analysis_lab_a.astype(np.float32)
    common["analysis_lab_b"] = analysis_lab_b.astype(np.float32)
    common["analysis_neutrality_map"] = analysis_neutrality_map.astype(np.float32)

    analysis_freq = compute_frequency_maps(
        analysis_gray,
        enable_heavier_relief=enable_heavier_relief,
    )
    for key, value in analysis_freq.items():
        common[f"analysis_{key}"] = value

    analysis_struct = compute_structure_maps(
        analysis_gray,
        analysis_hsv_s,
        analysis_lab_a,
        analysis_lab_b,
        common["analysis_local_contrast_map"],
        work_mask,
    )
    for key, value in analysis_struct.items():
        common[f"analysis_{key}"] = value

    analysis_common_for_regions = {
        "work_mask": work_mask,
        "gray": analysis_gray,
        "lab_l": analysis_lab_l,
        "lab_a": analysis_lab_a,
        "lab_b": analysis_lab_b,
        "hsv_s": analysis_hsv_s,
        "neutrality_map": analysis_neutrality_map,
        "local_contrast_map": common["analysis_local_contrast_map"],
        "cavity_map": common["analysis_cavity_map"],
        "highlight_candidate_map": common["analysis_highlight_candidate_map"],
    }
    analysis_regions = compute_region_maps(analysis_common_for_regions)
    common["analysis_region_seed_map"] = analysis_regions["region_seed_map"]
    common["analysis_region_id_map"] = analysis_regions["region_id_map"]
    common["analysis_region_stats"] = analysis_regions["region_stats"]

    return common
