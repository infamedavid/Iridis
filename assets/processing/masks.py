import numpy as np

from .color_analysis import clamp01


def generate_cavity_mask(common: dict) -> np.ndarray:
    return clamp01(
        0.50 * common["dark_residue_map"] +
        0.30 * common["grad_mag"] +
        0.20 * common["detail_abs"]
    ) * common["work_mask"]


def generate_dirt_mask(common: dict, eff: dict) -> np.ndarray:
    rust_bias = clamp01(common["lab_a"] * 0.65 + common["lab_b"] * 0.45)
    dirt = clamp01(
        0.45 * common["dirt_candidate_map"] +
        0.20 * common["cavity_map"] +
        0.20 * rust_bias * eff["corrosion_expected"] +
        0.15 * common["local_contrast_map"]
    )
    return dirt * common["work_mask"]


def generate_metal_mask(metallic_map: np.ndarray, common: dict) -> np.ndarray:
    return clamp01(
        metallic_map * 0.75 +
        common["neutrality_map"] * 0.15 +
        common["edge_map"] * 0.10
    ) * common["work_mask"]


def generate_edge_wear_mask(common: dict, eff: dict) -> np.ndarray:
    wear = clamp01(
        0.55 * common["strong_edge_map"] +
        0.20 * common["edge_map"] +
        0.15 * common["highlight_sharpness_map"] +
        0.10 * common["neutrality_map"]
    )
    wear *= (0.35 + eff["exposed_edge_expected"] * 0.65)
    wear *= common["border_falloff_map"]
    return clamp01(wear) * common["work_mask"]
