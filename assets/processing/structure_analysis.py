import cv2
import numpy as np

from .color_analysis import clamp01


def compute_structure_maps(
    gray: np.ndarray,
    hsv_s: np.ndarray,
    lab_a: np.ndarray,
    lab_b: np.ndarray,
    local_contrast_map: np.ndarray,
    mask: np.ndarray,
) -> dict:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    if grad_mag.max() > 0.0:
        grad_mag = grad_mag / grad_mag.max()

    edge_map = grad_mag.astype(np.float32)
    strong_edge_map = clamp01((edge_map - 0.18) / 0.30)

    kernel_small = np.ones((5, 5), np.uint8)
    kernel_med = np.ones((9, 9), np.uint8)

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_small)
    dark_residue_map = blackhat.astype(np.float32)
    if dark_residue_map.max() > 0.0:
        dark_residue_map = dark_residue_map / dark_residue_map.max()

    cavity_map = clamp01(
        0.50 * dark_residue_map +
        0.30 * edge_map +
        0.20 * local_contrast_map
    )

    rust_bias = clamp01((lab_a * 0.65) + (lab_b * 0.45))
    dirt_candidate_map = clamp01(
        0.45 * cavity_map +
        0.25 * dark_residue_map +
        0.20 * rust_bias +
        0.10 * (1.0 - hsv_s)
    )

    highlight_seed = clamp01((gray - 0.62) / 0.30)
    sat_reject = clamp01(1.0 - hsv_s * 0.85)
    highlight_candidate_map = clamp01(
        highlight_seed * (0.70 * sat_reject + 0.30 * local_contrast_map)
    )

    blur_small = cv2.GaussianBlur(highlight_candidate_map, (5, 5), 0)
    blur_large = cv2.GaussianBlur(highlight_candidate_map, (11, 11), 0)
    highlight_sharpness_map = clamp01(blur_small - blur_large)
    if highlight_sharpness_map.max() > 0.0:
        highlight_sharpness_map = highlight_sharpness_map / highlight_sharpness_map.max()

    mask_u8 = (mask > 0.5).astype(np.uint8)
    uv_distance_map = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    if uv_distance_map.max() > 0.0:
        uv_distance_map = uv_distance_map / uv_distance_map.max()
    else:
        uv_distance_map = np.ones_like(mask, dtype=np.float32)

    border_falloff_map = clamp01(uv_distance_map * 2.0)

    return {
        "grad_x": grad_x.astype(np.float32),
        "grad_y": grad_y.astype(np.float32),
        "grad_mag": edge_map,
        "edge_map": edge_map,
        "strong_edge_map": strong_edge_map,
        "dark_residue_map": clamp01(dark_residue_map.astype(np.float32)),
        "cavity_map": cavity_map,
        "dirt_candidate_map": dirt_candidate_map,
        "highlight_candidate_map": highlight_candidate_map,
        "highlight_sharpness_map": highlight_sharpness_map,
        "uv_distance_map": uv_distance_map.astype(np.float32),
        "border_falloff_map": border_falloff_map.astype(np.float32),
    }
