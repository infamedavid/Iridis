import cv2
import numpy as np

from .color_analysis import clamp01


def _region_scalar_map(region_id_map: np.ndarray, region_stats: dict, key: str, default: float = 0.0) -> np.ndarray:
    out = np.full(region_id_map.shape, default, dtype=np.float32)
    for region_id, stats in region_stats.items():
        out[region_id_map == region_id] = float(stats.get(key, default))
    return out


def generate_albedo_map(common: dict, eff: dict) -> np.ndarray:
    rgb = common["src_rgb"]
    mask = common["work_mask"]
    gray = common["gray"]
    illum_low = common["illum_low"]
    hsv_s = common["hsv_s"]
    lab_a = common["lab_a"]
    lab_b = common["lab_b"]
    cavity = common["cavity_map"]
    highlight = common["highlight_candidate_map"]
    border_falloff = common["border_falloff_map"]
    region_id_map = common["region_id_map"]
    region_stats = common["region_stats"]

    delight_strength = eff["albedo_delight_strength"]
    highlight_suppression = eff["albedo_highlight_suppression"]
    color_protection = eff["albedo_color_protection"]
    dirt_cleanup = eff["albedo_dirt_cleanup_bias"]
    corrosion_expected = eff["corrosion_expected"]
    painted_expected = eff["painted_surface_expected"]

    # More stable delight than simple division.
    illum_safe = np.maximum(illum_low, 1e-4)
    normalized = gray / illum_safe
    p02 = np.percentile(normalized[mask > 0.5], 2) if np.any(mask > 0.5) else 0.0
    p98 = np.percentile(normalized[mask > 0.5], 98) if np.any(mask > 0.5) else 1.0
    normalized = clamp01((normalized - p02) / max(1e-6, p98 - p02))

    target_luma = (1.0 - delight_strength) * gray + delight_strength * normalized

    # Regional color protection: painted / rusty regions keep more identity.
    region_sat = _region_scalar_map(region_id_map, region_stats, "mean_saturation")
    region_cavity = _region_scalar_map(region_id_map, region_stats, "mean_cavity")
    rust_hint = clamp01(lab_a * 0.65 + lab_b * 0.45)
    paint_hint = clamp01(hsv_s * 0.65 + region_sat * 0.35)
    protect_map = clamp01(
        color_protection * 0.55 +
        paint_hint * painted_expected * 0.30 +
        rust_hint * corrosion_expected * 0.25
    )

    # Highlight suppression should spare colorful paint and rust a bit.
    highlight_kill = clamp01(
        highlight * highlight_suppression * (1.0 - paint_hint * 0.35)
    )
    target_luma = clamp01(target_luma * (1.0 - highlight_kill * 0.42))

    # Optional dirt cleanup in cavity, but conservative on rust.
    cavity_cleanup = cavity * dirt_cleanup * (1.0 - rust_hint * corrosion_expected * 0.65)
    target_luma = clamp01(target_luma + cavity_cleanup * 0.10)

    gray_safe = np.maximum(gray, 1e-4)
    luminance_scale = (target_luma / gray_safe)[:, :, None]
    flattened = rgb * ((1.0 - protect_map[:, :, None]) + protect_map[:, :, None] * luminance_scale)

    # Mild bilateral cleanup to reduce ugly photobake noise without killing edges.
    flat32 = np.clip(flattened, 0.0, 1.0).astype(np.float32)
    bilateral = cv2.bilateralFilter(flat32, d=0, sigmaColor=0.05, sigmaSpace=5)
    flattened = flat32 * 0.75 + bilateral * 0.25

    flattened = clamp01(flattened)
    background = np.zeros_like(flattened)
    edge_mix = border_falloff[:, :, None]
    return ((background * (1.0 - edge_mix)) + (flattened * edge_mix)) * mask[:, :, None]
