import numpy as np

from .color_analysis import clamp01


def _region_scalar_map(region_id_map: np.ndarray, region_stats: dict, key: str, default: float = 0.0) -> np.ndarray:
    out = np.full(region_id_map.shape, default, dtype=np.float32)
    for region_id, stats in region_stats.items():
        out[region_id_map == region_id] = float(stats.get(key, default))
    return out


def _compress_signal(x: np.ndarray, threshold: float, gain: float, gamma: float = 1.0) -> np.ndarray:
    """
    Soft threshold + gain to avoid treating every tiny variation as strong roughness evidence.
    """
    y = np.maximum(x - threshold, 0.0) * gain
    y = clamp01(y)
    if gamma != 1.0:
        y = np.power(y, gamma)
    return y.astype(np.float32)


def _regional_mean_map(region_id_map: np.ndarray, value_map: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full(region_id_map.shape, default, dtype=np.float32)
    region_ids = np.unique(region_id_map)
    for region_id in region_ids:
        sel = region_id_map == region_id
        if np.any(sel):
            out[sel] = float(np.mean(value_map[sel]))
    return out


def generate_roughness_map(common: dict, eff: dict) -> np.ndarray:
    mask = common["work_mask"]

    detail_abs = common["detail_abs"]
    local_contrast = common["local_contrast_map"]
    cavity = common["cavity_map"]
    dirt = common["dirt_candidate_map"]
    highlight = common["highlight_candidate_map"]
    highlight_sharp = common["highlight_sharpness_map"]
    neutrality = common["neutrality_map"]
    region_id_map = common["region_id_map"]
    region_stats = common["region_stats"]

    lab = common["lab"]
    hsv = common["hsv"]

    # LAB normalized as in the rest of the pipeline:
    # L in [0..1], a and b roughly in [-1..1]
    lab_a = lab[:, :, 1]
    lab_b = lab[:, :, 2]

    hsv_s = hsv[:, :, 1]
    hsv_v = hsv[:, :, 2]

    base = eff["roughness_base"]
    micro_w = eff["roughness_microdetail_weight"]
    cavity_w = eff["roughness_cavity_weight"]
    highlight_resp = eff["roughness_highlight_response"]
    region_coh = eff["roughness_region_coherence"]
    dirt_expected = eff["dirt_expected"]
    corrosion_expected = eff["corrosion_expected"]

    # -------------------------------------------------------------------------
    # 1) Material cues
    # -------------------------------------------------------------------------

    # Rust / oxidation score:
    # positive a + positive b + moderate saturation => more likely rust
    rust_chroma = clamp01((lab_a * 0.70) + (lab_b * 0.60))
    rust_sat = clamp01((hsv_s - 0.10) * 1.25)
    rust_value_gate = clamp01(1.0 - np.abs(hsv_v - 0.55) * 1.35)
    rust_score = clamp01(rust_chroma * 0.60 + rust_sat * 0.25 + rust_value_gate * 0.15)

    # Smooth exposed metal hint:
    # neutral, less rusty, not too dirty/cavity-heavy, some highlight evidence
    compact_highlight = clamp01(highlight * 0.55 + highlight_sharp * 0.45)
    smooth_metal_score = clamp01(
        neutrality * 0.55
        + compact_highlight * 0.35
        - rust_score * 0.45
        - dirt * 0.25
        - cavity * 0.20
    )

    # -------------------------------------------------------------------------
    # 2) Compress noisy contributors so they do not push everything to white
    # -------------------------------------------------------------------------

    detail_comp = _compress_signal(detail_abs, threshold=0.08, gain=2.2, gamma=1.15)
    contrast_comp = _compress_signal(local_contrast, threshold=0.06, gain=1.9, gamma=1.10)
    cavity_comp = _compress_signal(cavity, threshold=0.10, gain=1.6, gamma=1.00)
    dirt_comp = _compress_signal(dirt, threshold=0.05, gain=1.8, gamma=1.00)

    # Detail should matter much more in rust / dirt than on cleaner exposed metal
    detail_selector = clamp01(0.20 + rust_score * 0.70 + dirt_comp * 0.35 + cavity_comp * 0.20)

    # -------------------------------------------------------------------------
    # 3) Base roughness build
    # -------------------------------------------------------------------------

    rough = np.full_like(detail_abs, base, dtype=np.float32)

    # Rust and dirt are the main roughness drivers in this kind of surface
    rough += rust_score * (0.22 + 0.30 * corrosion_expected)
    rough += dirt_comp * (0.14 + 0.24 * dirt_expected)

    # Cavity should act, but localized
    rough += cavity_comp * cavity_w * (0.12 + 0.18 * corrosion_expected)

    # Fine detail should no longer whiten the whole plate
    rough += detail_comp * detail_selector * micro_w * 0.18
    rough += contrast_comp * detail_selector * micro_w * 0.10

    # Smooth exposed neutral metal should pull roughness down
    rough -= smooth_metal_score * (0.16 + 0.24 * highlight_resp)

    # Compact highlights suggest smoother surface
    rough -= compact_highlight * highlight_resp * 0.18

    # -------------------------------------------------------------------------
    # 4) Regional coherence
    # -------------------------------------------------------------------------

    region_neutrality = _region_scalar_map(region_id_map, region_stats, "mean_neutrality")
    region_highlight = _region_scalar_map(region_id_map, region_stats, "mean_highlight")

    regional_smooth_hint = clamp01(
        region_neutrality * 0.55
        + region_highlight * 0.30
        - rust_score * 0.20
    )

    # Small extra pull down on likely smooth metallic regions
    rough -= regional_smooth_hint * region_coh * 0.08

    # Blend toward regional mean so results are less speckled
    region_mean_map = _regional_mean_map(region_id_map, rough, default=base)
    rough = rough * (1.0 - region_coh * 0.40) + region_mean_map * (region_coh * 0.40)

    # -------------------------------------------------------------------------
    # 5) Safety clamps / non-metal bias
    # -------------------------------------------------------------------------

    # Colored non-neutral regions get a slight push upward, but much milder than before
    rough += (1.0 - neutrality) * 0.03

    rough = clamp01(rough)

    return rough * mask