import numpy as np

from .color_analysis import clamp01


def _region_scalar_map(region_id_map: np.ndarray, region_stats: dict, key: str, default: float = 0.0) -> np.ndarray:
    out = np.full(region_id_map.shape, default, dtype=np.float32)
    for region_id, stats in region_stats.items():
        out[region_id_map == region_id] = float(stats.get(key, default))
    return out


def generate_metallic_map(common: dict, eff: dict) -> np.ndarray:
    mask = common["work_mask"]
    metallic_control_mask = common.get("metallic_control_mask")
    neutrality = common["neutrality_map"]
    hsv_s = common["hsv_s"]
    highlight = common["highlight_candidate_map"]
    edge_map = common["edge_map"]
    highlight_sharp = common["highlight_sharpness_map"]
    dirt = common["dirt_candidate_map"]
    cavity = common["cavity_map"]
    region_id_map = common["region_id_map"]
    region_stats = common["region_stats"]

    base = eff["metallic_base_probability"]
    if metallic_control_mask is not None:
        centered_mask = (metallic_control_mask - 0.5) * 2.0
        local_metallic_bias = centered_mask * eff.get("metallic_bias_slider", 0.0)
        base = (base - eff.get("metallic_bias_slider", 0.0) * 0.5) + local_metallic_bias
    neutrality_w = eff["metallic_neutrality_weight"]
    color_rejection = eff["metallic_color_rejection"]
    region_coh = eff["metallic_region_coherence"]
    edge_exposure = eff["metallic_edge_exposure_weight"]
    painted_expected = eff["painted_surface_expected"]
    corrosion_expected = eff["corrosion_expected"]

    color_penalty = hsv_s * color_rejection
    compact_highlight = clamp01(highlight * 0.40 + highlight_sharp * 0.60)
    paint_mass = clamp01(hsv_s * 0.65 + (1.0 - neutrality) * 0.35)
    if np.isscalar(base):
        score = np.full_like(neutrality, base, dtype=np.float32)
    else:
        score = base.astype(np.float32).copy()
    score += neutrality * neutrality_w * 0.55
    score += compact_highlight * 0.16
    score += edge_map * edge_exposure * 0.22

    # Painted surfaces get pushed down except on edges.
    score -= color_penalty * (0.48 + painted_expected * 0.24)
    score -= paint_mass * painted_expected * 0.20
    score -= dirt * corrosion_expected * 0.10
    score -= cavity * corrosion_expected * 0.06

    # Region logic.
    region_neutrality = _region_scalar_map(region_id_map, region_stats, "mean_neutrality")
    region_sat = _region_scalar_map(region_id_map, region_stats, "mean_saturation")
    region_cavity = _region_scalar_map(region_id_map, region_stats, "mean_cavity")
    region_highlight = _region_scalar_map(region_id_map, region_stats, "mean_highlight")
    region_contrast = _region_scalar_map(region_id_map, region_stats, "mean_contrast")
    regional_score = (
        base * 0.50 +
        region_neutrality * neutrality_w * 0.48 +
        region_highlight * 0.16 +
        region_contrast * 0.06 -
        region_sat * color_rejection * (0.42 + painted_expected * 0.10) -
        region_cavity * corrosion_expected * 0.12
    )
    score = score * (1.0 - region_coh * 0.55) + regional_score * (region_coh * 0.55)

    # Exposed edges on painted surfaces recover some metallicity.
    score += edge_map * painted_expected * edge_exposure * 0.20

    # Non-metal presets should stay close to zero metallic except tiny highlights/edges.
    low_metal_prior = clamp01((0.20 - base) / 0.20)
    nonmetal_kill = clamp01(low_metal_prior * (0.75 + color_rejection * 0.25))
    score *= (1.0 - nonmetal_kill * 0.70)

    softness = eff["metallic_threshold_softness"]
    threshold = 0.52
    if softness < 0.01:
        metallic = (score > threshold).astype(np.float32)
    else:
        metallic = 1.0 / (1.0 + np.exp(-(score - threshold) / max(0.02, softness * 0.16)))

    return clamp01(metallic) * mask
