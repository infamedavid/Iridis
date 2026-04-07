import numpy as np

from .color_analysis import clamp01


def _region_scalar_map(region_id_map: np.ndarray, region_stats: dict, key: str, default: float = 0.0) -> np.ndarray:
    out = np.full(region_id_map.shape, default, dtype=np.float32)
    for region_id, stats in region_stats.items():
        out[region_id_map == region_id] = float(stats.get(key, default))
    return out


def generate_metallic_map(common: dict, eff: dict) -> np.ndarray:
    mask = common["work_mask"]
    neutrality = common["neutrality_map"]
    hsv_s = common["hsv_s"]
    highlight = common["highlight_candidate_map"]
    edge_map = common["edge_map"]
    dirt = common["dirt_candidate_map"]
    region_id_map = common["region_id_map"]
    region_stats = common["region_stats"]

    base = eff["metallic_base_probability"]
    neutrality_w = eff["metallic_neutrality_weight"]
    color_rejection = eff["metallic_color_rejection"]
    region_coh = eff["metallic_region_coherence"]
    edge_exposure = eff["metallic_edge_exposure_weight"]
    painted_expected = eff["painted_surface_expected"]
    corrosion_expected = eff["corrosion_expected"]

    color_penalty = hsv_s * color_rejection
    score = np.full_like(neutrality, base, dtype=np.float32)
    score += neutrality * neutrality_w * 0.55
    score += highlight * 0.18
    score += edge_map * edge_exposure * 0.22

    # Painted surfaces get pushed down except on edges.
    score -= color_penalty * (0.50 + painted_expected * 0.20)
    score -= dirt * corrosion_expected * 0.08

    # Region logic.
    region_neutrality = _region_scalar_map(region_id_map, region_stats, "mean_neutrality")
    region_sat = _region_scalar_map(region_id_map, region_stats, "mean_saturation")
    region_cavity = _region_scalar_map(region_id_map, region_stats, "mean_cavity")
    regional_score = (
        base * 0.50 +
        region_neutrality * neutrality_w * 0.50 -
        region_sat * color_rejection * 0.45 -
        region_cavity * corrosion_expected * 0.10
    )
    score = score * (1.0 - region_coh * 0.55) + regional_score * (region_coh * 0.55)

    # Exposed edges on painted surfaces recover some metallicity.
    score += edge_map * painted_expected * edge_exposure * 0.18

    softness = eff["metallic_threshold_softness"]
    threshold = 0.52
    if softness < 0.01:
        metallic = (score > threshold).astype(np.float32)
    else:
        metallic = 1.0 / (1.0 + np.exp(-(score - threshold) / max(0.02, softness * 0.16)))

    return clamp01(metallic) * mask
