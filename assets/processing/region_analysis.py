import cv2
import numpy as np

from .color_analysis import clamp01


def compute_region_maps(common: dict, max_regions: int = 6) -> dict:
    mask = common["work_mask"] > 0.5
    h, w = common["gray"].shape

    region_seed_map = np.zeros((h, w), dtype=np.float32)
    region_id_map = np.zeros((h, w), dtype=np.int32)
    region_stats = {}

    if np.count_nonzero(mask) < 16:
        region_seed_map[mask] = 1.0
        region_stats[0] = {
            "mean_saturation": float(np.mean(common["hsv_s"][mask])) if np.any(mask) else 0.0,
            "mean_neutrality": float(np.mean(common["neutrality_map"][mask])) if np.any(mask) else 0.0,
            "mean_luminance": float(np.mean(common["gray"][mask])) if np.any(mask) else 0.0,
            "mean_contrast": float(np.mean(common["local_contrast_map"][mask])) if np.any(mask) else 0.0,
            "mean_cavity": float(np.mean(common["cavity_map"][mask])) if np.any(mask) else 0.0,
            "mean_highlight": float(np.mean(common["highlight_candidate_map"][mask])) if np.any(mask) else 0.0,
            "mean_lab_a": float(np.mean(common["lab_a"][mask])) if np.any(mask) else 0.0,
            "mean_lab_b": float(np.mean(common["lab_b"][mask])) if np.any(mask) else 0.0,
            "size": int(np.count_nonzero(mask)),
        }
        return {
            "region_seed_map": region_seed_map,
            "region_id_map": region_id_map,
            "region_stats": region_stats,
        }

    yy, xx = np.indices((h, w), dtype=np.float32)
    xx = xx / max(1.0, float(w - 1))
    yy = yy / max(1.0, float(h - 1))

    feats = np.stack([
        common["lab_l"][mask],
        common["lab_a"][mask] * 0.8,
        common["lab_b"][mask] * 0.8,
        common["hsv_s"][mask] * 0.7,
        common["neutrality_map"][mask] * 0.7,
        common["local_contrast_map"][mask] * 0.5,
        xx[mask] * 0.18,
        yy[mask] * 0.18,
    ], axis=1).astype(np.float32)

    region_count = int(np.clip(max_regions, 2, 8))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.15)
    _compactness, labels, _centers = cv2.kmeans(
        feats,
        region_count,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.reshape(-1)
    base_region_map = np.zeros((h, w), dtype=np.int32)
    base_region_map[mask] = labels + 1

    # Split disconnected islands so "regional coherence" is spatially meaningful.
    next_region_id = 1
    for region_idx in range(region_count):
        region_bin = (base_region_map == (region_idx + 1)).astype(np.uint8)
        comp_count, comp_labels = cv2.connectedComponents(region_bin, connectivity=8)
        for comp_id in range(1, comp_count):
            comp_mask = comp_labels == comp_id
            if np.count_nonzero(comp_mask) < 24:
                continue
            region_id_map[comp_mask] = next_region_id
            next_region_id += 1

    # Keep full mask covered, including tiny residual islands.
    region_id_map[(mask) & (region_id_map <= 0)] = base_region_map[(mask) & (region_id_map <= 0)]

    # Fallback if everything was tiny.
    if next_region_id == 1:
        region_id_map = base_region_map.copy()

    # Smooth region ids a bit inside the mask.
    id_max = max(1, int(region_id_map.max()))
    region_seed_map[mask] = region_id_map[mask].astype(np.float32) / float(id_max)
    region_seed_map = cv2.GaussianBlur(region_seed_map, (5, 5), 0)
    region_seed_map *= common["work_mask"]

    for region_idx in np.unique(region_id_map):
        if region_idx <= 0:
            continue
        reg_mask = region_id_map == region_idx
        if not np.any(reg_mask):
            continue
        region_stats[int(region_idx)] = {
            "mean_saturation": float(np.mean(common["hsv_s"][reg_mask])),
            "mean_neutrality": float(np.mean(common["neutrality_map"][reg_mask])),
            "mean_luminance": float(np.mean(common["gray"][reg_mask])),
            "mean_contrast": float(np.mean(common["local_contrast_map"][reg_mask])),
            "mean_cavity": float(np.mean(common["cavity_map"][reg_mask])),
            "mean_highlight": float(np.mean(common["highlight_candidate_map"][reg_mask])),
            "mean_lab_a": float(np.mean(common["lab_a"][reg_mask])),
            "mean_lab_b": float(np.mean(common["lab_b"][reg_mask])),
            "size": int(np.count_nonzero(reg_mask)),
        }

    return {
        "region_seed_map": clamp01(region_seed_map.astype(np.float32)),
        "region_id_map": region_id_map,
        "region_stats": region_stats,
    }
