import cv2
import numpy as np

from .color_analysis import clamp01


def _majority_filter_labels(region_id_map: np.ndarray, mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    filtered = cv2.medianBlur(region_id_map.astype(np.uint8), ksize)
    out = region_id_map.copy()
    out[mask] = filtered[mask].astype(np.int32)
    return out


def _remove_tiny_components(region_id_map: np.ndarray, mask: np.ndarray, region_count: int) -> np.ndarray:
    out = region_id_map.copy()
    min_size = max(20, int(np.count_nonzero(mask) * 0.0012))
    kernel = np.ones((3, 3), np.uint8)

    for region_id in range(1, region_count + 1):
        region_bin = ((out == region_id) & mask).astype(np.uint8)
        if np.count_nonzero(region_bin) == 0:
            continue
        comp_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(region_bin, connectivity=8)
        for comp_idx in range(1, comp_count):
            comp_size = int(stats[comp_idx, cv2.CC_STAT_AREA])
            if comp_size >= min_size:
                continue
            comp_mask = labels == comp_idx
            dilated = cv2.dilate(comp_mask.astype(np.uint8), kernel, iterations=1) > 0
            ring = dilated & (~comp_mask) & mask
            neighbor_ids = out[ring]
            neighbor_ids = neighbor_ids[neighbor_ids > 0]
            if neighbor_ids.size == 0:
                continue
            new_region = int(np.bincount(neighbor_ids).argmax())
            out[comp_mask] = new_region

    return out


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
    x_norm = xx / max(1.0, float(w - 1))
    y_norm = yy / max(1.0, float(h - 1))

    feats = np.stack([
        common["lab_l"][mask],
        common["lab_a"][mask] * 0.8,
        common["lab_b"][mask] * 0.8,
        common["hsv_s"][mask] * 0.7,
        common["neutrality_map"][mask] * 0.7,
        common["local_contrast_map"][mask] * 0.5,
        x_norm[mask] * 0.12,
        y_norm[mask] * 0.12,
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
    region_id_map[mask] = labels + 1
    region_id_map = _majority_filter_labels(region_id_map, mask, ksize=3)
    region_id_map = _remove_tiny_components(region_id_map, mask, region_count)
    region_id_map = _majority_filter_labels(region_id_map, mask, ksize=3)

    # Smooth region ids a bit inside the mask.
    region_seed_map[mask] = region_id_map[mask].astype(np.float32) / float(region_count)
    region_seed_map = cv2.GaussianBlur(region_seed_map, (5, 5), 0)
    region_seed_map *= common["work_mask"]

    for region_idx in range(region_count):
        reg_mask = region_id_map == (region_idx + 1)
        if not np.any(reg_mask):
            continue
        region_stats[region_idx + 1] = {
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
