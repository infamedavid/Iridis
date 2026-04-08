import cv2
import numpy as np


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[:, :, 0] +
        0.7152 * rgb[:, :, 1] +
        0.0722 * rgb[:, :, 2]
    ).astype(np.float32)


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb8 = (clamp01(rgb) * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


def rgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb8 = (clamp01(rgb) * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0
    a = (lab[:, :, 1] - 128.0) / 127.0
    b = (lab[:, :, 2] - 128.0) / 127.0
    return l, a, b


def compute_neutrality_map(hsv_s: np.ndarray, lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    chroma_mag = np.sqrt(lab_a * lab_a + lab_b * lab_b)
    chroma_mag = chroma_mag / (float(chroma_mag.max()) + 1e-8)
    neutrality = (1.0 - hsv_s) * 0.55 + (1.0 - chroma_mag) * 0.45
    return clamp01(neutrality.astype(np.float32))
