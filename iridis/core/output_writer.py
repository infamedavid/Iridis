import os
import bpy
import numpy as np


def ensure_output_dir(path: str):
    if not path:
        raise ValueError("Output folder is empty.")
    os.makedirs(path, exist_ok=True)


def _resolve_output_path(output_dir: str, filename: str, overwrite: bool) -> str:
    full_path = os.path.join(output_dir, filename)

    if overwrite or not os.path.exists(full_path):
        return full_path

    name, ext = os.path.splitext(filename)
    version = 2
    while True:
        candidate = os.path.join(output_dir, f"{name}_v{version:02d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        version += 1


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape (H, W, 3)")
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Gray image must have shape (H, W)")
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _save_float_image_rgba(filepath: str, rgba: np.ndarray):
    h, w, c = rgba.shape
    if c != 4:
        raise ValueError("RGBA image must have 4 channels.")

    img = bpy.data.images.new(
        name="IRIDIS_TEMP_SAVE",
        width=w,
        height=h,
        alpha=True,
        float_buffer=True,
    )

    try:
        img.filepath_raw = filepath
        img.file_format = 'PNG'
        img.pixels = rgba.ravel().tolist()
        img.save()
    finally:
        bpy.data.images.remove(img)


def save_rgb_map(image: np.ndarray, output_dir: str, filename: str, overwrite: bool) -> str:
    path = _resolve_output_path(output_dir, filename, overwrite)
    rgb = _ensure_rgb(image)
    alpha = np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
    rgba = np.concatenate((rgb, alpha), axis=2)
    _save_float_image_rgba(path, rgba)
    return path


def save_gray_map(image: np.ndarray, output_dir: str, filename: str, overwrite: bool) -> str:
    path = _resolve_output_path(output_dir, filename, overwrite)
    gray = _ensure_gray(image)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    alpha = np.ones((gray.shape[0], gray.shape[1], 1), dtype=np.float32)
    rgba = np.concatenate((rgb, alpha), axis=2)
    _save_float_image_rgba(path, rgba)
    return path