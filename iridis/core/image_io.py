import bpy
import numpy as np
import cv2


def get_active_image_from_image_editor(context):
    area = context.area
    if area is None or area.type != 'IMAGE_EDITOR':
        return None

    space = context.space_data
    if not space or not hasattr(space, "image"):
        return None

    return space.image


def read_image_to_numpy(image: bpy.types.Image):
    if image is None:
        raise ValueError("No active image found.")

    width = int(image.size[0])
    height = int(image.size[1])

    if width <= 0 or height <= 0:
        raise ValueError("Active image has invalid dimensions.")

    pixels = np.array(image.pixels[:], dtype=np.float32)

    if pixels.size == width * height * 4:
        rgba = pixels.reshape((height, width, 4))
        rgb = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3].copy()
    elif pixels.size == width * height * 3:
        rgb = pixels.reshape((height, width, 3))
        alpha = np.ones((height, width), dtype=np.float32)
        rgba = np.dstack((rgb, alpha))
    else:
        raise ValueError(
            f"Unexpected pixel buffer size. Got {pixels.size}, expected {width * height * 4} or {width * height * 3}."
        )

    return {
        "rgba": rgba,
        "rgb": rgb,
        "alpha": alpha,
        "width": width,
        "height": height,
    }


def build_work_mask(alpha: np.ndarray, threshold: float = 0.001):
    if alpha is None:
        raise ValueError("Alpha buffer is missing.")

    mask = (alpha > threshold).astype(np.float32)

    if np.max(mask) <= 0.0:
        mask = np.ones_like(alpha, dtype=np.float32)

    return mask


def read_mask_image_as_gray(image: bpy.types.Image, width: int, height: int) -> np.ndarray:
    if image is None:
        return None

    src = read_image_to_numpy(image)
    rgb = src["rgb"].astype(np.float32)

    # Grayscale luminance in linearized [0, 1] space.
    gray = (
        rgb[:, :, 0] * 0.2126 +
        rgb[:, :, 1] * 0.7152 +
        rgb[:, :, 2] * 0.0722
    ).astype(np.float32)

    if gray.shape[0] != height or gray.shape[1] != width:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)

    return np.clip(gray, 0.0, 1.0).astype(np.float32)
