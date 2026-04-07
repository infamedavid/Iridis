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


def build_work_mask(alpha: np.ndarray, threshold: float = 0.001, soft_edges: bool = True):
    if alpha is None:
        raise ValueError("Alpha buffer is missing.")

    mask = (alpha > threshold).astype(np.float32)

    if np.max(mask) <= 0.0:
        mask = np.ones_like(alpha, dtype=np.float32)
        return mask

    if not soft_edges:
        return mask

    # Keep interior solid but soften the boundary transition for cleaner UV borders.
    mask_u8 = (mask > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3).astype(np.float32)
    if dist.max() <= 0.0:
        return mask

    dist /= dist.max()
    soft = np.clip(dist * 3.0, 0.0, 1.0)
    if alpha is not None:
        soft = np.minimum(soft, np.clip(alpha.astype(np.float32), 0.0, 1.0))

    return soft.astype(np.float32)
