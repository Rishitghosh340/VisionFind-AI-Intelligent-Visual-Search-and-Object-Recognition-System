"""Image loading / conversion helpers (robust to EXIF rotation and PNG alpha)."""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_pil(file_or_bytes):
    """Open an upload/bytes as an RGB PIL image (EXIF-rotated, transparency on white)."""
    if isinstance(file_or_bytes, (bytes, bytearray)):
        file_or_bytes = io.BytesIO(file_or_bytes)
    img = Image.open(file_or_bytes)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        img = bg
    return img.convert("RGB")


def pil_to_bgr(pil_image):
    rgb = np.asarray(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image):
    if image is None or image.size == 0:
        raise ValueError("Image is empty.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def limit_size(image, max_side):
    """Downscale so the longest side <= max_side. Returns (image, scale_factor)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    s = max_side / float(longest)
    out = cv2.resize(image, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                     interpolation=cv2.INTER_AREA)
    return out, s


def encode_png(image_bgr):
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Could not encode image.")
    return buf.tobytes()


def crop(image, box, pad=0):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    x2, y2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
    return image[y1:y2, x1:x2]
