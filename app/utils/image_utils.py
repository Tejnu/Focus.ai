"""
Image utility helpers – loading, validation, and colour-space conversions.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


def load_image(data: bytes) -> Image.Image:
    """Load a PIL image from raw bytes; raise ValueError for bad input."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image exceeds maximum allowed size of {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    if img.format and img.format.upper() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {img.format}")
    return img


def to_rgb_array(img: Image.Image) -> np.ndarray:
    """Return the image as a uint8 (H, W, 3) RGB NumPy array."""
    return np.array(img.convert("RGB"), dtype=np.uint8)


def to_grayscale_array(img: Image.Image) -> np.ndarray:
    """Return the image as a float32 (H, W) grayscale NumPy array in [0, 1]."""
    gray = img.convert("L")
    return np.array(gray, dtype=np.float32) / 255.0


def image_dimensions(img: Image.Image) -> Tuple[int, int]:
    """Return (width, height)."""
    return img.size


def resize_for_analysis(img: Image.Image, max_side: int = 512) -> Image.Image:
    """Down-scale (preserving aspect ratio) so the longer side ≤ *max_side*."""
    w, h = img.size
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img
