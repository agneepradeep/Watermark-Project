import cv2
import numpy as np
import os

# ============================================================
# 🧩 BASIC IMAGE OPERATIONS
# ============================================================

def load_image(path):
    """Load image from path in RGB format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path, img):
    """Save image in RGB format."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def rgb_to_ycbcr(img):
    """Convert RGB image to YCbCr color space."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def ycbcr_to_rgb(img):
    """Convert YCbCr image back to RGB color space."""
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)


def split_channels_ycbcr(img):
    """Split Y, Cb, Cr channels."""
    y, cb, cr = cv2.split(img)
    return y, cb, cr


def merge_channels_ycbcr(y, cb, cr):
    """Merge Y, Cb, Cr channels back to YCbCr image."""
    return cv2.merge((y, cb, cr))

# ============================================================
# ⚙️ PREPROCESSING + RESIZING UTILITIES
# ============================================================

def resize_to_multiple_of_16(image):
    """
    Ensure image dimensions are multiples of 16.
    This helps maintain DWT/DCT compatibility for block-based processing.
    """
    h, w = image.shape[:2]
    new_h = int(np.ceil(h / 16) * 16)
    new_w = int(np.ceil(w / 16) * 16)
    if (h, w) != (new_h, new_w):
        print(f"[INFO] Resizing cover image from ({h}, {w}) → ({new_h}, {new_w})")
        image = cv2.resize(image, (new_w, new_h))
    return image


def resize_watermark(watermark, available_bits):
    """
    Resize watermark to fit embedding capacity.
    The watermark must not exceed the number of embeddable bits.
    """
    total_pixels = watermark.shape[0] * watermark.shape[1]
    if total_pixels > available_bits:
        new_side = int(np.sqrt(available_bits))
        print(f"[INFO] Resizing watermark from {watermark.shape} → ({new_side}, {new_side})")
        watermark = cv2.resize(watermark, (new_side, new_side))
    return watermark


def convert_to_grayscale(image):
    """Convert image to grayscale if it’s RGB."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def prepare_images(cover_path, watermark_path):
    """
    Load and preprocess cover & watermark images.
    Ensures:
    - Cover is RGB and dimensions are multiples of 16.
    - Watermark is grayscale and fits embedding capacity.
    """
    cover = load_image(cover_path)
    watermark = load_image(watermark_path)

    cover_rgb = resize_to_multiple_of_16(cover)
    watermark_gray = convert_to_grayscale(watermark)

    # Compute embedding capacity (based on LH + HL subbands)
    h, w = cover_rgb.shape[:2]
    sub_h, sub_w = h // 2, w // 2
    blocks_per_subband = (sub_h // 8) * (sub_w // 8)
    available_bits = blocks_per_subband * 2  # LH + HL subbands

    watermark_ready = resize_watermark(watermark_gray, available_bits)

    return cover_rgb, watermark_ready


def resize_image_to_match(base_img, img_to_resize):
    """
    Resize the watermark proportionally to the cover image size (≈30% width).
    Places the watermark at the bottom-right corner.
    """
    h_base, w_base = base_img.shape[:2]
    h_wm, w_wm = img_to_resize.shape[:2]

    # Scale watermark to 25–35% of cover image width
    target_width = int(w_base * 0.3)
    aspect_ratio = w_wm / h_wm
    target_height = int(target_width / aspect_ratio)

    resized = cv2.resize(img_to_resize, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Place resized watermark at bottom-right corner
    canvas = np.zeros_like(base_img)
    y_offset = h_base - target_height - 10
    x_offset = w_base - target_width - 10
    canvas[y_offset:y_offset + target_height, x_offset:x_offset + target_width] = resized

    return canvas
