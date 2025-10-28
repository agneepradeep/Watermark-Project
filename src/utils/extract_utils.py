# src/utils/extract_utils.py

import numpy as np
import pywt
from scipy.fftpack import dct
from src.utils.image_utils import load_image, rgb_to_ycbcr, split_channels_ycbcr
from src.utils.watermark_utils import unshuffle_bits_by_key
from src.config.settings import BLOCK_SIZE


# ---------- Helper DCT 2D ----------
def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


# ---------- Main Extraction ----------
def extract_robust_watermark_from_image(watermarked_img, secret_key, wm_shape=None):
    """
    Extract robust watermark bits from a watermarked image.

    Args:
        watermarked_img: numpy array (RGB) or path (str) to watermarked image
        secret_key: key used during embedding (for deterministic shuffle)
        wm_shape: tuple (H, W) of watermark shape; if None, auto-estimates based on capacity

    Returns:
        recovered_bits (flat array)
        recovered_image (2D grayscale watermark 0–255)
    """

    # --- Accept both path and array ---
    if isinstance(watermarked_img, str):
        img = load_image(watermarked_img)
    else:
        img = watermarked_img

    # --- Convert to YCbCr and extract Y ---
    ycbcr = rgb_to_ycbcr(img)
    Y, _, _ = split_channels_ycbcr(ycbcr)
    Y = Y.astype(np.float32)

    # --- Perform 1-level DWT ---
    LL, (LH, HL, HH) = pywt.dwt2(Y, "haar")

    # --- Block traversal setup ---
    bh = BLOCK_SIZE
    nh = LH.shape[0] // bh
    nw = LH.shape[1] // bh
    capacity = nh * nw * 2  # LH + HL

    mid = bh // 2
    posA, posB = (mid - 1, mid), (mid, mid - 1)

    bits = []

    # --- Extract bits from LH + HL ---
    for i in range(nh):
        for j in range(nw):
            block_lh = LH[i * bh:(i + 1) * bh, j * bh:(j + 1) * bh]
            block_hl = HL[i * bh:(i + 1) * bh, j * bh:(j + 1) * bh]

            dct_lh = dct2(block_lh)
            dct_hl = dct2(block_hl)

            try:
                cL1, cR1 = dct_lh[posA], dct_lh[posB]
                cL2, cR2 = dct_hl[posA], dct_hl[posB]
            except Exception:
                continue  # skip problematic blocks

            bit1 = 1 if abs(cL1) > abs(cR1) else 0
            bit2 = 1 if abs(cL2) > abs(cR2) else 0

            # majority decision
            if bit1 == bit2:
                bits.append(bit1)
            else:
                diff1 = abs(abs(cL1) - abs(cR1))
                diff2 = abs(abs(cL2) - abs(cR2))
                bits.append(bit1 if diff1 >= diff2 else bit2)

    bits = np.array(bits, dtype=np.uint8)

    # --- Determine expected watermark shape automatically ---
    if wm_shape is None:
        # Try to approximate square watermark
        length = int(np.floor(np.sqrt(bits.size)))
        wm_shape = (length, length)
        print(f"[ℹ] Auto-inferred watermark shape: {wm_shape}")

    total_bits_needed = wm_shape[0] * wm_shape[1]
    if bits.size < total_bits_needed:
        print(f"[⚠] Warning: Extracted bits ({bits.size}) < required ({total_bits_needed}). Padding with zeros.")
        bits = np.pad(bits, (0, total_bits_needed - bits.size), mode='constant')

    bits = bits[:total_bits_needed]

    # --- Unshuffle the extracted bits using secret key ---
    unshuffled = unshuffle_bits_by_key(bits, secret_key)

    # --- Reshape into watermark image ---
    recovered_image = (unshuffled.reshape(wm_shape) * 255).astype(np.uint8)

    return unshuffled, recovered_image
