import numpy as np
from src.utils.transform_utils import dwt2, idwt2, blockwise_dct, blockwise_idct
from src.config.settings import STRENGTH_FACTOR, BLOCK_SIZE

def embed_robust_watermark(y_channel: np.ndarray, watermark_bits: np.ndarray) -> np.ndarray:
    """
    Embed the robust (shuffled) watermark bits into the Y-channel
    of the cover image using 1-level DWT and blockwise DCT.

    Steps:
    1. Apply 1-level DWT on Y-channel to get LL, LH, HL, HH.
    2. Embed bits into LH and HL subbands using mid-frequency DCT coefficients.
    3. Perform inverse DWT to reconstruct the watermarked Y-channel.

    Args:
        y_channel: 2D array of Y (luminance) component of cover image.
        watermark_bits: Flattened shuffled binary watermark bits (0/1).

    Returns:
        np.ndarray: Watermarked Y-channel (uint8).
    """
    coeffs2 = dwt2(y_channel)
    LL, (LH, HL, HH) = coeffs2

    LH_embedded = _embed_in_subband(LH, watermark_bits)
    HL_embedded = _embed_in_subband(HL, watermark_bits)

    watermarked_y = idwt2((LL, (LH_embedded, HL_embedded, HH)))
    return np.uint8(np.clip(watermarked_y, 0, 255))


def _embed_in_subband(subband: np.ndarray, watermark_bits: np.ndarray) -> np.ndarray:
    """
    Embed watermark bits in the given DWT subband (LH or HL) using DCT.

    Each 8×8 block contributes one watermark bit:
      - Extract 2 mid-frequency coefficients.
      - Compare c1, c2 and modify according to the bit value (Eq.7 & Eq.8 style).
    """
    h, w = subband.shape
    block_size = BLOCK_SIZE
    idx = 0

    modified = subband.copy()

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if idx >= len(watermark_bits):
                break

            if i + block_size <= h and j + block_size <= w:
                block = subband[i:i+block_size, j:j+block_size]
                dct_block = blockwise_dct(block)

                mid = block_size // 2
                c1, c2 = dct_block[mid-1, mid], dct_block[mid, mid-1]
                bit = watermark_bits[idx]

                # Apply robust embedding rule (c1 vs c2)
                if bit == 1 and c1 <= c2:
                    c1, c2 = c2 + STRENGTH_FACTOR, c1
                elif bit == 0 and c1 > c2:
                    c2, c1 = c1 + STRENGTH_FACTOR, c2

                dct_block[mid-1, mid], dct_block[mid, mid-1] = c1, c2
                modified[i:i+block_size, j:j+block_size] = blockwise_idct(dct_block)
                idx += 1

    return modified
