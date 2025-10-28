import numpy as np
import cv2
from .transforms import dwt2, idwt2, block_dct, block_idct
from .utils import shuffle_watermark

# ---------------- Robust Watermark Embedding ----------------
def embed_robust_watermark(img, watermark, key, strength=5):
    """
    Embed grayscale robust watermark into image using DWT + DCT
    img: RGB image (H,W,3)
    watermark: grayscale watermark (0-255)
    key: secret key for shuffling
    strength: embedding factor
    """
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y = ycbcr[:,:,0].astype(np.float32)

    # Shuffle watermark
    wm_flat = watermark.flatten()
    wm_flat = shuffle_watermark(wm_flat, key)

    # DWT
    LL, LH, HL, HH = dwt2(Y)

    # Embed into LH and HL
    wm_index = 0
    coeffs = [(3,4),(4,3),(4,4)]

    for subband in [LH, HL]:
        h, w = subband.shape
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = subband[i:i+8,j:j+8]
                if block.shape != (8,8) or wm_index >= len(wm_flat):
                    continue
                dct_block = block_dct(block)
                # Modify coefficients proportionally
                for x,y in coeffs:
                    if wm_index >= len(wm_flat):
                        break
                    intensity = wm_flat[wm_index]  # 0-255
                    delta = (intensity - 128) * (strength / 128)
                    dct_block[x,y-1] += delta
                    dct_block[x,y+1] -= delta
                    wm_index += 1
                block_mod = block_idct(dct_block)
                subband[i:i+8,j:j+8] = block_mod

    # Reconstruct Y channel
    Y_mod = idwt2(LL, LH, HL, HH)
    ycbcr[:,:,0] = np.clip(Y_mod,0,255).astype(np.uint8)
    robust_watermarked_img = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
    return robust_watermarked_img

# ---------------- Fragile Watermark Embedding ----------------
def embed_fragile_watermark(img, fragile_wm, key):
    """
    Embed fragile watermark (tamper detection) in LSBs
    fragile_wm: binary image (0/1)
    """
    wm_bin = (fragile_wm > 0).astype(np.uint8)
    wm_shuffled = shuffle_watermark(wm_bin, key)

    output = img.copy()
    h, w = img.shape[:2]

    for ch in range(3):  # R,G,B
        channel = output[:,:,ch]
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                block = channel[i:i+4, j:j+4]
                if block.shape != (4,4):
                    continue
                # Strip LSBs
                block = (block & 0b11111100)
                # Embed one bit from shuffled watermark
                bit = wm_shuffled[i % wm_shuffled.shape[0], j % wm_shuffled.shape[1]]
                block[0,0] |= bit
                channel[i:i+4, j:j+4] = block
        output[:,:,ch] = channel
    return output

# ---------------- Full Encoding Pipeline ----------------
def encode_watermark(original_img, robust_wm, fragile_wm, key, strength=5):
    """
    Full pipeline: robust + fragile watermark embedding
    """
    img1 = embed_robust_watermark(original_img, robust_wm, key, strength)
    img2 = embed_fragile_watermark(img1, fragile_wm, key)
    return img2
