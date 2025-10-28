import numpy as np
import cv2
from .transforms import dwt2, block_dct
from .utils import shuffle_watermark, save_image

# ---------------- Fragile Watermark Extraction ----------------
def extract_fragile_watermark(img, fragile_shape, key):
    h, w, _ = img.shape
    fragile_wm = np.zeros(fragile_shape, dtype=np.uint8)
    tamper_map = np.zeros((h//4, w//4), dtype=np.uint8)

    for ch in range(3):
        channel = img[:,:,ch]
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                block = channel[i:i+4, j:j+4]
                if block.shape != (4,4):
                    continue
                extracted_bit = block[0,0] & 0b1
                fragile_wm[i%fragile_shape[0], j%fragile_shape[1]] = extracted_bit

                # Simple tamper detection
                block_no_lsb = block & 0b11111100
                recomputed_seed = (block_no_lsb[0,0] & 0b1)
                if recomputed_seed != extracted_bit:
                    tamper_map[i//4, j//4] = 1

    fragile_wm_unscrambled = shuffle_watermark(fragile_wm, key, reverse=True)
    return fragile_wm_unscrambled, tamper_map

# ---------------- Robust Watermark Extraction ----------------
def extract_robust_watermark_grayscale(img, wm_shape, key, strength=5):
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y = ycbcr[:,:,0].astype(np.float32)

    LL, LH, HL, HH = dwt2(Y)

    extracted = []
    coeffs = [(3,4),(4,3),(4,4)]

    for subband in [LH, HL]:
        h, w = subband.shape
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = subband[i:i+8,j:j+8]
                if block.shape != (8,8) or len(extracted) >= wm_shape[0]*wm_shape[1]:
                    continue
                dct_block = block_dct(block)
                for x,y in coeffs:
                    if len(extracted) >= wm_shape[0]*wm_shape[1]:
                        break
                    left = dct_block[x,y-1]
                    right = dct_block[x,y+1]
                    # When embedding we add delta to the left coefficient and subtract
                    # delta from the right one. Therefore (left - right) contains 2*delta
                    # (plus any original difference). Embedding used:
                    #   delta = (I - 128) * (strength / 128)
                    # so to recover I we should invert that:
                    #   I = 128 + (left-right) * (128 / (2*strength))
                    # which simplifies to (64/strength) factor below.
                    intensity = 128 + ((left - right) * (64.0 / strength))
                    intensity = np.clip(intensity, 0, 255)
                    extracted.append(intensity)

    robust_wm = np.array(extracted[:wm_shape[0]*wm_shape[1]])
    robust_wm = shuffle_watermark(robust_wm, key, reverse=True)
    robust_wm_img = robust_wm.reshape(wm_shape).astype(np.uint8)
    return robust_wm_img

# ---------------- Full Decoding Pipeline ----------------
def decode_watermark(watermarked_img, robust_shape, fragile_shape, key, strength=5):
    fragile_wm, tamper_map = extract_fragile_watermark(watermarked_img, fragile_shape, key)
    robust_wm = extract_robust_watermark_grayscale(watermarked_img, robust_shape, key, strength)

    # Save extracted robust watermark
    save_image("data/output_images/extracted_robust.png", robust_wm)

    return fragile_wm, tamper_map, robust_wm
