import pywt
import numpy as np
import cv2

# --- Wavelet Transform ---
def dwt2(image):
    coeff2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeff2
    return LL, LH, HL, HH

def idwt2(LL,LH, HL, HH):
    coeff2 = LL, (LH, HL, HH)
    return pywt.idwt2(coeff2, 'haar')

# --- Block DCT / IDCT ---
def block_dct(block):
    return cv2.dct(block.astype(np.float32))

def block_idct(block):
    return cv2.idct(block.astype(np.float32))
