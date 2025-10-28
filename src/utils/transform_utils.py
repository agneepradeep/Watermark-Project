# src/utils/transform_utils.py

import pywt
import cv2
import numpy as np
from src.config.settings import WAVELET_TYPE, BLOCK_SIZE

def dwt2(image):
    return pywt.dwt2(image, WAVELET_TYPE)

def idwt2(coeffs):
    return pywt.idwt2(coeffs, WAVELET_TYPE)

def blockwise_dct(block):
    return cv2.dct(np.float32(block))

def blockwise_idct(block):
    return cv2.idct(np.float32(block))

def split_into_blocks(channel, block_size=BLOCK_SIZE):
    h, w = channel.shape
    return [
        channel[i:i+block_size, j:j+block_size]
        for i in range(0, h, block_size)
        for j in range(0, w, block_size)
        if i+block_size <= h and j+block_size <= w
    ]
