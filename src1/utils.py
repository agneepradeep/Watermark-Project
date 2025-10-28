import numpy as np
import cv2
from PIL import Image
import os


def read_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image (check file integrity): {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(path, image):
    """Save RGB image using PIL"""
    im = Image.fromarray(image.astype(np.uint8))
    im.save(path)


def shuffle_watermark(watermark, key, reverse=False):
    """
    Shuffle or unscramble watermark using a secret key.
    If reverse=True, it unscrambles using the same key.
    """
    flat = watermark.flatten()
    np.random.seed(key)
    idx = np.arange(len(flat))
    np.random.shuffle(idx)

    if reverse:
        # Create inverse mapping to unscramble
        inv_idx = np.zeros_like(idx)
        inv_idx[idx] = np.arange(len(idx))
        flat = flat[inv_idx]
    else:
        flat = flat[idx]

    return flat.reshape(watermark.shape)
