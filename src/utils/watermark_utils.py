# src/utils/watermark_utils.py

import numpy as np
import cv2
from random import Random
from src.config.settings import RANDOM_SEED


def preprocess_watermark(watermark_input, secret_key):
    """
    Preprocess the watermark image:
      - Accepts either a file path (str) or a loaded NumPy array
      - Converts to grayscale if needed
      - Binarizes (0/1)
      - Flattens and shuffles bits using the secret key
    Returns:
      shuffled_bits, original_shape
    """
    # --- Handle both path and array inputs ---
    if isinstance(watermark_input, str):
        img = cv2.imread(watermark_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Watermark not found at {watermark_input}")
    elif isinstance(watermark_input, np.ndarray):
        if len(watermark_input.shape) == 3:  # RGB → Grayscale
            img = cv2.cvtColor(watermark_input, cv2.COLOR_RGB2GRAY)
        else:
            img = watermark_input
    else:
        raise TypeError("❌ Invalid watermark input: must be file path or numpy array")

    # --- Convert to binary (0/1) ---
    binary = (img > 128).astype(np.uint8).flatten()

    # --- Shuffle bits deterministically ---
    shuffled, _ = shuffle_bits(binary, secret_key)

    return shuffled, img.shape


def shuffle_bits(bits, key):
    """
    Shuffle bits based on a secret key for reproducibility.
    """
    rnd = Random(hash(key) + RANDOM_SEED)
    indices = list(range(len(bits)))
    rnd.shuffle(indices)
    shuffled = bits[indices]
    return shuffled, indices


def unshuffle_bits(bits, indices):
    """
    Reverse the shuffling of bits (given explicit indices).
    """
    unshuffled = np.zeros_like(bits)
    unshuffled[indices] = bits
    return unshuffled


def get_shuffle_indices(length: int, key) -> list:
    """
    Deterministically produce the shuffle permutation indices for `length`
    using the provided secret key (same algorithm used by shuffle_bits).
    """
    rnd = Random(hash(key) + RANDOM_SEED)
    indices = list(range(length))
    rnd.shuffle(indices)
    return indices


def unshuffle_bits_by_key(bits: np.ndarray, key) -> np.ndarray:
    """
    Unshuffle flattened bits using only the secret key (no indices needed).
    """
    indices = get_shuffle_indices(bits.size, key)
    inv = np.empty_like(indices)
    for i, p in enumerate(indices):
        inv[p] = i
    return bits[inv]
