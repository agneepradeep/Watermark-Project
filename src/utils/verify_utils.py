import numpy as np
import cv2
import os

def compute_similarity(original_bits, extracted_bits):
    """
    Compare two watermark bit sequences and return a similarity score.
    """
    min_len = min(len(original_bits), len(extracted_bits))
    original_bits = np.array(original_bits[:min_len])
    extracted_bits = np.array(extracted_bits[:min_len])

    matches = np.sum(original_bits == extracted_bits)
    similarity = matches / min_len
    return similarity


def verify_authenticity(original_bits, extracted_bits, threshold=0.8):
    """
    Verify watermark authenticity based on bit similarity.
    Returns True if similarity >= threshold.
    """
    similarity = compute_similarity(original_bits, extracted_bits)
    is_authentic = similarity >= threshold

    print(f"\n🔍 Watermark Verification Result:")
    print(f"→ Similarity Score: {similarity * 100:.2f}%")
    print(f"→ Authenticity Status: {'✅ AUTHENTIC' if is_authentic else '❌ NOT AUTHENTIC'}")

    return is_authentic, similarity


def visualize_extracted_bits(extracted_bits, output_path="outputs/extracted_visual.png"):
    """
    Optional: Visualize the extracted watermark bits as an image for debugging.
    This will appear as a noisy black-and-white pattern (not a readable watermark).
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert 1D bits to 2D visual grid
    length = len(extracted_bits)
    size = int(np.sqrt(length))
    visual = np.array(extracted_bits[:size*size]).reshape((size, size)).astype(np.uint8) * 255

    cv2.imwrite(output_path, visual)
    print(f"🖼️ Extracted watermark visualization saved to: {output_path}")
