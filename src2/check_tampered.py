import cv2
import numpy as np

# ---- Load images ----
# path to your original (embedded) watermark/signature image
original_watermark = cv2.imread('watermark.png', cv2.IMREAD_GRAYSCALE)
# path to the extracted watermark
extracted_watermark = cv2.imread('extracted_robust_watermark.png', cv2.IMREAD_GRAYSCALE)

# resize both to same shape (in case of mismatch)
extracted_watermark = cv2.resize(extracted_watermark, (original_watermark.shape[1], original_watermark.shape[0]))

# normalize to [0,1]
orig_norm = original_watermark.astype(np.float32) / 255.0
ext_norm = extracted_watermark.astype(np.float32) / 255.0

# ---- Compute Normalized Correlation (NC) ----
numerator = np.sum(orig_norm * ext_norm)
denominator = np.sqrt(np.sum(orig_norm**2) * np.sum(ext_norm**2))
NC = numerator / denominator

# ---- Compute Bit Error Rate (BER) ----
# if your watermark is binary (0/1)
orig_bin = (orig_norm > 0.5).astype(np.uint8)
ext_bin = (ext_norm > 0.5).astype(np.uint8)
BER = np.sum(orig_bin != ext_bin) / orig_bin.size

# ---- Decision ----
print(f"Normalized Correlation (NC): {NC:.4f}")
print(f"Bit Error Rate (BER): {BER:.4f}")

if NC > 0.75:
    print("✅ Watermark detected — Image likely NOT tampered.")
elif NC > 0.5:
    print("⚠️ Possible mild distortion — watermark partially intact.")
else:
    print("❌ Watermark may be TAMPERED or destroyed.")
