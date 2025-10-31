# extracting_robust.py

import os
import cv2
from src.utils.extract_utils import extract_robust_watermark_from_image

def extract_robust():
    watermarked_path = "data/output_images/robust_watermarked.png"  # or attacked version
    output_dir = "data/output_images"
    os.makedirs(output_dir, exist_ok=True)
    recovered_path = os.path.join(output_dir, "recovered_robust.png")
    secret_key = "my_secret_key"

    # Read saved shape
    wm_shape_file = os.path.join(output_dir, "wm_shape.txt")
    if os.path.exists(wm_shape_file):
        with open(wm_shape_file, "r") as f:
            shape_str = f.read().strip().split(",")
            wm_shape = (int(shape_str[0]), int(shape_str[1]))
    else:
        wm_shape = None

    print("🔹 Extracting watermark from image...")
    recovered_bits, recovered_img = extract_robust_watermark_from_image(
        watermarked_path,
        secret_key,
        wm_shape
    )
    cv2.imwrite(recovered_path, recovered_img)
    print(f"✅ Recovered watermark saved at: {recovered_path}")

if __name__ == "__main__":
    extract_robust()
