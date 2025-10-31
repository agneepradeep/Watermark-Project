# checking_robust.py

import os
from src.utils.image_utils import load_image, resize_image_to_match
from src.utils.watermark_utils import preprocess_watermark
from src.utils.verify_utils import verify_authenticity, visualize_extracted_bits
from src.utils.extract_utils import extract_robust_watermark_from_image

def check_robustness():
    original_wm_path = "data/watermarks/robust.png"
    watermarked_or_attacked_path = "data/output_images/robust_watermarked.png"
    output_dir = "data/output_images"
    os.makedirs(output_dir, exist_ok=True)
    secret_key = "my_secret_key"

    print("🔹 Loading original watermark...")
    watermark = load_image(original_wm_path)
    watermark_resized = resize_image_to_match(load_image(watermarked_or_attacked_path), watermark)
    watermark_bits, wm_shape = preprocess_watermark(watermark_resized, secret_key)

    print("🔹 Extracting watermark from possibly attacked image...")
    recovered_bits, recovered_img = extract_robust_watermark_from_image(
        watermarked_or_attacked_path,
        secret_key,
        wm_shape
    )

    print("🔹 Verifying authenticity...")
    is_authentic, similarity = verify_authenticity(watermark_bits, recovered_bits, threshold=0.8)

    print(f"\n🔍 Similarity Score: {similarity * 100:.2f}%")
    print(f"✅ Authentic: {is_authentic}")

    visualize_extracted_bits(recovered_bits, os.path.join(output_dir, "recovered_visual.png"))
    print(f"🖼️ Visualization saved to {os.path.join(output_dir, 'recovered_visual.png')}")

if __name__ == "__main__":
    check_robustness()
