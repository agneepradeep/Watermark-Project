import os
import cv2
from src.utils.image_utils import (
    load_image,
    save_image,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    split_channels_ycbcr,
    merge_channels_ycbcr,
    resize_image_to_match
)
from src.utils.watermark_utils import preprocess_watermark
from src.utils.embed_utils import embed_robust_watermark
from src.utils.extract_utils import extract_robust_watermark_from_image
from src.utils.verify_utils import verify_authenticity, visualize_extracted_bits


def main():
    # --- File Paths ---
    original_path = "data/input_images/cover.png"
    watermark_path = "data/watermarks/robust.png"
    output_dir = "data/output_images"
    os.makedirs(output_dir, exist_ok=True)
    watermarked_path = os.path.join(output_dir, "robust_watermarked.png")
    recovered_path = os.path.join(output_dir, "recovered_robust.png")

    # --- Key & Settings ---
    secret_key = "my_secret_key"

    print("🔹 Loading images...")
    original = load_image(original_path)
    watermark = load_image(watermark_path)

    # --- Step 1: Resize watermark to fit cover image ---
    print("🔹 Resizing watermark to fit cover image...")
    watermark_resized = resize_image_to_match(original, watermark)

    # --- Step 2: Convert cover to YCbCr and split ---
    ycbcr = rgb_to_ycbcr(original)
    y, cb, cr = split_channels_ycbcr(ycbcr)

    # --- Step 3: Prepare watermark bits ---
    print("🔹 Preparing watermark bits...")
    watermark_bits, wm_shape = preprocess_watermark(watermark_resized, secret_key)

    # --- Step 4: Embed robust watermark ---
    print("🔹 Embedding robust watermark...")
    watermarked_y = embed_robust_watermark(y, watermark_bits)

    # --- Step 5: Reconstruct final image ---
    print("🔹 Reconstructing final image...")
    watermarked_ycbcr = merge_channels_ycbcr(watermarked_y, cb, cr)
    watermarked_rgb = ycbcr_to_rgb(watermarked_ycbcr)

    # --- Step 6: Save watermarked image ---
    print("🔹 Saving watermarked image...")
    save_image(watermarked_path, watermarked_rgb)
    print(f"✅ Watermarked image saved at: {watermarked_path}")

    # --- Step 7: Extraction Process (updated) ---
    print("\n🔹 Extracting watermark from image...")
    recovered_bits, recovered_img = extract_robust_watermark_from_image(
        watermarked_path,
        secret_key,
        wm_shape=None  # auto-detect or specify manually like (32, 32)
    )
    cv2.imwrite("data/output_images/recovered_robust.png", recovered_img)
    print("✅ Recovered watermark saved at data/output_images/recovered_robust.png")

    # --- Step 8: Authenticity Verification ---
    print("\n🔹 Verifying authenticity of watermark...")
    is_authentic, similarity = verify_authenticity(watermark_bits, recovered_bits, threshold=0.8)

    # --- Step 9: Visualize Extracted Watermark (Optional) ---
    visualize_extracted_bits(recovered_bits, os.path.join(output_dir, "recovered_visual.png"))

    print("\n🎯 Process completed successfully!")


if __name__ == "__main__":
    main()
