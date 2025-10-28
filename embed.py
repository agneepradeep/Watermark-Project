import os
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

def main():
    # --- File Paths ---
    original_path = "data/input_images/cover.png"
    watermark_path = "data/watermarks/robust.png"
    output_dir = "data/output_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "robust_watermarked.png")

    secret_key = "my_secret_key"

    print("🔹 Loading images...")
    original = load_image(original_path)
    watermark = load_image(watermark_path)

    # --- Step 1: Resize images (cover + watermark) ---
    print("🔹 Resizing watermark to fit cover image...")
    watermark_resized = resize_image_to_match(original, watermark)

    # --- Step 2: Convert cover to YCbCr and split ---
    ycbcr = rgb_to_ycbcr(original)
    y, cb, cr = split_channels_ycbcr(ycbcr)

    # --- Step 3: Prepare watermark bits ---
    print("🔹 Preparing watermark bits...")
    watermark_bits, _ = preprocess_watermark(watermark_resized, secret_key)

    # --- Step 4: Embed robust watermark ---
    print("🔹 Embedding robust watermark...")
    watermarked_y = embed_robust_watermark(y, watermark_bits)

    # --- Step 5: Reconstruct the final image ---
    print("🔹 Reconstructing final image...")
    watermarked_ycbcr = merge_channels_ycbcr(watermarked_y, cb, cr)
    watermarked_rgb = ycbcr_to_rgb(watermarked_ycbcr)

    # --- Step 6: Save the final image ---
    print("🔹 Saving output...")
    save_image(output_path, watermarked_rgb)
    print(f"✅ Watermarked image saved at: {output_path}")

if __name__ == "__main__":
    main()
