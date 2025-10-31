# embedding_robust.py

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

def embed_robust():
    original_path = "data/input_images/cover.png"
    watermark_path = "data/watermarks/robust.png"
    output_dir = "data/output_images"
    os.makedirs(output_dir, exist_ok=True)
    watermarked_path = os.path.join(output_dir, "robust_watermarked.png")

    secret_key = "my_secret_key"

    print("🔹 Loading images...")
    original = load_image(original_path)
    watermark = load_image(watermark_path)

    print("🔹 Resizing watermark to fit cover image...")
    watermark_resized = resize_image_to_match(original, watermark)

    print("🔹 Preparing watermark bits...")
    watermark_bits, wm_shape = preprocess_watermark(watermark_resized, secret_key)

    print("🔹 Embedding robust watermark...")
    ycbcr = rgb_to_ycbcr(original)
    y, cb, cr = split_channels_ycbcr(ycbcr)
    watermarked_y = embed_robust_watermark(y, watermark_bits)

    print("🔹 Reconstructing final image...")
    watermarked_ycbcr = merge_channels_ycbcr(watermarked_y, cb, cr)
    watermarked_rgb = ycbcr_to_rgb(watermarked_ycbcr)

    print("🔹 Saving watermarked image...")
    save_image(watermarked_path, watermarked_rgb)
    print(f"✅ Watermarked image saved at: {watermarked_path}")

    # Save watermark shape for later use
    with open(os.path.join(output_dir, "wm_shape.txt"), "w") as f:
        f.write(f"{wm_shape[0]},{wm_shape[1]}")
    print("📝 Watermark shape saved to wm_shape.txt")

if __name__ == "__main__":
    embed_robust()
