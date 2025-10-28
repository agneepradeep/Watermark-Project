import cv2
from src1.utils import read_image, save_image
from src1.embed import encode_watermark

if __name__ == "__main__":
    cover_path = r"D:\Watermark Project\data\input_images\cover.png"
    robust_path = r"D:\Watermark Project\data\watermarks\robust.png"
    fragile_path = r"D:\Watermark Project\data\watermarks\fragile.png"
    output_path = r"D:\Watermark Project\data\output_images\final_watermarked.png"

    # Read images
    cover = read_image(cover_path)
    robust_wm = read_image(robust_path)[:,:,0]  # use single channel
    fragile_wm = read_image(fragile_path)[:,:,0]

    # Encode watermark
    final_img = encode_watermark(cover, robust_wm, fragile_wm, key=1234, strength=5)

    # Save result
    save_image(output_path, final_img)

    print(f"Watermarked image saved at {output_path}")
