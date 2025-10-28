from src1.utils import read_image, save_image
from src1.extract import decode_watermark

if __name__ == "__main__":
    watermarked_path = r"D:\Watermark Project\data\output_images\final_watermarked.png"
    watermarked_img = read_image(watermarked_path)

    # Shapes of watermarks (height, width)
    robust_shape = (64, 64)   # example, same as embedded robust watermark
    fragile_shape = (64, 64)  # example, same as embedded fragile watermark

    key = 1234

    fragile_wm, tamper_map, robust_wm = decode_watermark(watermarked_img, robust_shape, fragile_shape, key)

    # Save fragile watermark and tamper map
    save_image(r"D:\Watermark Project\data/output_images/extracted_fragile.png", fragile_wm*255)
    save_image(r"D:\Watermark Project\data/output_images/tamper_map.png", tamper_map*255)

    print("Decoding completed. Extracted watermarks and tamper map saved.")
