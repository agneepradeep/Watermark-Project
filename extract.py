from src.utils.extract_utils import extract_robust_watermark_from_image

watermarked_path = "data/output_images/robust_watermarked.png"   # result from embedding
secret_key = "my_secret_key"
wm_shape = (32, 32)   # the shape you used when embedding (height, width)

recovered = extract_robust_watermark_from_image(watermarked_path, secret_key, wm_shape)
# Save the recovered watermark for inspection:
import cv2
cv2.imwrite("data/output_images/recovered_robust.png", recovered)
print("Recovered watermark saved to data/output_images/recovered_robust.png")
