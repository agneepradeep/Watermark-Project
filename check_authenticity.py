from src.utils.verify_utils import verify_authenticity, visualize_extracted_bits

# Example: compare and verify
is_authentic, similarity = verify_authenticity(original_watermark_bits, extracted_watermark_bits)

# Optional: visualize the extracted bits
visualize_extracted_bits(extracted_watermark_bits)
