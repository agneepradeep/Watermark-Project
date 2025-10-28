import cv2
import numpy as np
import pywt

# -----------------------------
# Function: Extract Robust Watermark (Corrected)
# -----------------------------
def extract_robust_watermark(dual_watermarked_image_path, watermark_shape, key, output_path="extracted_watermark.png"):
    # Step 1: Load the dual watermarked image
    img = cv2.imread(dual_watermarked_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cb, Cr = cv2.split(img)

    # Step 2: Apply 1st-level DWT on Y channel
    coeffsY = pywt.dwt2(Y, 'haar')
    LL_Y, (LH_Y, HL_Y, HH_Y) = coeffsY

    # Step 3: Define subbands for extraction (LH and HL)
    subbands = [LH_Y, HL_Y]
    watermark_bits = []

    for band in subbands:
        h, w = band.shape
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = band[i:i+8, j:j+8]
                if block.shape != (8, 8):
                    continue

                # Apply DCT on each 8x8 block
                dct_block = cv2.dct(np.float32(block))

                # Step 4: Extract Embedding Blocks (EB-1 and EB-2)
                EB = dct_block[0:2, 0:4]  # 2x4 embedding block
                EB1 = EB
                EB2 = EB

                # Step 5: Split vertically into LHS and RHS
                EB1_LHS = EB1[:, 0:2]
                EB1_RHS = EB1[:, 2:4]
                EB2_LHS = EB2[:, 0:2]
                EB2_RHS = EB2[:, 2:4]

                # Step 6: Extract max coefficients
                EB1_LHS_max = np.max(EB1_LHS)
                EB1_RHS_max = np.max(EB1_RHS)
                EB2_LHS_max = np.max(EB2_LHS)
                EB2_RHS_max = np.max(EB2_RHS)

                # Step 7: Compare and extract watermark bits
                bit1 = 1 if EB1_LHS_max > EB1_RHS_max else 0
                bit2 = 1 if EB2_LHS_max > EB2_RHS_max else 0

                watermark_bits.extend([bit1, bit2])

    # Step 8: Reshape extracted bits into image
    total_bits = watermark_shape[0] * watermark_shape[1]
    watermark_bits = np.array(watermark_bits[:total_bits])
    watermark_extracted = watermark_bits.reshape(watermark_shape)

    # Step 9: Correct Unscrambling using inverse permutation
    np.random.seed(key)
    indices = np.arange(total_bits)
    np.random.shuffle(indices)

    inverse_indices = np.argsort(indices)
    unscrambled = watermark_extracted.flatten()[inverse_indices]
    watermark_unscrambled = unscrambled.reshape(watermark_shape)

    # Step 10: Post-process for better visibility
    watermark_img = np.uint8(watermark_unscrambled * 255)
    watermark_img = cv2.GaussianBlur(watermark_img, (3, 3), 0)

    # Step 11: Save and display
    cv2.imwrite(output_path, watermark_img)
    print(f"✅ Watermark extracted and saved as: {output_path}")

    cv2.imshow("Extracted Watermark", watermark_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return watermark_img


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    extracted_wm = extract_robust_watermark(
        dual_watermarked_image_path="host_robust_watermarked.png",
        watermark_shape=(64, 64),
        key=1234,
        output_path="extracted_robust_watermark.png"
    )
