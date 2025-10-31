## 🖼️ Robust Digital Image Watermarking System

A **modular Python project** implementing a **robust digital image watermarking system**, designed for **embedding**, **extracting**, and **authenticity checking** of watermarks in images.
This system uses **frequency-domain (DCT-based)** embedding within the **YCbCr color space** to ensure robustness against mild compression, noise, and filtering.

---

### 📁 Project Structure

```
Watermark Project/
│
├── data/
│   ├── input_images/          # Original (cover) image
│   ├── watermarks/            # Watermark/signature images
│   └── output_images/         # Outputs: watermarked, recovered, visualized images
│
├── src/
│   └── utils/                 # Utility modules for image, DCT, and verification
│       ├── image_utils.py
│       ├── watermark_utils.py
│       ├── embed_utils.py
│       ├── extract_utils.py
│       └── verify_utils.py
│
├── embedding_robust.py        # Embed the watermark
├── extracting_robust.py       # Extract the watermark
├── checking_robust.py         # Check watermark authenticity/robustness
├── main.py                    # Runs all 3 processes sequentially
├── requirements.txt           # Dependency list
└── README.md
```

---

### ⚙️ Setup Instructions

#### **1. Install dependencies**

Make sure Python ≥ 3.8 is installed, then run:

```bash
pip install -r requirements.txt
```

*(This installs `numpy`, `opencv-python`, and `pillow` automatically.)*

---

#### **2. Add input files**

* Place your **cover image** (e.g., `cover.png`) in:
  `data/input_images/`
* Place your **watermark image** (e.g., `robust.png`) in:
  `data/watermarks/`

---

### 🧩 Running the Project

You can **run each module separately** or execute the **entire pipeline** at once.

#### 🧱 Option 1 — Run modules manually

**Step 1:** Embed the watermark

```bash
python embedding_robust.py
```

**Step 2:** Extract the watermark

```bash
python extracting_robust.py
```

**Step 3:** Check authenticity

```bash
python checking_robust.py
```

---

#### ⚡ Option 2 — Run everything in one go

To automate all three steps (embed → extract → check):

```bash
python main.py
```

This sequentially:

1. Embeds the watermark
2. Extracts it from the generated image
3. Checks its authenticity

---

### 🧠 Algorithm Overview

#### **Embedding Process**

1. Convert the cover image from **RGB → YCbCr** and extract the Y channel.
2. Convert watermark to **binary bits** and **shuffle** them using a secret key.
3. Perform **DCT (Discrete Cosine Transform)** on 8×8 image blocks.
4. Modify mid-frequency DCT coefficients slightly to hide the bits.
5. Apply **Inverse DCT** to get the watermarked Y channel.
6. Merge Y, Cb, Cr → RGB and save the **watermarked image**.

#### **Extraction Process**

1. Convert the **watermarked image** to YCbCr.
2. Apply DCT on each 8×8 block and read embedded bit patterns.
3. Reverse the bit shuffling using the **same secret key**.
4. Reconstruct the watermark image.

#### **Authenticity Check**

* Compute the **bit similarity** (cosine correlation) between the original and extracted watermark.
* If similarity ≥ threshold (e.g., 80%), the image is considered **authentic**.

---

### 📊 Example Output

```
🔹 Loading images...
🔹 Embedding robust watermark...
✅ Watermarked image saved at: data/output_images/robust_watermarked.png
📝 Watermark shape saved to wm_shape.txt

🔹 Extracting watermark from image...
✅ Recovered watermark saved at data/output_images/recovered_robust.png

🔹 Verifying authenticity...
→ Similarity Score: 82.45%
→ Authenticity Status: ✅ AUTHENTIC
```

---

### ⚠️ Current Limitations

Although functional, the current implementation has several **technical limitations**:

1. **Resizing and shape mismatch**

   * If the watermark dimensions differ greatly from the cover, extracted bits may not reshape correctly (e.g., `(1554 vs 13680)` size error).

2. **Low resistance to heavy attacks**

   * Severe noise, cropping, scaling, or rotation cause major degradation.

3. **Static embedding strength**

   * Coefficient modification strength is fixed and not adaptive to image features.

4. **Only grayscale watermark supported**

   * RGB or color watermark embedding not supported in this version.

5. **Similarity fluctuation even without attacks**

   * Floating-point precision during DCT/IDCT can slightly alter recovery.

6. **Dependent on secret key and watermark shape**

   * Extraction fails without the correct key and shape file (`wm_shape.txt`).

---

### 🚀 Future Enhancements

* Add **adaptive embedding strength** (based on texture or luminance).
* Introduce **error correction codes (ECC)** for better recovery.
* Support **color watermarks**.
* Improve robustness against **geometric transformations**.
* Add a **blind extraction mode** (no original watermark required).
