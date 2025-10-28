from PIL import Image
import os

# Input signature path
input_path = "data/watermarks/robust.png"   # replace with your signature path
output_path = "data/output_images/signature_gray.png"

# Check if input exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Signature not found: {input_path}")

# Open and convert to grayscale
signature = Image.open(input_path).convert("L")  # "L" mode = grayscale

# Save the grayscale image
signature.save(output_path)
print(f"Grayscale signature saved at: {output_path}")

# Optional: display the image
signature.show()
