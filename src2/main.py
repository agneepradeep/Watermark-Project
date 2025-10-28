from skimage import data, io, color
from robust_watermark import RobustWatermarker
import numpy as np

# 1) Load host image and watermark (both as numpy arrays uint8)
host = io.imread('host_image.png')      # RGB uint8
wm = io.imread('watermark.png')         # greyscale or RGB; thresholded inside code

# 2) instantiate
rw = RobustWatermarker()

# 3) choose a secret key (integer)
secret_key = 123456

# 4) Select alpha automatically (this will embed and test a bunch of alphas)
alpha_candidates = np.linspace(0.01, 0.35, 30)
alpha, metrics = rw.select_alpha(host, wm, secret_key, alpha_candidates)

print("Chosen alpha:", alpha)
print(metrics)

# 5) Final embed with selected alpha
wm_image, meta = rw.embed(host, wm, alpha, secret_key)
io.imsave('host_robust_watermarked.png', wm_image)

# 6) Extract & unscramble
extracted_bits = rw.extract(wm_image, meta)
# reshape to watermark dims if you saved original watermark shape separately
