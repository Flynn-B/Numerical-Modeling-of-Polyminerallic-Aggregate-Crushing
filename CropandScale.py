"""
    Code crops and scales input images (mineralogy maps) to fit square. Necessary for input into MCRpy code
    - FB
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# === CONFIGURABLE TARGET OUTPUT SIZE ===
TARGET_SIZE = 64 # Change this to 32, 64,128, 256, etc.

# --- Load and process image ---
input_path = "Important\96x96Expansion.png"

# --- Save image ---
save_path = "Important\96x96ExpansionShrunkto32x32.png"
save_image = False

# Load image (BGR)
img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"Image not found at path: {input_path}")

def crop_to_square(image):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return image[top:top+min_dim, left:left+min_dim]

def dominant_pixel(block):
    pixels = block.reshape(-1, block.shape[-1])
    counter = Counter(map(tuple, pixels))
    return np.array(counter.most_common(1)[0][0], dtype=np.uint8)

def shrink_image_dominant(image, target_size):
    h, w = image.shape[:2]
    block_size = h // target_size
    new_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    for i in range(target_size):
        for j in range(target_size):
            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            new_img[i, j] = dominant_pixel(block)

    return new_img

# Convert to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Crop to square and resize to multiple of TARGET_SIZE
square_img = crop_to_square(img)
adjusted_size = (square_img.shape[0] // TARGET_SIZE) * TARGET_SIZE
square_img_resized = cv2.resize(square_img, (adjusted_size, adjusted_size), interpolation=cv2.INTER_NEAREST)

# Shrink using dominant-pixel sampling
dominant_img = shrink_image_dominant(square_img_resized, TARGET_SIZE)
dominant_img_rgb = cv2.cvtColor(dominant_img, cv2.COLOR_BGR2RGB)


# --- Display before and after ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(dominant_img_rgb)
plt.title(f"{TARGET_SIZE}x{TARGET_SIZE} (Dominant Pixels)")
plt.axis("off")

if save_image == True:
    plt.imsave(save_path, dominant_img_rgb)


plt.tight_layout()
plt.show()
