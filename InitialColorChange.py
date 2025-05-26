"""
    Code is a basic way to change RGB colors of XRF map into a RGB greyscale format that is more readable
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Load the image
image = io.imread('Important\XRFSample2BulkyBiotite.jpg')  # RGB format
new_image = image.copy()

# Color replacements: (old RGB) -> (new RGB)
color_map = {
    (109, 89, 160): (0, 0, 0), #Biotite
    (102, 139, 89): (90, 84, 84), #Feldspar
    (64, 179, 221): (250, 250, 250), #Plagioclase
    (233, 148, 68): (130, 130, 130) #Quartz
}

# Tolerance per channel
tolerance = 10

def fuzzy_match(mask_img, target_rgb, tolerance):
    lower = np.maximum(0, np.array(target_rgb) - tolerance)
    upper = np.minimum(255, np.array(target_rgb) + tolerance)
    return np.all((mask_img >= lower) & (mask_img <= upper), axis=-1)

# Apply fuzzy color replacement
for old_rgb, new_rgb in color_map.items():
    mask = fuzzy_match(new_image, old_rgb, tolerance)
    new_image[mask] = new_rgb

# Save the result
#Note save as PNG first and convert online to JPG with 0% loss (!!!)
#io.imsave('XRFSample2SectionBulkyBiotite.png', new_image)

# Optional: show before and after
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(new_image)
axes[1].set_title("Modified (Fuzzy Match)")
axes[1].axis('off')

plt.tight_layout()
plt.show()
