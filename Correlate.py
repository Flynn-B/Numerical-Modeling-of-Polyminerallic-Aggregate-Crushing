"""
    Code characterizes the correlations between the RGB colors of the surface image and of the mineralogy map (ideal image).
    Saves results in a pickle file containing the dictionary rgb_values_to_target.
    This code (Part I), then passes the saved dictionary to Part II to extrapolate and apply onto unseen images.    
    IMPORTANT: The original and ideal images must overlap as perfectly as possible.
    - FB
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import median_filter, grey_erosion, grey_dilation
import pickle

import src.ColorSpace as ColorSpace
import src.Erosion as Erosion
import src.ColorMapping as ColorMapping

# Load original and ideal images
original_img = io.imread('Important\\SURFACESample2Section.jpg') # RGB Surface image
ideal_img = io.imread('Important\\XRFSample2Section.jpg') # Mineralogy Map

# Convert to images to uint8 if needed
if original_img.dtype != np.uint8:
    original_img = (original_img * 255).astype(np.uint8)
if ideal_img.dtype != np.uint8:
    ideal_img = (ideal_img * 255).astype(np.uint8)

# Define intensity targets
target_intensities = [0, 90, 130, 250]

#Colors that corespond to intensities (minerals)
target_intensities_to_color = {
    0:(6,0,157),
    88:(255,192,203), #90 in other
    128:(255,255,1), #130 in other
    248:(0,255,1) } #250 in other

def rgb_to_grayscale(image):
    #return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image[..., 0] * 0.2989 + image[..., 1] * 0.5870 + image[..., 2] * 0.1140

# Get all unique RGB values in the original image
unique_colors = np.unique(original_img.reshape(-1, 3), axis=0)

# Dictionary to store RGB to target intensity mappings
rgb_values_to_target = {}

def euclidean_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2))**2))

# Compute the average difference between two images
def average_difference(array1, array2):
    if array1.shape != array2.shape:
        raise TypeError("Arrays are not the same size")
    gray1 = rgb_to_grayscale(array1)
    gray2 = rgb_to_grayscale(array2)
    difference = np.abs(gray1 - gray2)
    normalized = np.clip(difference, 0, 255).astype(np.uint8)
    return np.mean(normalized) / 255

# Find the best target intensity match for a given RGB color
def find_best_rgb_match(color, img_rgb, ideal_rgb):
    print(f"Processing color: {color}")
    mask = np.all(img_rgb == color, axis=-1)
    if not np.any(mask):
        print("  Color not found in image.")
        return img_rgb

    best_choice = 0 #Grey value
    best_diff = 1.0 #Lowest statistical difference

    masked_indices = np.where(mask)
    original_pixels = img_rgb[masked_indices]

    for target in target_intensities:
        test_color = np.array([target, target, target], dtype=np.uint8)
        img_rgb[masked_indices] = test_color
        diff = average_difference(img_rgb, ideal_rgb)
        if diff < best_diff:
            best_diff = diff
            best_choice = target
            print(f"  Found better target {test_color} with diff {diff:.4f}")

        img_rgb[masked_indices] = original_pixels

    # Store the best target intensity for the color in the mapping
    rgb_values_to_target[tuple(color)] = best_choice

    # Apply best target color
    final_color = np.array([best_choice] * 3, dtype=np.uint8)
    img_rgb[mask] = final_color
    return img_rgb


def find_best_morphology(img_rgb, ideal_rgb):
    print("Trying different grayscale morphology combinations...")

    # Convert to grayscale for morphology
    gray_img = rgb_to_grayscale(img_rgb).astype(np.uint8)
    best_combo = (0, 0)  # (erosion size, dilation size)
    best_diff = 1.0

    for erode_size in range(1, 5):
        for dilate_size in range(1, 5):
            print(f"    Testing combo: erosion={erode_size}, dilation={dilate_size}")

            # Apply grayscale erosion then dilation (opening)
            eroded = grey_erosion(gray_img, size=(erode_size, erode_size))
            opened = grey_dilation(eroded, size=(dilate_size, dilate_size))

            test_img = np.stack([opened] * 3, axis=-1)

            diff = average_difference(test_img, ideal_rgb)

            if diff < best_diff:
                best_diff = diff
                best_combo = (erode_size, dilate_size)
                print(f"    Found better combo with diff {diff:.4f}")

    print(f"\n Best morphology combo: {best_combo} with average difference {best_diff:.4f}")
    return best_combo

def find_best_denoising(img_rgb, ideal_rgb):
    print("Trying different median filter size combinations...")

    best_combo = (0, 0)  # (size1, size2)
    best_diff = 1.0      # Lowest statistical difference

    for size1 in range(1, 7):  # Size range can be adjusted
        for size2 in range(1, 7):
            print(f"    Testing combo: size=({size1}, {size2})")
            
            img_filtered = median_filter(img_rgb, size=size1)
            img_filtered = median_filter(img_filtered, size=size2)

            diff = average_difference(img_filtered, ideal_rgb)

            if diff < best_diff:
                best_diff = diff
                best_combo = (size1, size2)
                print(f"    Found better combo with diff {diff:.4f}")

    print(f"\n Best filter sizes: {best_combo} with average difference {best_diff:.4f}")
    return best_combo

def replace_with_best_rgb_match(color, img_rgb, rgb_values_to_target):
    print(f"Processing color: {color}")
    #Optimized way to find mask
    mask=(img_rgb[..., 0] == color[0]) & \
        (img_rgb[..., 1] == color[1]) & \
        (img_rgb[..., 2] == color[2])
    if not np.any(mask):
        print("  Color not found in image.") # Should not ever be printed
        return img_rgb
    
    # Store the best target intensity for the color in the mapping
    best_choice = rgb_values_to_target[tuple(color)]

    # Apply best target color
    final_color = np.array([best_choice] * 3, dtype=np.uint8)
    img_rgb[mask] = final_color
    return img_rgb

# Load the dictionary from the pickle file TODO TODO: This necessary here???
# with open('data\data_rgb_values_to_target.pkl', 'rb') as file:
#    rgb_values_to_target = pickle.load(file)

# Initialize modified image
modified_img = original_img.copy()

for color in unique_colors:
    modified_img = replace_with_best_rgb_match(color, modified_img, rgb_values_to_target)

# Map every RGB value in the image to the closest RGB value in the unique_colors list
height, width, _ = original_img.shape
for i in range(height):
    for j in range(width):
        pixel_value = tuple(modified_img[i, j])
        
        # Check if the pixel already has a target intensity
        if pixel_value not in rgb_values_to_target:
            # Find the closest RGB value from the unique_colors
            closest_color = min(unique_colors, key=lambda c: euclidean_distance(c, pixel_value))
            # Assign the target intensity of the closest color
            rgb_values_to_target[pixel_value] = rgb_values_to_target[tuple(closest_color)]
            
            # Apply the closest color's target intensity
            target_intensity = rgb_values_to_target[pixel_value]
            modified_img[i, j] = [target_intensity] * 3

# Output the RGB to target intensity mapping
print("\nRGB to Target Intensity Mapping:")
for rgb, target in rgb_values_to_target.items():
    print(f"{rgb} -> {target}")

# Output stats
print("\nFinal average difference (original vs ideal):", average_difference(original_img, ideal_img))
print("Final average difference (modified vs ideal):", average_difference(modified_img, ideal_img))


erosion_dilation_combo = find_best_morphology(modified_img, ideal_img)

# Convert to grayscale
gray = rgb_to_grayscale(modified_img).astype(np.uint8)
# Apply erosion and dilation (opening)
eroded = grey_erosion(gray, size=erosion_dilation_combo[0])
opened = grey_dilation(eroded, size=erosion_dilation_combo[1])
# Convert back to 3-channel image if needed
eroded_opened_img = np.stack([opened] * 3, axis=-1)

denoise_combo = find_best_denoising(eroded_opened_img, ideal_img)

cropped_denoised = median_filter(eroded_opened_img, size=denoise_combo[0])  # Use size=3 to focus on small isolated noise
cropped_denoised = median_filter(cropped_denoised, size=denoise_combo[1])

# Final check to make sure every unique RGB value has a corresponding target intensity
missing_values = []
for color in unique_colors:
    if tuple(color) not in rgb_values_to_target:
        missing_values.append(tuple(color))

if missing_values:
    print(f"\nWarning: The following RGB values are missing target intensities:")
    for missing_color in missing_values:
        print(f"  {missing_color}")
else:
    print("\nAll RGB values have a corresponding target intensity.")


# Save the rgb_values_to_target dictionary to a pickle file to use for Part II without rerunning code each time
with open('data\data_rgb_values_to_target.pkl', 'wb') as file:
    pickle.dump(rgb_values_to_target, file)


# Visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(original_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(ideal_img)
axes[1].set_title('Ideal Image')
axes[1].axis('off')

axes[2].imshow(modified_img)
axes[2].set_title('Modified Image')
axes[2].axis('off')

diff_display = np.abs(rgb_to_grayscale(ideal_img) - rgb_to_grayscale(modified_img))
axes[3].imshow(diff_display.astype(np.uint8), cmap='gray')
axes[3].set_title('Grayscale Difference')
axes[3].axis('off')

plt.tight_layout()
plt.show()


#Visualization 2
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(modified_img)
axes[0].set_title('Modified Image')
axes[0].axis('off')

axes[1].imshow(cropped_denoised)
axes[1].set_title('Optimized Blending Image')
axes[1].axis('off')

axes[2].imshow(ideal_img)
axes[2].set_title('Ideal Image')
axes[2].axis('off')

diff_display = np.abs(rgb_to_grayscale(ideal_img) - rgb_to_grayscale(cropped_denoised))
axes[3].imshow(diff_display.astype(np.uint8), cmap='gray')
axes[3].set_title('Grayscale Difference')
axes[3].axis('off')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 1, figsize=(20, 5))
axes.imshow(ColorMapping.apply_color_mapping(cropped_denoised, target_intensities_to_color))
axes.set_title('Ideal Image')
axes.axis('off')

plt.tight_layout()
plt.show()
