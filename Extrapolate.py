"""
    Code extrapolates findings of Part I to a new surface image.
    Note, pickle file containing the correlations (rgb_values_to_target), must be already created by Part I.

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import label, center_of_mass, binary_erosion
from scipy.ndimage import median_filter, grey_erosion, grey_dilation
from scipy.spatial import KDTree
import pickle

import src.ColorSpace as ColorSpace
import src.Erosion as Erosion
import src.ColorMapping as ColorMapping

#Part II of code, apply correlations found in Part I, can use just saved pickle data
# Load the dictionary from the pickle file, change path if needed
with open('data\data_rgb_values_to_target.pkl', 'rb') as file:
    rgb_values_to_target = pickle.load(file)

erosion_dilation_combo = (3,4)
denoise_combo = (5,6)

applied_img_path = "Important\Sample2Surfaces\Granite1.jpg"
applied_img = io.imread(applied_img_path)

img_save : bool = False
img_save_path = "color_transformed.png" #Include .png, does not readily work for other file types

#OG Color Palette
#target_intensities_to_color = {
#    0:(6,0,157),
#    88:(255,192,203), #90 in other
#    128:(255,255,1), #130 in other
#    248:(0,255,1) } #250 in other

#Second XRF Color Palette
target_intensities_to_color = {
    0:(109,89,160),
    88:(102,139,89), #90 in other
    128:(233,148,68), #130 in other
    248:(64,179,221) } #250 in other

#Import repeat functions from file TODO: simplify and unify code
def rgb_to_grayscale(image):
    #return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image[..., 0] * 0.2989 + image[..., 1] * 0.5870 + image[..., 2] * 0.1140

def euclidean_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2))**2))


#Create 3D Color Space Visualization
ColorSpace.create_space(rgb_values_to_target, target_intensities_to_color)

# Ensure the image is in the correct format (0 to 255 range)
if applied_img.max() <= 1:
    applied_img = (applied_img * 255).astype(np.uint8)

rgb_values_to_target = rgb_values_to_target

# Get all unique RGB values in the original image
unique_colors = np.unique(applied_img.reshape(-1, 3), axis=0)


# Build the KDTree with scaled RGB values
weights = np.array([1.0, 1.0, 1.0])  # Scale blue more heavily
color_keys = list(rgb_values_to_target.keys())
color_array = np.array(color_keys) * weights  # Element-wise scaling

# Build KDTree using the scaled values
color_tree = KDTree(color_array)

# Loop through missing colors and add them if they are not there, When querying, scale the query color the same way
for color in unique_colors:
    color_tuple = tuple(color)
    if color_tuple not in rgb_values_to_target:
        print(f"Missing color tuple: {color_tuple}")

        scaled_color = np.array(color_tuple) * weights
        dist, idx = color_tree.query(scaled_color)
        closest_color = color_keys[idx]

        target_intensity = rgb_values_to_target[closest_color]
        rgb_values_to_target[color_tuple] = target_intensity

#ColorSpace.create_space(rgb_values_to_target, rgb_values_to_target)

# Convert the image to a standard Python list of lists (normal array)
img = applied_img.tolist()

# Loop through the image and modify the pixels based on the RGB values
for i in range(len(img)):  # Iterate over rows
    for j in range(len(img[i])):  # Iterate over columns
        pixel_value = tuple(img[i][j])  # Get the pixel value (RGB as a tuple)
        
        # Check if the pixel RGB value exists in the defined dictionary
        if pixel_value in rgb_values_to_target:
            target_intensity = rgb_values_to_target[pixel_value]  # Get the target intensity
            # Replace the pixel with the target intensity (make it a gray value)
            img[i][j] = [target_intensity] * 3  # Apply the same target intensity to R, G, B

#TODO MOVE FUNCTION
def link_dark_region_edges_with_path_check(
    image, area_tolerance=50, distance_tolerance=100, path_dark_fraction=0.8):
    """
    Connects dark regions in a grayscale image using lines drawn between the closest
    edge pixels, only if the path is sufficiently dark (e.g., 80% of pixels < 100).

    Parameters:
        image (np.ndarray): 2D grayscale image.
        area_tolerance (int): Minimum area to consider a region.
        distance_tolerance (float): Max distance between edge pixels to connect.
        path_dark_fraction (float): Min fraction of pixels along the path that must be <100.

    Returns:
        np.ndarray: Modified image with lines connecting dark regions.
    """
    if image.ndim != 2:
        raise ValueError("Image must be a 2D grayscale image.")

    # Step 1: Identify dark regions
    black_mask = (image == 0).astype(np.uint8)
    labeled, num_features = label(black_mask)

    # Step 2: Get edge pixels for large-enough regions
    regions = []
    for i in range(1, num_features + 1):
        region_mask = (labeled == i)
        area = np.sum(region_mask)
        if area < area_tolerance:
            continue
        eroded = binary_erosion(region_mask)
        edge_mask = region_mask & ~eroded
        edge_pixels = np.column_stack(np.nonzero(edge_mask))
        regions.append(edge_pixels)

    # Step 3: Prepare output image
    output = image.copy()

    def draw_line_if_dark_enough(img, p1, p2, value=0):
        x1, y1 = p1[1], p1[0]
        x2, y2 = p2[1], p2[0]
        length = max(abs(x2 - x1), abs(y2 - y1)) + 1
        xs = np.linspace(x1, x2, length).astype(int)
        ys = np.linspace(y1, y2, length).astype(int)
        
        # Clip to stay inside bounds
        xs = np.clip(xs, 0, img.shape[1] - 1)
        ys = np.clip(ys, 0, img.shape[0] - 1)

        # Check how many pixels are already dark (< 100)
        values = img[ys, xs]
        dark_ratio = np.sum(values < 100) / len(values)

        if dark_ratio >= path_dark_fraction:
            img[ys, xs] = value  # Draw line

    # Step 4: For each pair of regions, check edge-to-edge distance and path darkness
    for i in range(len(regions)):
        tree = KDTree(regions[i])
        for j in range(i + 1, len(regions)):
            distances, indexes = tree.query(regions[j])
            min_idx = np.argmin(distances)
            if distances[min_idx] < distance_tolerance:
                p1 = regions[i][indexes[min_idx]]
                p2 = regions[j][min_idx]
                draw_line_if_dark_enough(output, p1, p2)

    return output

# Convert the modified image back to a numpy array
img = np.array(img, dtype=np.uint8)
#TODO revisit relevance here
# Convert to grayscale
gray = rgb_to_grayscale(img).astype(np.uint8)

#Link together areas > area_tolerance within distance < distance tolerance (25 or 55?), slightly arbitrary number
gray = link_dark_region_edges_with_path_check(gray, area_tolerance=5, distance_tolerance=55,path_dark_fraction=0.75)

# Apply erosion and dilation (opening)
eroded = grey_erosion(gray, size=erosion_dilation_combo[0])
opened = grey_dilation(eroded, size=erosion_dilation_combo[1])
# Convert back to 3-channel image if needed
eroded_opened_img = np.stack([opened] * 3, axis=-1)

# Apply median filtering ONLY to the modified image (to remove isolated pixels)
denoised_img = median_filter(eroded_opened_img, size=denoise_combo[0]) 
denoised_img = median_filter(denoised_img, size=denoise_combo[1]) 

denoised_img=Erosion.replace_white_spots(denoised_img,100,100)

# === Window 1: Original, Modified, Denoised ===
plt.figure(figsize=(18, 6))

# 1. Original Image
plt.subplot(1, 3, 1)
plt.imshow(applied_img)
plt.title('Original Image')
plt.axis('off')

# 2. Modified Image
plt.subplot(1, 3, 2)
plt.imshow(img)
plt.title('Modified Image (Before Denoising)')
plt.axis('off')

# 3. Denoised Image
plt.subplot(1, 3, 3)
plt.imshow(denoised_img, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')

plt.tight_layout()
plt.show()  # This opens the first window



# === Window 2: Color-Transformed Denoised Image ===
plt.figure(figsize=(6, 6))

color_transformed = ColorMapping.apply_color_mapping(denoised_img, target_intensities_to_color)

plt.imshow(color_transformed)
plt.title('Color-Transformed Denoised Image')
plt.axis('off')

if img_save == True:
    plt.imsave(img_save_path, color_transformed)

plt.tight_layout()
plt.show()  # This opens the second window