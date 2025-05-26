"""
    Module erodes noise of a certain area in the the image
    - Flynn Basehart
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure
from collections import Counter

def replace_white_spots(image, area_threshold=100, max_iterations=100):
    # Convert image to grayscale if it's not already
    if image.ndim == 3:
        image = np.dot(image[..., :3], [0.2989, 0.587, 0.114])  # Grayscale conversion

    # Step 1: Threshold to identify unique grayscale values in the image
    unique_values = np.unique(image)  # Get all unique grayscale values in the image
    regions_mask = np.zeros_like(image, dtype=bool)

    # Loop over each unique grayscale value to find connected regions
    for value in unique_values:
        # Skip black regions (value = 0)
        if value == 0:
            continue

        # Create a binary mask for the current grayscale value
        binary_mask = image == value

        # Label connected components
        labeled_image, num_labels = measure.label(binary_mask, connectivity=2, return_num=True)

        # Filter out regions with area >= area_threshold pixels (we keep only small regions)
        properties = measure.regionprops(labeled_image)
        for prop in properties:
            if prop.area >= area_threshold:
                labeled_image[labeled_image == prop.label] = 0  # Remove large regions (>= 100 px)

        # Combine the small valid regions (area < area_threshold px) into the final mask
        regions_mask = np.logical_or(regions_mask, labeled_image > 0)

    # Step 2: Replace identified small regions with white (255) in the image
    image_with_white_spots = image.copy()
    image_with_white_spots[regions_mask] = 255  # Set the identified small regions to white

    # Step 3: Function to replace white pixels with the most dominant neighboring gray value
    def replace_white_pixels_with_neighbors(image_with_white_spots):
        rows, cols = image_with_white_spots.shape
        # Create a copy to modify the image
        new_image = image_with_white_spots.copy()

        # Loop through the image and replace each white pixel
        for i in range(rows):  # Include the edge pixels
            for j in range(cols):  # Include the edge pixels
                if image_with_white_spots[i, j] == 255:  # If the pixel is white
                    # Get the 4-connectivity neighbors
                    neighbors = [
                        image_with_white_spots[i-1, j] if i > 0 else None,  # Above
                        image_with_white_spots[i+1, j] if i < rows - 1 else None,  # Below
                        image_with_white_spots[i, j-1] if j > 0 else None,  # Left
                        image_with_white_spots[i, j+1] if j < cols - 1 else None  # Right
                    ]

                    # Exclude the white pixels (255) and None (out of bounds) from the neighbors
                    neighbors = [n for n in neighbors if n is not None and n != 255]

                    # If there are neighbors to replace the white pixel
                    if neighbors:
                        most_common = Counter(neighbors).most_common(1)[0][0]  # Get the most common gray value
                        new_image[i, j] = most_common  # Replace the white pixel with the dominant gray value

        return new_image

    # Step 4: Iteratively replace white pixels until no white pixels are left (or max iterations)
    iterations = 0
    while np.any(image_with_white_spots == 255) and iterations < max_iterations:  # As long as there are white pixels
        image_with_white_spots = replace_white_pixels_with_neighbors(image_with_white_spots)
        iterations += 1
        if iterations % 10 == 0:  # Print progress every 10 iterations, should not take more than typically 20. More indicates an error
            print(f"Iteration {iterations}: White pixels remaining: {np.sum(image_with_white_spots == 255)}")

    return image_with_white_spots