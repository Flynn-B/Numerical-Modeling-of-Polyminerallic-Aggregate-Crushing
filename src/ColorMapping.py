import numpy as np

# Function to convert RGB to grayscale
def rgb_to_grayscale(image):
    #return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image[..., 0] * 0.2989 + image[..., 1] * 0.5870 + image[..., 2] * 0.1140

def apply_color_mapping(image, intensity_to_color):
    """
    Transforms a grayscale or RGB image into a color-mapped RGB image using a given intensity-to-RGB mapping.

    Parameters:
        image (np.ndarray): Input image (grayscale or RGB).
        intensity_to_color (dict): Dictionary mapping intensity values (int) to RGB tuples.

    Returns:
        np.ndarray: Color-transformed RGB image (uint8).
    """

    # Ensure the input image is in uint8 format
    image = image.astype(np.uint8)

    # Convert RGB to grayscale if needed
    if image.ndim == 3 and image.shape[2] == 3:
        image_gray = rgb_to_grayscale(image)
        image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)
    else:
        image_gray = image

    print(np.unique(image_gray))

    # Create an empty RGB image
    height, width = image_gray.shape
    color_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Map grayscale intensities to colors
    for intensity, rgb in intensity_to_color.items():
        mask = image_gray == intensity
        color_img[mask] = np.array(rgb, dtype=np.uint8)

    return color_img