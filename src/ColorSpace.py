"""
    Module displays the color space of the input color_dictionary gathered from the surface image or created within the correlation code. 
    - FB
"""

import matplotlib.pyplot as plt
import numpy as np

# Example usage: create_space(rgb_values_to_target, target_intensities_to_color)
def create_space(color_dictionary, target_intensities_to_color):

    color_dictionary = color_dictionary # Surface image colors
    target_intensities_to_color=target_intensities_to_color # Mineralogy colors

    # Extract RGB and greyscale values into lists
    rgb_values = np.array(list(color_dictionary.keys()), dtype=np.uint8)
    grey_values = np.array(list(color_dictionary.values()), dtype=np.uint8)

    print(np.unique(grey_values))

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(121, projection='3d')

    # Scatter plot using RGB as color and position
    colors1 = rgb_values / 255.0   # Normalize RGB for matplotlib color

    ax1.scatter(
        rgb_values[:, 0], rgb_values[:, 1], rgb_values[:, 2],
        c=colors1,  # color of the point
        s=2, #size
        edgecolors='none'
    )

    # Labeling axes
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    ax1.set_title('RGB Color Space with True Colors')

    # Fuzzy color matching function
    def get_fuzzy_color(value, mapping, fuzziness):
        for k in mapping:
            if abs(value - k) <= fuzziness:
                return mapping[k]
        raise ValueError("Unexpected Greyscale Color. Either fuzziness too low or other error")  # fallback error

    # Create figure
    ax2 = fig.add_subplot(122, projection='3d')

    # Scatter plot using RGB as color and position
    colors2 = np.array([
        tuple(c / 255.0 for c in get_fuzzy_color(val, target_intensities_to_color, 5))
        for val in grey_values
    ])
    ax2.scatter(
        rgb_values[:, 0], rgb_values[:, 1], rgb_values[:, 2],
        c=colors2,  # color of the point
        s=2, #size
        edgecolors='none'
    )

    # Labeling axes
    ax2.set_xlabel('Red')
    ax2.set_ylabel('Green')
    ax2.set_zlabel('Blue')
    ax2.set_title('RGB Color Space with Minerals Identified')

    plt.show()