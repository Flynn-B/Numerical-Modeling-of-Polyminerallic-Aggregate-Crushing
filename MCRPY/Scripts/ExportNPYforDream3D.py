import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the 3D array
file_name = 'last_frame_64x64x64.npy'
data : list = np.load(f'results\{file_name}')

assert file_name.endswith('.npy')

dimensions = np.shape(data)
z_axis = dimensions[2]

for layer in range(z_axis):
    plt.imsave(f'resultstemp\Dream3D\{file_name[:-4]}Layer{layer}.png', data[layer], cmap='viridis')

for layer in range(z_axis):
    img_path = os.path.join('resultstemp\Dream3D', f'{file_name[:-4]}Layer{layer}.png')
    with Image.open(img_path) as img:
        img_rgb = img.convert('RGB')  # Ensure image is in RGB mode
        pixels = list(img_rgb.getdata())
        unique_colors = set(pixels)
        print(f'Layer {layer}: {len(unique_colors)} unique colors')