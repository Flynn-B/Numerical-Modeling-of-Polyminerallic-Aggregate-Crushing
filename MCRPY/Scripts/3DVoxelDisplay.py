import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Slider

# Load the 3D array
array : list = np.load('results\last_frame.npy')  # Adjust path as needed


# Create a mask for filled voxels
filled = array >= 0

# Calculate and print the volume (voxel count) and percentage of each phase
unique, counts = np.unique(array[filled], return_counts=True)
phase_volumes = dict(zip(unique, counts))
total_filled_voxels = np.sum(filled)

print("Phase Volumes (in voxel units and percentage of filled space):")
for phase, volume in phase_volumes.items():
    percent = (volume / total_filled_voxels) * 100
    print(f"Phase {int(phase)}: {volume} voxels, {percent:.2f}% of filled volume")

# Confirm it's 3D
if array.ndim != 3:
    raise ValueError("Expected a 3D numpy array.")

# Define colormap: 0=black, 1=darkgray, 2=gray, 3=white
cmap = colors.ListedColormap(['black', 'dimgray', 'gray', 'white'])
norm = colors.Normalize(vmin=0, vmax=3)

# ----- Setup Plot -----
fig = plt.figure(figsize=(12, 6))
ax_full = fig.add_subplot(121, projection='3d')
ax_slider = fig.add_subplot(122, projection='3d')

# Plot full cube
facecolors_full = cmap(norm(array))
ax_full.voxels(filled, facecolors=facecolors_full)
ax_full.set_title("Full 3D Voxel Plot")

# Initial half-slice
initial_z = array.shape[2] // 2
slice_data = array[:, :, :initial_z]
filled_slice = slice_data >= 0
facecolors_slice = cmap(norm(slice_data))
voxels = ax_slider.voxels(filled_slice, facecolors=facecolors_slice)
ax_slider.set_title(f"Z-Slice: 0 to {initial_z}")

# ----- Slider Setup -----
ax_slider_area = plt.axes([0.25, 0.05, 0.5, 0.03])
z_slider = Slider(ax_slider_area, 'Z-Depth', 1, array.shape[2], valinit=initial_z, valstep=1)

# ----- Slider Update Function -----
def update(val):
    z = int(z_slider.val)
    ax_slider.cla()
    slice_data = array[:, :, :z]
    filled_slice = slice_data >= 0
    facecolors_slice = cmap(norm(slice_data))
    ax_slider.voxels(filled_slice, facecolors=facecolors_slice)
    ax_slider.set_title(f"Z-Slice: 0 to {z}")
    ax_slider.set_zlim(ax_full.get_zlim())  # Maintain same height range
    plt.draw()

z_slider.on_changed(update)

plt.show()
