from PIL import Image as im
import numpy as np 
from matplotlib import pyplot as plt

# Load and convert image to grayscale
file_path : str = "results\last_frame.png"#"inputdata\Sample2MineralMaps32x32\\32x32Viridis\\Sample2-32x32-3x4-48x64.png"
save_path : str = "results\last_frame.npy"

assert file_path.endswith('.png')
assert save_path.endswith('.npy')

img = im.open(file_path).convert("L")
data_npy = np.array(img)

# Get sorted unique values
unique_vals = np.unique(data_npy)

# Group close values (within 1 difference) and map them to the max of the group
grouped_map = {}
group = [unique_vals[0]]

for val in unique_vals[1:]:
    if val - group[-1] <= 1:
        group.append(val)
    else:
        max_val = max(group)
        for gval in group:
            grouped_map[gval] = max_val
        group = [val]
# Don't forget the last group
max_val = max(group)
for gval in group:
    grouped_map[gval] = max_val

# Apply the grouped mapping to the data
vectorized_map = np.vectorize(lambda x: grouped_map[x])
grouped_data = vectorized_map(data_npy)

# Relabel the grouped data to 0, 1, 2, ...
final_unique, relabeled_data = np.unique(grouped_data, return_inverse=True)
relabeled_data = relabeled_data.reshape(data_npy.shape) 

# Print mappings
print("Original unique values:", unique_vals)
print("Grouped value map:", grouped_map)
print("Final label mapping:", final_unique)

# Display the relabeled image
plt.imshow(relabeled_data)
plt.title("Relabeled Image with Grouped Values")
plt.axis('off')
plt.colorbar(label='Label Index')
plt.tight_layout()

# Save the result
#np.save('croppedforMCRpy_relabel_grouped.npy', relabeled_data)
np.save(save_path, relabeled_data)

print(f"Number of unique: {np.unique(relabeled_data)}")
plt.show()
