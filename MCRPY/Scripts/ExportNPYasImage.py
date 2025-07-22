# Helpful for double checking existing npy files
import numpy as np
import matplotlib.pyplot as plt


file_list = ['64x64Granite1XRF.npy','64x64Granite2XRF.npy',
             '64x64Granite3XRF180degrees.npy',
             '64x64Granite4XRF.npy','64x64Granite5XRF.npy',
             '64x64Granite6XRF.npy']

for file in file_list:
    assert file.endswith('.npy')
    data = np.load(f'inputdata\Sample2MineralMaps64x64\\{file}')
    plt.imsave(f'{file[:-4]}.png', data, cmap='viridis')