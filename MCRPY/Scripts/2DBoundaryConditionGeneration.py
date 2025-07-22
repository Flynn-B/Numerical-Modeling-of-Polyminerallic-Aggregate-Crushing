import subprocess
import mcrpy
import numpy as np
from mcrpy.src import fileutils
from mcrpy.src.Settings import ReconstructionSettings
from mcrpy.reconstruct import save_results, reconstruct
from mcrpy.src.Microstructure import Microstructure

import matplotlib.pyplot as plt

number_of_minerals = 4

#Only used for comparison, boundaries must be applied in SPOptimizerEdge
front_img = np.load('inputdata\Sample2MineralMaps32x32\\32x32Viridis\\npy32x32Granite3XRF180degrees.npy') #np.load("results\\npy32x32Granite1XRF.npy")


descriptor_filename = "results\\64x64-1x6_characterization.pickle"
descriptor_dict = fileutils.load(descriptor_filename)



square_side_length = 32
desired_shape = (square_side_length, square_side_length) #yx


settings = ReconstructionSettings(
    limit_to=16,
    descriptor_types=["Correlations", "MultiPhaseGramMatrices", "Variation", "VolumeFractions"],
    descriptor_weights=[1.0, 2.0, 250.0, 100.0],
    max_iter=10,
    use_multiphase=True,
    use_multigrid_reconstruction=True,
    convergence_data_steps=5,
    target_folder="results",
    optimizer_type="LBFGSB_2D_Bounds",
    loss_type='MSE',
    periodic=True
    )



initial_microstructure = None #np.load("results\last_frame.npy") # None if starting blank

def main() -> None:
    if initial_microstructure is None:
        print("input_microstructure : None")
        random_array = np.random.randint(0, number_of_minerals, size=desired_shape).astype(np.float64)
        #apply_boundary_faces(random_array)
        microstructure = Microstructure(random_array, use_multiphase=True)
    else: #Reapply boundry conditions
        #apply_boundary_faces(initial_microstructure)
        microstructure = Microstructure(initial_microstructure, use_multiphase=True)

    print(microstructure)

    # reconstruct
    convergence_data, last_frame = reconstruct(
        descriptor_dict, 
        desired_shape, 
        settings=settings,
        initial_microstructure=microstructure)

    print("DONE with reconstruction, wait for save.")

    # save results
    save_results(settings, convergence_data, last_frame)

    print("DONEDONE")

def get_final()->np.array:
    return np.load("results\last_frame.npy")

def visualize() -> None:
    target_data = front_img
    final_data = np.load("results\last_frame.npy")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(target_data, cmap='viridis') # Use a colormap
    axes[0].set_title('Target')
    im2 = axes[1].imshow(final_data, cmap='viridis')
    axes[1].set_title('Generated')
    plt.tight_layout()
    plt.title("Mictrostructrue")
    plt.show()
    
    #plt.imsave('results\\BOTTOMRIGHT.png', final_data, cmap='viridis')
    #plt.imsave('results\\CENTER.png', target_data, cmap='viridis')

if __name__ == "__main__":
    main()
    visualize()