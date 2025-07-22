import subprocess
import mcrpy
import numpy as np
from mcrpy.src import fileutils
from mcrpy.src.Settings import ReconstructionSettings
from mcrpy.reconstruct import save_results, reconstruct
from mcrpy.src.Microstructure import Microstructure

#Boundary Conditions:

number_of_minerals = 4

#Optional unless using existing (full) microstructure and initial_microstructure = True, not needed if doing bounded generation from scratch
front_img = np.load("inputdata\Sample2MineralMaps32x32/npy32x32Granite1XRF.npy")
back_img = np.load("inputdata\Sample2MineralMaps32x32/npy32x32Granite3XRF180degrees.npy")
left_img = np.load("inputdata\Sample2MineralMaps32x32/npy32x32Granite5XRF.npy")
right_img = np.load("inputdata\Sample2MineralMaps32x32/npy32x32Granite6XRF.npy")
top_img = np.load("inputdata\Sample2MineralMaps32x32/npy32x32Granite2XRF.npy")
bottom_img = np.load("inputdata\Sample2MineralMaps32x32/npy32x32Granite4XRF.npy")

def apply_boundary_faces(cube: np.ndarray):
    # Apply faces with necessary inline transformations
    cube[0, :, :]   = np.flip(np.rot90(front_img, k=1), axis=1)     # Front (x = 0): CCW + horizontal flip
    cube[-1, :, :]  = np.rot90(back_img, k=-1)                        # Back (x = -1): Clockwise
    cube[:, 0, :]   = np.rot90(right_img, k=-1)                       # Right (y = 0): Clockwise
    cube[:, -1, :]  = np.flip(np.rot90(left_img, k=1), axis=1)      # Left (y = -1): CCW + horizontal flip
    cube[:, :, 0]   = np.flip(bottom_img, axis=1)                   # Bottom (z = 0): Horizontal flip
    cube[:, :, -1]  = np.rot90(top_img, k=2)                         # Top (z = -1): 180° rotation


descriptor_filename = "results\Sample2-32x32-3x4-48x64_characterization.pickle"
descriptor_dict = fileutils.load(descriptor_filename)


cube_side_length = 64
desired_shape = (cube_side_length, cube_side_length , cube_side_length)

settings = ReconstructionSettings(
    limit_to=16,
    descriptor_types=["Correlations", "MultiPhaseGramMatrices", "Variation"],
    descriptor_weights=[1.0, 1.0, 100.0],
    max_iter=1000,
    use_multiphase=True,
    use_multigrid_reconstruction=True,
    isotropic=False,
    convergence_data_steps=1,
    target_folder="results",
    optimizer_type="LBFGSB",
    nl_method="relu"
    )

initial_microstructure = None #np.load("results\last_frame.npy") # None if starting blank

if initial_microstructure is None:
    print("No Microstrure")
    random_array = np.random.randint(0, number_of_minerals, size=(cube_side_length, cube_side_length, cube_side_length)).astype(np.float64)
    #apply_boundary_faces(random_array)
    initial_microstructure = Microstructure(random_array, use_multiphase=True)
else: #Reapply boundry conditions
    apply_boundary_faces(initial_microstructure)
    initial_microstructure = Microstructure(initial_microstructure, use_multiphase=True)

print(initial_microstructure)

# reconstruct
convergence_data, last_frame = reconstruct(
    descriptor_dict, 
    desired_shape, 
    settings=settings,
    initial_microstructure=initial_microstructure)

print("DONE with reconstruction, wait for save.")

# save results
save_results(settings, convergence_data, last_frame)

print("DONEDONE")