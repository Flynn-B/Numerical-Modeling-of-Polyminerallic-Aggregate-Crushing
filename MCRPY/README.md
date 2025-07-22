# Utilizing MCRpy for Bounded Granite Reconstruction


Microstructure Characterization and Reconstruction in Python (MCRpy) is a modular and extensible model based on descriptors and gradient optimization techniques.

As part of the project "Numerical Modeling of Polyminerallic Aggregate Crushing", MCRpy's capabilities are expanded with custom descriptors to support bounded generation. An internal microstructure is generated based on a real sample shell as derived using surface extrapolations of micro-XRF scans. The output conforms to the shell in a logical and continuous manner.

MCRpy was written by Paul Seibert and Alexander Rassloff in the lab of Markus Kaestner at the institute of solid mechanics, TU Dresden, Germany. More information [here](https://github.com/NEFM-TUDresden/MCRpy).

## Installation

It is **strongly** recommended to have MCRpy built in a seperate project. Furthermore, newer Python versions fail to run as expected! Python 3.8.0 has been tested and is **strongly** recommended.

An editable install is needed, so clone repo.

```shell
git clone https://github.com/NEFM-TUDresden/MCRpy.git
```

Dependencies and package versions **do** matter. To simplify process, include and utilize `requirements.txt` found in this folder.

```shell
pip install -r requirements.txt
```

Alternatively, original installation instructions can be found on [MCRpy Github](https://github.com/NEFM-TUDresden/MCRpy), but note versions of Python and dependencies can be misleading. 

## Getting Started
### Setting up Custom Pluings

This step is critical for bounded generation! Copy the files from inside the `CustomPlugins` folder into the folder `mcrpy\optimizers`. 


### First Generation

A complete generation requires the following steps:

1. Preparing surface mineralogy maps for MCRpy
2. Characterizing input surface
3. Reconstruction using bounds
4. Visualizing and Exporting Results


#### Preparing surface mineralogy maps for MCRp

To begin MCR of a given sample, the surface scans must be laid out in a manner conducive for generation. Furthermore, the images must be the right scale for generation. Typically, this means that each image is of dimensions you want final microstructure to be (e.g. 64x64 for a cube of 64x64x64).

One such configuration is a stacked allignment of the six sides of a cube:

! TODO ! Allignment Image. 

An image must imported as a format readable by MCRpy (`.npy`) using `ImportImageasNPY.py`. This code works by grouping colors by proximity to each other. If results are not desirable, some pre-processing may be needed to differentiate colors or manually group them. 

#### Characterization of microstructure

The easiest method to characterize a microstructure is to use the built-in GUI. To open, run `gui_mcrpy.py`. Then go to Characterize under the Actions toolbar. There, choose filename, descriptors, and set multiphase to true.

Recommended descriptors:

|Correlations | MultiPhaseGramMatrices | Variation | VolumeFractions |
| --- | --- | --- | --- |

Running the characterization results in a file of `.pickle` format.

#### Reconstruction of 3D microstructure with bounds

Reconstruction utilizes the `.pickle` characterization file generated in the previous step. The easiest method to generate a bounded microstructure is to use the prebuilt scripts `3DBoundaryConditionGeneration.py` or `2DBoundaryConditionGeneration.py`. Explicitly set the values for `number_of_minerals`, `cube_side_length`, and file path `descriptor_filename`. 

Additionally, within the parameters of ReconstructionSettings, the following descriptors (must match characterization) and weights are recommended:

| Correlations | MultiPhaseGramMatrices | Variation | VolumeFractions |
|--------------|------------------------|-----------|-----------------|
| 1.0          | 2.0                    | 250.0     | 0.4             |

The resulting output is a 3D/2D `.npy` array. Note, this step can take many hours or even days. 

#### Visualizing and Exporting Results

Visualization of the results can be acheived multiple ways:

1. Export to Paraview
2. Axial scans 
3. Basic voxel visualization

Exporting to a Paraview format allows for the most comprehensive analysis and visualization of data. The easiest way to do so is opening the MCRpy GUI (run `gui_mcrpy.py`), go to view under Actions side bar, input file, and set savefig to True. The saved file will be either an image (if 2D) or a file readible by Paraview (if 3D).

An easy way to view axial slices in all 3 dimensions is to set file path `data_npy` in `3Dto2Dimageslider.py` and run. The code provides a colored view and a gray-scale view with green location bars. These bars can be used to view a single feature from 3 different angles, as is done with brain imaging. 

A final visualization can be achieved by setting file path in `3DVoxelDisplay.py` and running. However, note this is unoptimized and slow. It is mainly for a quick external check without needing to open Paraview. It does **not** provide an easy way to gauge quality as do the other two methods. 


