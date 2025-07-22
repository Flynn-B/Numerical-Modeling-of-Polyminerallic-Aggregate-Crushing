"""
   Copyright 10/2020 - 04/2021 Paul Seibert for Diploma Thesis at TU Dresden
   Copyright 05/2021 - 12/2021 TU Dresden (Paul Seibert as Scientific Assistant)
   Copyright 2022 TU Dresden (Paul Seibert as Scientific Employee)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from __future__ import annotations

from typing import List

import numpy as np
import tensorflow as tf
import scipy.optimize as sopt

from mcrpy.optimizers.Optimizer import Optimizer
from mcrpy.src.Microstructure import Microstructure

number_of_minerals = 4
# initial 32x32 image (only needed for 32x32x32 or if doing multigrid_reconstruction=True)
front_img  = np.load("Sample2MineralMaps32x32/npy32x32Granite1XRF.npy")
back_img   = np.load("Sample2MineralMaps32x32/npy32x32Granite3XRF180degrees.npy")
left_img   = np.load("Sample2MineralMaps32x32/npy32x32Granite5XRF.npy")
right_img  = np.load("Sample2MineralMaps32x32/npy32x32Granite6XRF.npy")
top_img    = np.load("Sample2MineralMaps32x32/npy32x32Granite2XRF.npy")
bottom_img = np.load("Sample2MineralMaps32x32/npy32x32Granite4XRF.npy")

#64x64 (needed if doing 64x64x64)
front_img_64  = np.load("Sample2MineralMaps64x64/64x64Granite1XRF.npy")
back_img_64   = np.load("Sample2MineralMaps64x64/64x64Granite3XRF180degrees.npy")
left_img_64   = np.load("Sample2MineralMaps64x64/64x64Granite5XRF.npy")
right_img_64  = np.load("Sample2MineralMaps64x64/64x64Granite6XRF.npy")
top_img_64    = np.load("Sample2MineralMaps64x64/64x64Granite2XRF.npy")
bottom_img_64 = np.load("Sample2MineralMaps64x64/64x64Granite4XRF.npy")

def one_hot_encode_face(face_img, num_classes=4):
    return np.eye(num_classes, dtype=np.uint8)[face_img]

def apply_boundary_faces(cube: np.ndarray) -> np.ndarray:
    global front_img, back_img, left_img, right_img, top_img, bottom_img
    new_cube = cube.copy()

    shape_of_cube = np.shape(new_cube)
    if shape_of_cube[1] ==64:
        front_img=front_img_64
        back_img =back_img_64
        left_img =left_img_64 
        right_img =right_img_64 
        top_img= top_img_64   
        bottom_img= bottom_img_64 
    # print(f"Shape of cube {np.shape(new_cube)}")

    transformed = np.flip(np.rot90(front_img, k=1), axis=1)
    new_cube[0, :, :, :] = one_hot_encode_face(transformed)

    transformed = np.rot90(back_img, k=-1)
    new_cube[-1, :, :, :] = one_hot_encode_face(transformed)

    transformed = np.rot90(right_img, k=-1)
    new_cube[:, 0, :, :] = one_hot_encode_face(transformed)

    transformed = np.flip(np.rot90(left_img, k=1), axis=1)
    new_cube[:, -1, :, :] = one_hot_encode_face(transformed)

    transformed = np.flip(bottom_img, axis=1)
    new_cube[:, :, 0, :] = one_hot_encode_face(transformed)

    transformed = np.rot90(top_img, k=2)
    new_cube[:, :, -1, :] = one_hot_encode_face(transformed)

    return new_cube

class SPOptimizer3D(Optimizer):
    is_gradient_based = True
    is_vf_based = False
    is_sparse = False

    def __init__(self,
                 max_iter: int = 100,
                 desired_shape_extended: tuple = None,
                 callback: callable = None):
        """ABC init for scipy optimizer. Subclasses simply specify self.optimizer_method and self.bounds."""
        self.max_iter = max_iter
        self.desired_shape_extended = desired_shape_extended
        self.reconstruction_callback = callback
        self.current_loss = None

        assert self.reconstruction_callback is not None
        self.optimizer_method = None
        self.bounds = None

    def step(self, x: np.ndarray) -> List[np.ndarray]:
        """Perform a single step. Typecasting from np to tf and back needed to couple scipy optimizers with tf backprop."""
        # Assign optimizer variable to the microstructure
        self.ms.x.assign(x.reshape(self.desired_shape_extended).astype(np.float64))

        # Apply fixed boundary faces again (this "sticks" the faces back on after update)
        new_val = self.ms.numpy()
        new_val = apply_boundary_faces(new_val)
        self.ms.x.assign(new_val.reshape(self.ms.x.shape).astype(np.float64))

        # Compute loss and gradients
        loss, grads = self.call_loss(self.ms)
        self.current_loss = loss
        self.reconstruction_callback(self.n_iter, loss, self.ms)
        self.n_iter += 1
        return [field.numpy().astype(np.float64).flatten() for field in [loss, grads[0]]]

    def optimize(self, ms: Microstructure, restart_from_niter: int = None) -> int:
        """Optimize."""
        print("Running SPOptimzer 1")
        self.n_iter = 0 if restart_from_niter is None else restart_from_niter
        sp_options = {
            'maxiter': self.max_iter - self.n_iter,
            'maxfun': self.max_iter - self.n_iter,
        }
        self.ms = ms
        self.opt_var = [self.ms.x]
        initial_solution = self.ms.x.numpy().astype(np.float64).flatten()
        resdd = sopt.minimize(fun=self.step, x0=initial_solution, jac=True, tol=0,
                              method=self.optimizer_method, bounds=self.bounds, options=sp_options)
        print("Running SPOptimzer 2")
        return self.n_iter