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
edge_img  = np.load("inputdata\Sample2MineralMaps32x32\\32x32Viridis\\npy32x32Granite3XRF180degrees.npy")

#np.fliplr(np.load("inputdata\Sample2MineralMaps32x32\\32x32Viridis\\npy32x32Granite3XRF180degrees.npy"))

combine_edges = False

if combine_edges:
    replacement_row = np.flipud(np.fliplr(np.load("results\last_frame.npy"))[-1,:])
    edge_img[0, :] = replacement_row


def one_hot_encode(image: np.ndarray, num_classes: int = number_of_minerals) -> np.ndarray:
    return np.eye(num_classes, dtype=np.uint8)[image]

top = True
bottom = True
left = True
right = True
flip_vert = False # Note, choose either flip_vert or flip_horz
flip_horz = False

if flip_vert == False and flip_horz == False:
    edge_encoded = one_hot_encode(edge_img)
elif flip_vert == True and flip_horz == False:
    edge_encoded = one_hot_encode(np.flipud(edge_img))
elif flip_vert == False and flip_horz == True:
    edge_encoded = one_hot_encode(np.fliplr(edge_img))
elif flip_vert == True and flip_horz == True:
    print('Warning: both flip_vert and flip_horz is True')
    edge_encoded = one_hot_encode(np.flipud(np.fliplr(edge_img)))


def apply_outer_edge_to_image(target_img: np.ndarray) -> np.ndarray:
    result = target_img.copy()
    if top:
        result[0, :, :]     = edge_encoded[0, :, :]   # Top row
    if bottom:
        result[-1, :, :]    = edge_encoded[-1, :, :]  # Bottom row
    if left:
        result[:, 0, :]     = edge_encoded[:, 0, :]   # Left column
    if right:
        result[:, -1, :]    = edge_encoded[:, -1, :]  # Right column

    return result

class SPOptimizer2D(Optimizer):
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

        print("Boundaries applied")
        # Apply fixed boundary edges
        new_val = self.ms.numpy()
        new_val = apply_outer_edge_to_image(new_val)
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