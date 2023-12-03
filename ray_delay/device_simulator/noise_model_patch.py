import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from numpy.typing import NDArray
import numpy as np
import math
from ray_delay.device_simulator.stim_simulator.patch import SurfaceCodePatch
from ray_delay.device_simulator.noise_model import NoiseModel, NoiseParams, StandardIdenticalNoiseParams

import time
import os

class NoiseModelPatch:
    """TODO
    """
    def __init__(
            self, 
            patch: SurfaceCodePatch, 
            noise_params: NoiseParams | None = StandardIdenticalNoiseParams, 
            noise_model: NoiseModel | None = None,
            seed: int | None = None,
        ):
        """TODO
        """
        self.patch = patch
        if noise_model is not None:
            self.noise_model = noise_model
        else:
            assert noise_params is not None, 'Either noise_model or noise_params must be given.'
            self.noise_model = NoiseModel(patch, noise_params, seed=seed)

        self.patch.set_error_vals(self.noise_model.get_error_val_dict())

    def step(self, elapsed_time: float):
        """TODO"""
        raise NotImplementedError