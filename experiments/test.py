import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import mpmath
import dill as pickle

from ray_delay.noise_model_patch import NoiseModelPatch
from ray_delay.ray_detector import RayDetectorSpec, RayImpactSimulator
from stim_surface_code.memory import MemoryPatch
from stim_surface_code.patch import Qubit, DataQubit, MeasureQubit

dx = 11
dz = 11
dm = 7

patch = NoiseModelPatch(MemoryPatch(dx, dz, dm))
patch.noise_model.save_error_vals = True

sim = RayImpactSimulator(patch, spatial_window_size=2, only_full_windows=True)

spec, baseline_fracs, ray_fracs = sim.generate_detector_spec(window_fpr=mpmath.mpf(1e-9), cycles_per_distillation=42, decay_nsteps=10)