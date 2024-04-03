"""TODO
"""
from stim_surface_code.patch import SurfaceCodePatch
from ray_delay.noise_model import NoiseModel, CosmicRayNoiseParams, GoogleNoiseParamsNoRandomRays

class NoiseModelPatch:
    """TODO

    TODO: multiple patches, one noise model
    """
    def __init__(
            self, 
            patch: SurfaceCodePatch, 
            noise_params: CosmicRayNoiseParams = GoogleNoiseParamsNoRandomRays.improve(10),
            noise_model: NoiseModel | None = None,
            seed: int | None = None,
        ):
        """Initialize the device.

        Args:
            patch: The SurfaceCodePatch object to use.
            noise_params: The noise parameters to use to generate a NoiseModel
                object, if none is explicitly given.
            noise_model: The NoiseModel object to use. Overrides noise_params.
            seed: The seed to use for NoiseModel generation if noise_model is
                None.
        """
        self.patch = patch
        self.noise_params = noise_params
        if noise_model is not None:
            self.noise_model = noise_model
        else:
            assert noise_params is not None, 'Either noise_model or noise_params must be given.'
            self.noise_model = NoiseModel(patch, noise_params, seed=seed)

        self.patch.set_error_vals(self.noise_model.get_error_val_dict())

    def step(self, elapsed_time: float):
        """Step the device forward in time.

        Args:
            elapsed_time: The amount of time to step forward.
        """
        self.noise_model.step(elapsed_time)
        self.patch.set_error_vals(self.noise_model.get_error_val_dict())

    def force_cosmic_ray(self, center_qubit: int, radius: float | None = None):
        """Create a cosmic ray in the device noise model.

        Args:
            center_qubit: The qubit at the center of the cosmic ray.
            radius: The radius of the cosmic ray. If not given, use default.
        """
        self.noise_model.add_cosmic_ray(center_qubit, radius)
        self.step(0)

    def force_cosmic_ray_by_coords(self, coords: tuple[int, int], radius: float | None = None):
        """Create a cosmic ray outside the patch in the device noise model.

        Args:
            coords: The device coordinates of center of the cosmic ray.
            radius: The radius of the cosmic ray. If not given, use default.
        """
        self.noise_model.add_cosmic_ray_by_coords(coords, radius)
        self.step(0)

    def reset(self):
        """Reset the noise model.
        """
        self.noise_model.reset()
        self.patch.set_error_vals(self.noise_model.get_error_val_dict())