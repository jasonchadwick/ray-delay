"""TODO
"""
import copy
from dataclasses import dataclass, field
import scipy
import numpy as np
from numpy.typing import NDArray
from stim_surface_code.patch import SurfaceCodePatch
from stim_surface_code.noise import NoiseParams

class CosmicRayNoiseParams(NoiseParams):
    """Describes the noise affecting a system.
    
    Attributes:
        baseline_error_means: Mean error vals (gate errors, T1/T2).
        baseline_error_stdevs: Standard deviations of error vals.
        error_distributions_log: Whether each error val is distributed on log
            scale. 
        cosmic_ray_chance: Rate of cosmic ray impacts per qubit.
        cosmic_ray_min_radius: Minimum radius of cosmic ray effect.
        cosmic_ray_max_radius: Maximum radius of cosmic ray effect.
    """
    baseline_error_means: dict[str, float]
    baseline_error_stdevs: dict[str, float]
    error_distributions_log: dict[str, bool]
    cosmic_ray_chance: float
    cosmic_ray_min_radius: float
    cosmic_ray_max_radius: float

    def __init__(
            self, 
            baseline_error_means: dict[str, float],
            baseline_error_stdevs: dict[str, float] = {
                'T1':0, 
                'T2':0, 
                'gate1_err':0, 
                'gate2_err':0, 
                'readout_err':0,
            },
            error_distributions_log: dict[str, bool] = {
                'T1':False, 
                'T2':False, 
                'gate1_err':True, 
                'gate2_err':True, 
                'readout_err':True,
            },
            cosmic_ray_chance: float = 0.1 / 27,
            cosmic_ray_min_radius: float = 4,
            cosmic_ray_max_radius: float = 4,
        ):
        """Initialize.

        Args:
            baseline_error_means: Mean error vals (gate errors, T1/T2).
            baseline_error_stdevs: Standard deviations of error vals.
            error_distributions_log: Whether each error val is distributed on log
                scale.
            cosmic_ray_chance: Rate of cosmic ray impacts per qubit.
            cosmic_ray_min_radius: Minimum radius of cosmic ray effect.
            cosmic_ray_max_radius: Maximum radius of cosmic ray effect.
        """
        self.baseline_error_means = baseline_error_means
        self.baseline_error_stdevs = baseline_error_stdevs
        self.error_distributions_log = error_distributions_log
        self.cosmic_ray_chance = cosmic_ray_chance
        self.cosmic_ray_min_radius = cosmic_ray_min_radius
        self.cosmic_ray_max_radius = cosmic_ray_max_radius

StandardIdenticalNoiseParams = CosmicRayNoiseParams(
    {
        'T1':20e-6, 
        'T2':30e-6, 
        'gate1_err':-5, 
        'gate2_err':-4, 
        'readout_err':-4
    },
)

GoogleIdenticalNoiseParams = CosmicRayNoiseParams(
    {
        'T1':20e-6, 
        'T2':30e-6, 
        'gate1_err':-4, # to give average gate error around 0.08%
        'gate2_err':-2.5, # to give average gate error around 0.5%
        'readout_err':-1.7
    },
)

GoogleNoiseParams = CosmicRayNoiseParams(
    {
        'T1':20e-6, 
        'T2':30e-6, 
        'gate1_err':-4, # to give average gate error around 0.08%
        'gate2_err':-2.5, # to give average gate error around 0.5%
        'readout_err':-1.7 # to give average around 2%
    },
    baseline_error_stdevs= {
        'T1':2e-6,
        'T2':5e-6,
        'gate1_err':0.1,
        'gate2_err':0.1,
        'readout_err':0.1,
    }
)

def get_T1T2_limited_params_by_err_rate(target_gate2_err_rate: float, gate2_time: float = 34e-9):
    """TODO
    """
    def get_gate2_err(t1t2_val):
        """TODO
        """
        p_x = max(0, 0.25 * (1 - np.exp(-gate2_time / t1t2_val)))
        p_y = p_x
        p_z = max(0, 0.5 * (1 - np.exp(-gate2_time / t1t2_val)) - p_x)
        return np.abs(1/target_gate2_err_rate - 1/(2*(p_x+p_y+p_z)))
    t1t2 = scipy.optimize.minimize(get_gate2_err, 100e-6).x[0]
    return t1t2

class CosmicRay:
    """TODO
    """
    def __init__(
            self,
            center_coords: tuple[int, int],
            affected_qubits: list[int],
            radius: float,
        ) -> None:
        """Create a cosmic ray.
        
        Args:
            center_qubit: Qubit at the center of the cosmic ray (or None if
                outside the patch).
            affected_qubits: List of qubits that are affected (this is not used
                internally, but saved for convenience).
        """
        self.init_strength = 0.99 # from https://www.nature.com/articles/s41567-021-01432-8
        self.current_strength = self.init_strength
        self.time_alive = 0.0
        self.halflife = 30e-3 # from https://www.nature.com/articles/s41567-021-01432-8
        self.center_coords = center_coords
        self.affected_qubits = affected_qubits
        self.radius = radius

    def step(self, elapsed_time: float):
        """TODO
        """
        self.time_alive += elapsed_time
        self.current_strength = self.init_strength * 0.5**(self.time_alive/self.halflife)

class NoiseModel:
    """Simulate the evolution of physical qubit error rates over time, as they
    fluctuate due to TLSs, constant drift, cosmic rays, etc.
    """
    def __init__(
            self,
            patch: SurfaceCodePatch,
            noise_params: NoiseParams,
            save_error_vals: bool = False,
            seed: int | None = None,
        ) -> None:
        """Initialize the noise model, giving it information about baseline
        error rates as well as rates of drift and disruption events.

        Args:
            num_qubits: Number of qubits.
            qubit_pairs: List of all qubit pairs for which a two-qubit gate is
                defined. 
            qubit_layout: 2D array encoding the positions of all physical
                qubits. 
            noise_params: NoiseParams object specifying all relevant noise
                values. 
            save_error_vals: If True, keep a record of past error values.
            seed: Seed for random number generator (controls noise events).
        """
        self.num_qubits = len(patch.all_qubits)
        self.qubit_pairs = patch.qubit_pairs
        self.qubit_layout: NDArray[np.int_] = np.array(
            [[(q.idx if q is not None else -1) for q in row]
             for row in patch.device],
            int
        )
        self.noise_params = noise_params

        self.error_max_vals = {
            'T1':np.inf,
            'T2':np.inf,
            'gate1_err':3/4,
            'gate2_err':15/16,
            'readout_err':1,
        }

        self.rng = np.random.default_rng(seed)

        self.event_history = []

        self.save_error_vals = save_error_vals
        self.error_val_history = []

        self._error_vals = self._initialize_error_vals()

        self.active_cosmic_rays: list[CosmicRay] = []
        self.cosmic_ray_delete_threshold = 1e-5

        self.qubit_distance_matrix = self._initialize_distance_matrix()
        self.constant_ray_effect = False

        if save_error_vals:
            self.error_val_history.append(self.get_error_val_dict())

    def reseed(self, seed: int):
        """TODO
        """
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """TODO
        """
        self.active_cosmic_rays = []
        self.error_val_history = []

    def step(
            self, 
            elapsed_time: float,
        ) -> dict[str, dict[int | tuple[int, int], float]]:
        """Advance the error model in time, adding drift and disruptions to the
        qubits.

        Args:
            time_elapsed: Time interval to simulate, in seconds.
        """
        self.event_history.append(('STEP', elapsed_time))

        # step cosmic rays
        self.step_cosmic_rays(elapsed_time)
        if self.noise_params.cosmic_ray_chance > 0:
            self.randomly_add_cosmic_rays(elapsed_time)

        error_val_dict = self.get_error_val_dict()
        if self.save_error_vals:
            self.error_val_history.append(error_val_dict)

        return error_val_dict

    def step_cosmic_rays(self, elapsed_time: float) -> None:
        """Simulate the decay of existing cosmic rays based on their half-lives.
        
        Args:
            elapsed_time: Time interval to simulate, in seconds.
        """
        kept_rays = []
        for ray in self.active_cosmic_rays:
            ray.step(elapsed_time)
            if ray.current_strength >= self.cosmic_ray_delete_threshold:
                kept_rays.append(ray)
        self.active_cosmic_rays = kept_rays

    def randomly_add_cosmic_rays(self, elapsed_time: float) -> None:
        """Randomly add new cosmic rays based on a pre-set chance and an amount
        of elapsed time to simulate.
        
        Args:
            elapsed_time: Time interval to simulate, in seconds.
        """
        for qubit in range(self.num_qubits):
            max_rays_added = 10
            num_rays_added = 0
            roll = self.rng.random()
            for i in range(max_rays_added+1):
                if i == max_rays_added:
                    prob = 1 - scipy.stats.poisson(self.noise_params.cosmic_ray_chance*elapsed_time).cdf(i-1)
                else:
                    prob = scipy.stats.poisson(self.noise_params.cosmic_ray_chance*elapsed_time).cdf(i)
                if roll < prob:
                    num_rays_added = i
                    break

            for i in range(num_rays_added):
                ray = self.add_cosmic_ray(
                    qubit, 
                    self.rng.random()*(self.noise_params.cosmic_ray_max_radius - self.noise_params.cosmic_ray_min_radius) + self.noise_params.cosmic_ray_min_radius,
                )
                ray_elapsed_time = self.rng.random() * elapsed_time
                ray.step(ray_elapsed_time)

    def add_cosmic_ray(
            self,
            center_qubit: int,
            radius: float,
        ) -> CosmicRay:
        """Create a new cosmic ray and add it to the noise model.
        
        Args:
            center_qubit: Center qubit of cosmic ray impact.
            radius: Radius of cosmic ray (in terms of qubit spacing).
        """
        affected_qubits = self.get_qubits_in_radius(radius, center_qubit=center_qubit)
        ray = CosmicRay(np.argwhere(self.qubit_layout == center_qubit)[0], affected_qubits, radius)
        self.active_cosmic_rays.append(ray)
        self.event_history.append(('COSMIC_RAY', (center_qubit, affected_qubits)))
        return ray

    def add_cosmic_ray_by_coords(
            self,
            coords: tuple[int, int],
            radius: float,
        ) -> CosmicRay:
        """Create a new cosmic ray and add it to the noise model.

        Args:
            coords: Coordinates of the center of the cosmic ray.
            radius: Radius of cosmic ray (in terms of qubit spacing).
        """
        affected_qubits = self.get_qubits_in_radius(radius, center_coords=coords)
        ray = CosmicRay(coords, affected_qubits, radius)
        self.active_cosmic_rays.append(ray)
        self.event_history.append(('COSMIC_RAY', (None, affected_qubits)))
        return ray

    def get_qubits_in_radius(
            self,
            radius: float,
            center_qubit: int | None = None,
            center_coords: tuple[int, int] | None = None,
        ) -> list[int]:
        """Return a list of all qubits within a certain radius of a qubit. List
        always contains center qubit, even for radius = 0. Searches
        self.qubit_layout. 
        
        Args:
            radius: Radius to search within.
            center_qubit: Center qubit index, or None if using center_coords.
            center_coords: Coordinates of the center qubit, or None if using
                center_qubit. Works even if coords do not correspond to a
                particular qubit, or if coords are beyond device edge
                boundaries.
        
        Returns:
            List of qubit indices of qubits within radius.
        """
        qubits = []
        if center_coords is None:
            qubits.append(center_qubit)
            center_coords = np.argwhere(self.qubit_layout == center_qubit)[0]
        for r_offset in range(-int(np.floor(radius)), int(np.ceil(radius))+1):
            for c_offset in range(-int(np.floor(radius)), int(np.ceil(radius))+1):
                if (np.sqrt(r_offset**2 + c_offset**2) < radius
                    and center_coords[0]+r_offset >= 0
                    and center_coords[0]+r_offset < self.qubit_layout.shape[0]
                    and center_coords[1]+c_offset >= 0
                    and center_coords[1]+c_offset < self.qubit_layout.shape[1]):
                    qubit = int(self.qubit_layout[center_coords[0]+r_offset, center_coords[1]+c_offset])
                    if qubit >= 0 and qubit != center_qubit:
                        qubits.append(qubit)
        return qubits

    def get_error_val_dict(self) -> dict[str, dict[int | tuple[int, int], float]]:
        """Get a dictionary containing error values that can be used to generate
        a Stim circuit.

        Returns:
            Dictionary where keys are error value names (e.g. 'gate1_err') and
            values are dictionaries mapping qubits or qubit pairs to their
            assigned error values.
        """
        actual_error_vals = copy.deepcopy(self._error_vals)

        # cosmic rays
        for cosmic_ray in self.active_cosmic_rays:
            for qubit in cosmic_ray.affected_qubits:
                if self.constant_ray_effect:
                    actual_error_vals['T1'][qubit] *= (1-cosmic_ray.current_strength)
                else:
                    # set T1 time so that decoherence error rate decreases
                    # linearly from center to edge. Modelling the linear radial
                    # decrease in error rates observed in Google paper (with
                    # sampling interval of 3us).
                    dist_from_center = self.qubit_distance_matrix[qubit, cosmic_ray.center_qubit]
                    scale_factor = ((cosmic_ray.radius - dist_from_center) / cosmic_ray.radius)
                    assert 0 <= scale_factor and scale_factor <= 1
                    t1_init = self._error_vals['T1'][qubit]
                    t1_worst = self._error_vals['T1'][qubit] * (1-cosmic_ray.current_strength)
                    scaled_t1 = -3e-6/np.log((1-scale_factor)*np.exp(-3e-6/t1_init) + scale_factor*np.exp(-3e-6/t1_worst))
                    actual_error_vals['T1'][qubit] = min(actual_error_vals['T1'][qubit], scaled_t1)

        return actual_error_vals

    def update_error_vals(
            self,
            update_dict: dict[str, dict[int | tuple[int, int], float]]
        ) -> dict[str, dict[int | tuple[int, int], float]]:
        """TODO
        """
        for val, updated_vals in update_dict.items():
            self._error_vals[val] = self._error_vals[val] | updated_vals
        return self.get_error_val_dict()

    def _initialize_distance_matrix(
            self,
        ) -> NDArray[np.float_]:
        """Create a matrix encoding the distance between qubits on the device.
        
        Returns:
            2D array of distances between qubits.
        """
        distance_matrix = np.full((self.num_qubits, self.num_qubits), np.nan)
        for r,row in enumerate(self.qubit_layout):
            for c,qubit in enumerate(row):
                if qubit >= 0:
                    for r2,row2 in enumerate(self.qubit_layout):
                        for c2,qubit2 in enumerate(row2):
                            if qubit2 >= qubit:
                                distance_matrix[qubit, qubit2] = np.sqrt((r-r2)**2 + (c-c2)**2)
                                distance_matrix[qubit2, qubit] = distance_matrix[qubit, qubit2]
        return distance_matrix

    def _initialize_error_vals(
            self,
        ) -> dict[str, dict[int | tuple[int, int], float]]:
        """Sample initial noise values from Gaussian distributions described by
        class attributes baseline_error_means and baseline_error_stdevs.

        Returns:
            Dictionary mapping error types to value assignments (for qubits or
            pairs). 
        """
        error_vals = {
            'T1': {q: self.rng.normal(
                    self.noise_params.baseline_error_means['T1'],
                    self.noise_params.baseline_error_stdevs['T1']
                )
                for q in range(self.num_qubits)
            },
            'T2': {q: self.rng.normal(
                    self.noise_params.baseline_error_means['T2'],
                    self.noise_params.baseline_error_stdevs['T2']
                )
                for q in range(self.num_qubits)
            },
            'readout_err': {q: self.rng.normal(
                    self.noise_params.baseline_error_means['readout_err'],
                    self.noise_params.baseline_error_stdevs['readout_err']
                )
                for q in range(self.num_qubits)
            },
            'gate1_err': {q: self.rng.normal(
                    self.noise_params.baseline_error_means['gate1_err'],
                    self.noise_params.baseline_error_stdevs['gate1_err']
                )
                for q in range(self.num_qubits)
            },
            'gate2_err': {qubit_pair: self.rng.normal(
                    self.noise_params.baseline_error_means['gate2_err'],
                    self.noise_params.baseline_error_stdevs['gate2_err']
                )
                for qubit_pair in self.qubit_pairs
            },
        }

        for k,assignments in error_vals.items():
            if self.noise_params.error_distributions_log[k]:
                # apply exponent if mean/stdev values are on log scale
                error_vals[k] = {q:10**v for q,v in assignments.items()}
            # clip to min/max allowed
            error_vals[k] = {q:np.clip(v, 0, self.error_max_vals[k])
                             for q,v in error_vals[k].items()}

        return error_vals