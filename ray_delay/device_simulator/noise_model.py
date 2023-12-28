"""TODO
"""
import copy
from dataclasses import dataclass, field
import scipy
import numpy as np
from numpy.typing import NDArray
from ray_delay.device_simulator.stim_simulator.patch import SurfaceCodePatch

@dataclass
class NoiseParams:
    """Describes the noise affecting a system.
    
    Attributes:
        baseline_error_means: Mean error vals (gate errors, T1/T2).
        baseline_error_stdevs: Standard deviations of error vals.
        error_distributions_log: Whether each error val is distributed on log
            scale. 
        cosmic_ray_chance: Rate of cosmic ray impacts
        cosmic_ray_max_radius: Maximum radius of cosmic ray effect.
    """
    baseline_error_means: dict[str, float]
    cosmic_ray_chance: float = 0.1 / 27
    cosmic_ray_min_radius: float = 4
    cosmic_ray_max_radius: float = 4
    baseline_error_stdevs: dict[str, float] = field(default_factory = lambda: {
        'T1':0, 
        'T2':0, 
        'gate1_err':0, 
        'gate2_err':0, 
        'readout_err':0,
    })
    error_distributions_log: dict[str, bool] = field(default_factory = lambda: {
        'T1':False, 
        'T2':False, 
        'gate1_err':True, 
        'gate2_err':True, 
        'readout_err':True,
    })

StandardIdenticalNoiseParams = NoiseParams(
    {
        'T1':20e-6, 
        'T2':30e-6, 
        'gate1_err':-5, 
        'gate2_err':-4, 
        'readout_err':-4
    },
)

GoogleNoiseParams = NoiseParams(
    {
        'T1':20e-6, 
        'T2':30e-6, 
        'gate1_err':-4, # to give average gate error around 0.08%
        'gate2_err':-2.5, # to give average gate error around 0.5%
        'readout_err':-np.inf
    },
    baseline_error_stdevs= {
        'T1':2e-6,
        'T2':5e-6,
        'gate1_err':0.1,
        'gate2_err':0.1,
        'readout_err':0
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
            affected_qubits: list[int],
        ) -> None:
        """Create a cosmic ray.
        
        Args:
            affected_qubits: List of qubits that are affected (this is not used
                internally, but saved for convenience).
        """
        self.init_strength = 0.99 # from https://www.nature.com/articles/s41567-021-01432-8
        self.current_strength = self.init_strength
        self.time_alive = 0
        self.halflife = 30e-3 # from https://www.nature.com/articles/s41567-021-01432-8
        self.affected_qubits = affected_qubits

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
        affected_qubits = self.get_qubits_in_radius(center_qubit, radius)
        ray = CosmicRay(affected_qubits)
        self.active_cosmic_rays.append(ray)
        self.event_history.append(('COSMIC_RAY', (center_qubit, affected_qubits)))
        return ray

    def get_qubits_in_radius(
            self,
            center_qubit: int,
            radius: float,
        ) -> list[int]:
        """Return a list of all qubits within a certain radius of a qubit. List
        always contains center qubit, even for radius = 0. Searches
        self.qubit_layout. 
        
        Args:
            center_qubit: Center qubit index.
            radius: Radius to search within.
        
        Returns:
            List of qubit indices of qubits within radius.
        """
        qubits = [center_qubit]
        coords = np.argwhere(self.qubit_layout == center_qubit)[0]
        for r_offset in range(-int(np.floor(radius)), int(np.ceil(radius))+1):
            for c_offset in range(-int(np.floor(radius)), int(np.ceil(radius))+1):
                if (np.sqrt(r_offset**2 + c_offset**2) < radius
                    and coords[0]+r_offset >= 0
                    and coords[0]+r_offset < self.qubit_layout.shape[0]
                    and coords[1]+c_offset >= 0
                    and coords[1]+c_offset < self.qubit_layout.shape[1]):
                    qubit = int(self.qubit_layout[coords[0]+r_offset, coords[1]+c_offset])
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
                actual_error_vals['T1'][qubit] *= (1-cosmic_ray.current_strength)

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