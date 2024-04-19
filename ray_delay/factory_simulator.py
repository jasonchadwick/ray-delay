"""TODO
"""
import numpy as np
from numpy.typing import NDArray
import scipy
from dataclasses import dataclass
import itertools
import mpmath
import pickle
import os

import qc_utils.stats
from ray_delay.ray_detector import RayDetectorSpec
from stim_surface_code import patch

class MagicStateFactory:
    """TODO
    """
    def __init__(
            self, 
            dx: int = 7,
            dz: int = 7,
            dm: int = 7,
            patch_offline_duration: float = 30e-3,
            cycle_time: float | None = None,
            cache_cycles_per_distillation: bool = False,
            cache_filename: str | None = None,
            rng: int | np.random.Generator | None = None,
        ):
        """Initializes the factory.

        Args:
            dx: X code distance for each patch.
            dz: Z code distance for each patch.
            dm: Temporal code distance.
            patch_offline_duration: Amount of time (in seconds) a patch is taken
                offline when a ray is detected.
            cycle_time: Duration of a single surface code stabilizer measurement
                cycle. If None, will be calculated based on the code distance.
            cache_cycles_per_distillation: If True, store results of previous
                calls to _cycles_per_distillation() to avoid redundant
                calculations.
            cache_filename: Filename to use for caching solutions to 
                _cycles_per_distillation(). If None, will not load or save
                cache.
            rng: Seed for the random number generator, or a numpy random number
                generator. If None, will use the default numpy random number
                generator.
        """
        self.dx = dx
        self.dz = dz
        self.dm = dm
        self.patch_offline_duration = patch_offline_duration

        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        self.rng = rng

        num_rows = 3
        num_cols = 5
        row_heights = np.full(num_rows, self.dx)
        col_widths = np.array([self.dx] + [self.dz]*4)

        self.num_patches = num_rows * num_cols
        self.patch_indices = np.reshape(np.arange(self.num_patches), (num_rows, num_cols))
        self.patch_coords_from_idx = np.zeros((self.num_patches, 2), int)
        for idx in range(self.num_patches):
            self.patch_coords_from_idx[idx] = [idx // self.patch_indices.shape[1], idx % self.patch_indices.shape[1]]

        # use stim_surface_code.patch to generate the physical qubit array
        # TODO: currently does not account for extra 4*dm space needed for magic
        # state injection (see figs. 10-11 of Litinski). Is this important?
        surface_code_patch = patch.SurfaceCodePatch(num_rows*dx + (num_rows-1), dx + 4*dz + (num_cols-1), dm)
        self.physical_qubit_array = np.array([[(q.idx if q is not None else -1) for q in row] for row in surface_code_patch.device], int)
        self.physical_qubit_coords_from_idx = np.zeros((len(surface_code_patch.all_qubits), 2), int)
        for q in surface_code_patch.all_qubits:
            self.physical_qubit_coords_from_idx[q.idx] = q.coords
        if cycle_time is None:
            self.cycle_time = surface_code_patch.cycle_time()
        else:
            self.cycle_time = cycle_time

        # maps each physical qubit to a patch index
        self.num_phys_qubits = len(surface_code_patch.all_qubits)
        self.patch_idx_from_physical_qubit_idx = np.full(self.num_phys_qubits, -1, dtype=int)
        for row in range(num_rows):
            min_phys_row = np.sum(2*row_heights[:row]+1) + row
            max_phys_row = min_phys_row + 2*row_heights[row]+1
            for col in range(num_cols):
                min_phys_col = np.sum(2*col_widths[:col]+1) + col
                max_phys_col = min_phys_col + 2*col_widths[col]+1
                for phys_row in range(min_phys_row, max_phys_row):
                    for phys_col in range(min_phys_col, max_phys_col):
                        phys_idx = self.physical_qubit_array[phys_row, phys_col]
                        if phys_idx != -1:
                            self.patch_idx_from_physical_qubit_idx[phys_idx] = self.patch_indices[row, col]

        self.physical_qubit_offline_time_remaining = np.zeros(self.num_phys_qubits, float)

        self.prev_patches_online = None
        self.prev_distillation_cycles = None

        self._cache_cycles_per_distillation = cache_cycles_per_distillation
        self.cache_filename = cache_filename
        self.cache_queries = 0
        self.cache_hits = 0
        self._cycles_per_distillation_cache = {}
        self.load_cache()

    def load_cache(
            self,
            cache_filename: str | None = None,
        ):
        """Load the cache of _cycles_per_distillation() from a file. Does
        nothing if the file does not exist.

        Args:
            cache_filename: Filename to load the cache from.
        """
        if cache_filename is None:
            if self.cache_filename is None:
                return
            cache_filename = self.cache_filename
        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                self._cycles_per_distillation_cache = pickle.load(f)

    def save_cache(
            self,
            cache_filename: str | None = None,
        ):
        """Save the cache of _cycles_per_distillation() to a file. Overwrites
        any existing file.

        Args:
            cache_filename: Filename to save the cache to.
        """
        if cache_filename is None:
            if self.cache_filename is None:
                return
            cache_filename = self.cache_filename
        with open(cache_filename, 'wb') as f:
            pickle.dump(self._cycles_per_distillation_cache, f)

    def calculate_avg_overhead_per_ray(
            self,
            ray_detector_spec: RayDetectorSpec,
            num_rays: int,
            num_distributions: int = 10,
            prob_cutoff: float = 1e-7,
            save_cache: bool = True,
        ): 
        """Simulate the impact of a number of cosmic rays on the factory

        TODO: rays centered just outside of factory?
        TODO: false positives
        
        Args:
            ray_detector_spec: Contains information about ray and detector
                behavior.
            num_rays: Number of ray impacts to simulate.
            num_distributions: Number of qubit detection distributions to sample
                for each ray impact.
            prob_cutoff: Minimum probability of a distribution for it to be
                considered.
            save_cache: Whether to save the cache after calculating the
                overhead.
            
        Returns:
            Average additional cycle cost of a ray impact.
        """
        self._reset()
        baseline_cycles = self._cycles_per_distillation()
        assert baseline_cycles is not None
        ray_results = self.simulate_ray_impacts(ray_detector_spec, num_rays)
        time_overheads = []
        offline_chances = []
        # for each ray result, calculate expected (average case) patch offline time
        for j,ray_result in enumerate(ray_results):
            qubit_detection_chances = ray_result[1]
            assert np.all(qubit_detection_chances >= 0) and np.all(qubit_detection_chances <= 1), (qubit_detection_chances.min(), qubit_detection_chances.max())

            # qubits_offline_list, distribution_chances =
            # qc_utils.stats.get_most_probable_bitstrings(qubit_detection_chances,
            # 10, probability_threshold=1e-3)
            qubits_offline_most_likely = qubit_detection_chances > 0.5
            qubits_offline_array = self.rng.random((num_distributions-1, len(qubit_detection_chances))) < qubit_detection_chances[None, :][[0]*(num_distributions-1)]
            qubits_offline_array = np.concatenate([qubits_offline_array, qubits_offline_most_likely[None, :]])
            
            time_overhead = 0.0
            total_prob = 0
            offline_chances.append(0.0)
            for i,qubits_offline in enumerate(qubits_offline_array):
                distribution_chance = np.prod(qubit_detection_chances[qubits_offline]) * np.prod(1-qubit_detection_chances[~qubits_offline])
                if distribution_chance < prob_cutoff:
                    continue
                self.physical_qubit_offline_time_remaining = qubits_offline.astype(float)
                distillation_cycles = self._cycles_per_distillation()
                if distillation_cycles is None:
                    offline_chances[-1] += distribution_chance
                else:
                    time_overhead += distillation_cycles / baseline_cycles * distribution_chance
                total_prob += distribution_chance
            if total_prob == 0:
                offline_chances.pop()
                continue
            working_total_prob = total_prob - offline_chances[-1]
            if working_total_prob > 0:
                time_overhead /= working_total_prob
                time_overheads.append(time_overhead)
            offline_chances[-1] /= total_prob

        self._reset()
        if save_cache:
            self.save_cache()

        return np.mean(time_overheads), np.mean(offline_chances), ray_results
    
    def calculate_avg_overhead_one_patch_offline(
            self,
        ): 
        """TODO
            
        Returns:
            Average additional cycle cost when one patch is taken offline.
        """
        self._reset()
        baseline_distillation_cycles = self._cycles_per_distillation()
        assert baseline_distillation_cycles is not None
        time_overheads = []
        offline_count = 0
        # for each ray result, calculate expected (average case) patch offline time
        for j in range(self.num_patches):
            self.patch_offline_time_remaining = np.zeros(self.num_patches, float)
            self.patch_offline_time_remaining[j] = self.patch_offline_duration
            time_overhead = 0.0
            
            distillation_cycles = self._cycles_per_distillation()
            if distillation_cycles is None:
                offline_count += 1
            else:
                time_overheads.append(distillation_cycles / baseline_distillation_cycles)
        self._reset()

        self.save_cache()

        return np.mean(time_overheads), offline_count / self.num_patches

    def calculate_avg_overhead_one_phys_qubit_offline(
            self,
        ): 
        """TODO
            
        Returns:
            Average additional cycle cost when one patch is taken offline.
        """
        # this will be way too expensive unless caching is enabled
        assert self._cache_cycles_per_distillation
        self._reset()
        baseline_distillation_cycles = self._cycles_per_distillation()
        assert baseline_distillation_cycles is not None
        time_overheads = []
        offline_count = 0
        # for each ray result, calculate expected (average case) patch offline time
        for j in range(self.num_phys_qubits):
            self.physical_qubit_offline_time_remaining = np.zeros(self.num_phys_qubits, float)
            self.physical_qubit_offline_time_remaining[j] = self.patch_offline_duration
            time_overhead = 0.0
            
            distillation_cycles = self._cycles_per_distillation()
            if distillation_cycles is None:
                offline_count += 1
            else:
                time_overheads.append(distillation_cycles / baseline_distillation_cycles)
        self._reset()

        self.save_cache()

        return np.mean(time_overheads), offline_count / self.num_phys_qubits, time_overheads

    def _reset(self):
        """Reset the factory to its initial state."""
        self.physical_qubit_offline_time_remaining = np.zeros(self.num_phys_qubits, float)

    def simulate(
            self, 
            num_distillations: int,
            ray_incidence_rate: float,
            ray_detector_spec: RayDetectorSpec,
            patch_offline_time: float,
            use_mpmath: bool = True,
        ):
        """Simulate the performance of the factory over a number of rounds, with
        cosmic rays.
        
        Args:
            num_distillations: Number of distillations to simulate.
            ray_incidence_rate: Chance of a ray per qubit per second.
            ray_detector_spec: Contains information about ray and detector
                behavior.
            patch_offline_time: Amount of time (in seconds) a patch is taken
                offline when a ray is detected.
            use_mpmath: Whether to use mpmath for higher precision.
            rng_seed: Seed for the random number generator.
            
        Returns:
            TODO
        """
        self._reset()

        ray_remove_time = 5*ray_detector_spec.ray_halflife

        # calculate baseline chances of patches being turned offline (when no
        # ray present)
        baseline_patch_offline_chances = self._calc_patch_offline_chances(ray_detector_spec, use_mpmath=use_mpmath)

        elapsed_time = 0.0
        last_distillation_elapsed_time = 0.0
        elapsed_time_per_distillation = []
        cosmic_ray_history = []
        active_cosmic_rays = []
        event_history = []
        distillations_accepted = []
        distillations_remaining = num_distillations
        while distillations_remaining > 0:
            self.patch_offline_time_remaining[self.patch_offline_time_remaining > 0.0] -= last_distillation_elapsed_time
            self.patch_offline_time_remaining[self.patch_offline_time_remaining < 0.0] = 0.0
            for p in np.where(np.isclose(self.patch_offline_time_remaining, 0.0))[0]:
                event_history.append(('PATCH_ONLINE', p, elapsed_time, distillations_remaining))
                self.patch_offline_time_remaining[p] = 0.0
            # remove old rays
            active_cosmic_rays = [ray for ray in active_cosmic_rays if elapsed_time-ray[0] < ray_remove_time]

            # determine whether we can distill
            cycles_per_distillation = self._cycles_per_distillation()
            wait_time = 0.0
            if cycles_per_distillation is None:
                # factory is offline
                patches_offline = (self.patch_offline_time_remaining > 0.0)
                wait_time = self._wait_for_factory_to_come_online()
                event_history.append(('WAIT', wait_time, elapsed_time, distillations_remaining))
                patches_that_came_online = (self.patch_offline_time_remaining == 0.0) & patches_offline
                for p in np.where(patches_that_came_online)[0]:
                    event_history.append(('PATCH_ONLINE', p, elapsed_time+wait_time, distillations_remaining))
                cycles_per_distillation = self._cycles_per_distillation()
                assert cycles_per_distillation is not None
            time_per_distillation = cycles_per_distillation * self.cycle_time + wait_time

            # generate new cosmic rays
            num_cosmic_rays = self.rng.poisson(ray_incidence_rate * time_per_distillation * self.num_phys_qubits)
            new_rays_this_round = []
            for i in range(num_cosmic_rays):
                center_qubit = self.rng.choice(np.arange(self.num_phys_qubits))
                event_history.append(('RAY', center_qubit, elapsed_time, distillations_remaining))
                new_rays_this_round.append((elapsed_time, center_qubit))

            elapsed_time += time_per_distillation

            patch_no_signal_chances = 1-baseline_patch_offline_chances
            # calculate chances of signals due to new rays
            for i,ray in enumerate(new_rays_this_round):
                patch_no_signal_chances *= np.prod([1-self._calc_patch_offline_chances(ray_detector_spec, ray_incidence_qubit=ray[1], cycles_after_ray_impact=c, use_mpmath=use_mpmath) for c in range(cycles_per_distillation)], axis=0)
            # calculate chances of signals due to old rays
            for i,ray in enumerate(active_cosmic_rays):
                patch_no_signal_chances *= (1-self._calc_patch_offline_chances(ray_detector_spec, ray_incidence_qubit=ray[1], time_after_ray_impact=elapsed_time-ray[0], use_mpmath=use_mpmath))**cycles_per_distillation
            patch_signal_chances = 1-patch_no_signal_chances
            
            # randomly decide if signals are generated; if so, discard
            # distillation and turn patches offline. Note: patches that are
            # already offline can still be triggered again; this will reset
            # their offline time.
            patch_signal_decisions = self.rng.random(self.num_patches) < patch_signal_chances
            if np.any(patch_signal_decisions):
                for p in np.where(patch_signal_decisions)[0]:
                    # record event if patch was not already offline
                    if self.patch_offline_time_remaining[p] == 0.0:
                        event_history.append(('PATCH_OFFLINE', p, elapsed_time, distillations_remaining))
                self.patch_offline_time_remaining[patch_signal_decisions] = patch_offline_time
                distillations_accepted.append(False)
            else:
                distillations_remaining -= 1
                distillations_accepted.append(True)
            
            elapsed_time_per_distillation.append(time_per_distillation)
            last_distillation_elapsed_time = time_per_distillation
            cosmic_ray_history += new_rays_this_round
            active_cosmic_rays += new_rays_this_round
        return elapsed_time, elapsed_time_per_distillation, event_history, cosmic_ray_history, distillations_accepted

    def simulate_ray_impacts(
            self,
            ray_detector_spec: RayDetectorSpec,
            num_rays: int | None = None,
            use_mpmath: bool = False,
            rng_seed: int | None = None,
        ):
        """Simulate the detection of a number of cosmic rays on the factory. Can
        be used to calculate the average overhead for the factory.
        
        Args:
            num_rays: Number of ray impacts to simulate. If None, simulate once
                for each physical qubit.
            ray_detector_spec: Contains information about ray and detector
                behavior.
            use_mpmath: Whether to use mpmath for higher precision.
            rng_seed: Seed for the random number generator.
            rng: Random number generator to use.
            tail_batch_duration: Discretization of the detection simulation
                during the exponential decay of the ray. Smaller values give
                more accurate results, but take longer to compute.
        
         Returns:
            TODO
        """
        results = []
        
        # center of each ray is a randomly-chosen qubit
        # TODO: allow for rays centered just outside of factory
        if num_rays is None or num_rays >= self.num_phys_qubits:
            impacted_qubits = np.arange(self.num_phys_qubits)
        else:
            impacted_qubits = self.rng.choice(np.arange(self.num_phys_qubits), num_rays, replace=False)

        for q in impacted_qubits:
            qubit_detection_chances = self._calc_qubit_offline_chances_in_first_distillation(ray_detector_spec, q, use_mpmath=use_mpmath)
            results.append((q, qubit_detection_chances))

        return results

    def _cycles_per_distillation(
            self,
        ) -> int | None:
        """Calculate the number of cycles required for one magic state
        distillation, based on the current state of
        self.patch_offline_time_remaining.

        This default function return 6*self.dm if all patches are online, and
        None otherwise. Subclasses may override this function to implement more
        complex behavior.
        
        Returns:
            The number of surface code stabilizer measurement cycles required
            for one magic state distillation, or None if the factory cannot
            currently produce magic states.
        """
        physical_qubits_online = (self.physical_qubit_offline_time_remaining <= 0.0)

        cycle_count = None
        if np.all(physical_qubits_online):
            # operating at full capacity
            return 6*self.dm
        else:
            # some patches are offline, and we haven't seen it before
            return None

    def _wait_for_factory_to_come_online(self):
        """Wait for the factory to come online, and return the amount of time
        waited. Requires that self._cycles_per_distillation() is None. Modifies
        self.patch_offline_time_remaining.
        
        Returns:
            Amount of time waited (in seconds). Once this function returns,
            self._cycles_per_distillation() will not be None.
        """
        assert self._cycles_per_distillation() is None
        wait_time = 0.0
        patches_in_order = np.argsort(self.patch_offline_time_remaining)
        for delay_time in np.min(self.patch_offline_time_remaining) + np.diff(self.patch_offline_time_remaining[patches_in_order]):
            wait_time += delay_time
            self.patch_offline_time_remaining -= delay_time
            self.patch_offline_time_remaining[self.patch_offline_time_remaining < 0.0] = 0.0
            self.patch_offline_time_remaining[np.isclose(self.patch_offline_time_remaining, 0.0)] = 0.0
            if self._cycles_per_distillation() is not None:
                return wait_time
        # should never reach this point
        print(self.patch_offline_time_remaining, self._cycles_per_distillation())
        raise Exception('Factory never came online.')

    def _calc_qubit_offline_chances_in_first_distillation(
            self,
            ray_detector_spec: RayDetectorSpec,
            ray_incidence_qubit: int | None = None,
            use_mpmath: bool = False,
        ) -> NDArray:
        """Calculate the chance that each qubit is turned offline due to a ray
        event.

        Args:
            ray_incidence_qubit: Index of the qubit where the ray is incident.
            ray_detector_spec: Argument to pass on to RayDetectorSpec.
        
        Returns:
            1D array of length self.num_patches, where each entry is the chance
            that the corresponding patch is taken offline due to a ray event.
        """
        if ray_incidence_qubit is None:
            physical_qubit_distances_from_ray = np.full(len(self.patch_idx_from_physical_qubit_idx), 1e10)
        else:
            ray_incidence_coords = self.physical_qubit_coords_from_idx[ray_incidence_qubit]
            physical_qubit_distances_from_ray = np.linalg.norm(self.physical_qubit_coords_from_idx - ray_incidence_coords, axis=1)
            assert physical_qubit_distances_from_ray.shape == (len(self.patch_idx_from_physical_qubit_idx),)
        
        qubit_detection_chances = ray_detector_spec.first_distillation_chance(physical_qubit_distances_from_ray)
        if use_mpmath:
            qubit_detection_chances = np.array([mpmath.mpf(x) for x in qubit_detection_chances], dtype=mpmath.mpf)

        return qubit_detection_chances

def boolean_array_BFS(
        array: NDArray, 
        start: tuple[int, int], 
        end: tuple[int, int] | None = None,
    ) -> dict[tuple[int, int], list[tuple[int, int]]] | list[tuple[int, int]] | None:
    """Use breadth-first search to find a path through the array from start to
    end.

    Args:
        array: 2D boolean array, where False values are considered obstacles.
        start: Starting coordinates.
        end: Ending coordinates, or None to find all paths.
    
    Returns:
        If end is None, a dictionary of all paths from start to any True
        coordinate pair in the array. If end is not None, a list of coordinates
        representing the shortest path from start to end, or None if there is no
        path. 
    """
    frontier = [start]
    paths_to = {start: [start]}
    while len(frontier) > 0:
        current = frontier[0]
        frontier = frontier[1:]
        neighbors = [
            (current[0]+1, current[1]), 
            (current[0]-1, current[1]), 
            (current[0], current[1]+1), 
            (current[0], current[1]-1)
        ]
        for n in neighbors:
            if n[0] >= 0 and n[0] < array.shape[0] and n[1] >= 0 and n[1] < array.shape[1] and array[n]:
                if n not in paths_to:
                    # first time visiting n
                    paths_to[n] = paths_to[current] + [n]
                    frontier.append(n)
                    if n == end:
                        break
                else:
                    # already visited n
                    if len(paths_to[current] + [n]) < len(paths_to[n]):
                        paths_to[n] = paths_to[current] + [n]

    if end is None:
        return paths_to
    else:
        if end not in paths_to:
            return None
        else:
            return paths_to[end]

def boolean_array_all_pairs_BFS(
        array: NDArray,
    ) -> dict[tuple[int, int], dict[tuple[int, int], list[tuple[int, int]]]]:
    """Compute all pairs shortest paths.
    
    Args:
        array: 2D boolean array, where False values are considered obstacles.

    Returns:
        Dictionary of dictionaries, where the outer dictionary maps starting
        coordinates to inner dictionaries, which map ending coordinates to paths
        from starting coordinates to ending coordinates.
    """
    all_paths = {}
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j]:
                all_paths[(i,j)] = boolean_array_BFS(array, (i,j))
    return all_paths

class Redundant15To1(MagicStateFactory):
    """TODO
    """
    def __init__(
            self, 
            dx: int = 7,
            dz: int = 7,
            dm: int = 7,
            patch_offline_duration: float = 30e-3,
            cycle_time: float | None = None,
            cache_cycles_per_distillation: bool = False,
            num_redundant_cols: int = 0,
            redundant_top_routing_space: int = 0,
            redundant_bot_routing_space: int = 0,
            mapping_mode: str = 'simple',
            rng: int | np.random.Generator | None = None,
        ):
        """Initializes the factory, using a layout from litinski_magic_2019.

        Args:
            dx: X code distance for each patch.
            dz: Z code distance for each patch.
            dm: Temporal code distance.
            patch_offline_duration: Amount of time (in seconds) a patch is taken
                offline when a ray is detected.
            cycle_time: Duration of a single surface code stabilizer measurement
                cycle. If None, will be calculated based on the code distance.
            cache_cycles_per_distillation: If True, store results of previous
                calls to _cycles_per_distillation() to avoid redundant
                calculations.
            num_redundant_cols: Number of redundant patches to include.
            redundant_top_routing_space: Number of redundant rows to include
                above the top routing space.
            redundant_bot_routing_space: Number of redundant rows to include
                below the bottom routing space.
            mapping_mode: 'simple', 'rearrange_greedy', or 'rearrange_full'. Method
                'simple' does not rearrange logical central patches, and simply
                checks to see if the patches can still communicate with each
                other. Method 'rearrange_greedy' rearranges logical central
                patches into a new configuration, but does not fully optimize
                the new configuration for cycle cost. Method 'rearrange_full'
                fully optimizes the new configuration for cycle cost.
            rng: Seed for the random number generator, or a numpy random number
                generator. If None, will use the default numpy random number
                generator.
        """
        super().__init__(
            dx, 
            dz, 
            dm, 
            patch_offline_duration, 
            cycle_time, 
            cache_cycles_per_distillation,
            f'data/mapping_cache_{mapping_mode}.pkl',
            rng,
        )

        self.patch_offline_duration = patch_offline_duration

        self.logical_qubit_row = 1 + redundant_top_routing_space

        num_rows = 3 + redundant_top_routing_space + redundant_bot_routing_space
        num_cols = 5 + num_redundant_cols
        self.current_wide_columns = [False]*num_cols
        self.current_wide_columns[0] = True
        row_heights = np.full(num_rows, self.dx)
        col_widths = np.array([self.dx]*1 + [self.dz]*(4+num_redundant_cols))

        self.num_patches = num_rows * num_cols
        self.patch_indices = np.reshape(np.arange(self.num_patches), (num_rows, num_cols))
        self.patch_coords_from_idx = np.zeros((self.num_patches, 2), int)
        for idx in range(self.num_patches):
            self.patch_coords_from_idx[idx] = [idx // self.patch_indices.shape[1], idx % self.patch_indices.shape[1]]

        # use stim_surface_code.patch to generate the physical qubit array
        # TODO: currently does not account for extra 4*dm space needed for magic
        # state injection (see figs. 10-11 of Litinski). Is this important?
        surface_code_patch = patch.SurfaceCodePatch(sum(row_heights) + (num_rows-1), sum(col_widths) + (num_cols-1), dm)
        self.physical_qubit_array = np.array([[(q.idx if q is not None else -1) for q in row] for row in surface_code_patch.device], int)
        self.physical_qubit_coords_from_idx = np.zeros((len(surface_code_patch.all_qubits), 2), int)
        for q in surface_code_patch.all_qubits:
            self.physical_qubit_coords_from_idx[q.idx] = q.coords
        if cycle_time is None:
            self.cycle_time = surface_code_patch.cycle_time()
        else:
            self.cycle_time = cycle_time

        self.num_phys_qubits = len(surface_code_patch.all_qubits)
        self.physical_qubit_offline_time_remaining = np.zeros(self.num_phys_qubits, float)

        self.prev_optimization_key = None
        self.prev_distillation_cycles = 6

        # see Litinski Fig. 3
        self.applied_rotations = np.array([
            [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
            [1,1,0,0,1,0,1,0,1,0,0,0,1,0,1,1],
            [1,0,1,0,0,1,1,0,0,1,0,0,1,1,0,1],
            [0,1,1,1,0,0,1,0,0,0,1,0,1,1,1,0],
            [0,0,0,1,1,1,1,0,0,0,0,1,0,1,1,1]
        ])
        # These rotations must be applied before the T state is moved out.
        self.initial_rotations = self.applied_rotations[:, :6]
        # These rotations can be applied after the T state is moved out.
        self.final_rotations = self.applied_rotations[:, 6:]

        self.mapping_mode = mapping_mode

        self._phys_qubit_patch_mapping_cache = dict()

    def _get_phys_qubit_patch_mapping(
            self,
            wide_columns: list[bool],
        ):
        """TODO
        """
        cache_key = tuple(wide_columns)
        if cache_key in self._phys_qubit_patch_mapping_cache:
            return self._phys_qubit_patch_mapping_cache[cache_key]

        col_widths = np.array([self.dx if wide_columns[i] else self.dz for i in range(self.patch_indices.shape[1])])
        patch_idx_from_physical_qubit_idx = np.full(self.num_phys_qubits, -1, dtype=int)
        physical_qubits_from_patch_idx = [[] for _ in range(self.num_patches)]
        for row in range(self.patch_indices.shape[0]):
            min_phys_row = np.sum(2*self.dx*row+1) + row
            max_phys_row = min_phys_row + 2*self.dx+1
            for col in range(self.patch_indices.shape[1]):
                min_phys_col = np.sum(2*col_widths[:col]+1) + col
                max_phys_col = min_phys_col + 2*col_widths[col]+1
                for phys_row in range(min_phys_row, max_phys_row):
                    for phys_col in range(min_phys_col, max_phys_col):
                        phys_idx = self.physical_qubit_array[phys_row, phys_col]
                        if phys_idx != -1:
                            patch_idx_from_physical_qubit_idx[phys_idx] = self.patch_indices[row, col]
                            physical_qubits_from_patch_idx[self.patch_indices[row, col]].append(phys_idx)

        self._phys_qubit_patch_mapping_cache[cache_key] = patch_idx_from_physical_qubit_idx, physical_qubits_from_patch_idx
        return patch_idx_from_physical_qubit_idx, physical_qubits_from_patch_idx
    
    def _get_patches_online(
            self,
            wide_columns: list[bool] | None = None,
        ):
        if wide_columns is None:
            wide_columns = self.current_wide_columns
        patch_idx_from_physical_qubit_idx, physical_qubits_from_patch_idx = self._get_phys_qubit_patch_mapping(wide_columns)
        patches_online = np.zeros(self.num_patches, bool)
        for p in range(self.num_patches):
            patches_online[p] = np.all(self.physical_qubit_offline_time_remaining[physical_qubits_from_patch_idx[p]] <= 0.0)
        return patches_online

    def _cycles_per_distillation(
            self,
        ) -> int | None:
        """Calculate the number of cycles required for one magic state
        distillation, based on the current state of
        self.patch_offline_time_remaining.

        Even if self.cache_cycles_per_distillation is False, this function
        remembers the result of the last call to this function, and will return
        the same result if the state of self.patch_offline_time_remaining has
        not changed. This uses barely any memory, and is useful for avoiding
        redundant calculations.
        
        Returns:
            The number of surface code stabilizer measurement cycles required
            for one magic state distillation, or None if the factory cannot
            currently produce magic states.
        """
        physical_qubits_online = (self.physical_qubit_offline_time_remaining <= 0.0)
        
        cycle_count = None
        if np.all(physical_qubits_online):
            # operating at full capacity
            cycle_count = 6
        else:
            if self.mapping_mode == 'none':
                cycle_count = None
            elif self.mapping_mode == 'simple':
                cycle_count = self._cycles_per_distillation_simple(
                    physical_qubits_online,
                )
            elif self.mapping_mode == 'remap':
                cycle_count = self._cycles_per_distillation_remap(
                    physical_qubits_online, 
                )[0]
            else:
                raise ValueError('Invalid mode.')
        return cycle_count

    def _cycles_per_distillation_simple(
            self, 
            physical_qubits_online: NDArray[np.bool_],
        ) -> int | None:
        patch_idx_from_physical_qubit_idx, physical_qubits_from_patch_idx = self._get_phys_qubit_patch_mapping([True]+[False]*(self.patch_indices.shape[1]-1))
        patches_online = np.zeros(self.num_patches, bool)
        for p in range(self.num_patches):
            patches_online[p] = np.all(physical_qubits_online[physical_qubits_from_patch_idx[p]])

        current_optimization_key = (tuple(self.current_wide_columns), tuple(patches_online))
        if current_optimization_key in self._cycles_per_distillation_cache:
            return self._cycles_per_distillation_cache[(tuple(self.current_wide_columns), tuple(patches_online))]

        array = patches_online.astype(float)
        array[array == 0] = -np.inf
        array = array[self.patch_indices]
        array[:,0] *= 100

        one_routing_space = np.ones((2,5))
        two_routing_spaces = np.ones((3,5))

        results_one_routing = scipy.signal.convolve(array, one_routing_space, mode='valid')
        results_two_routing = scipy.signal.convolve(array, two_routing_spaces, mode='valid')

        if np.max(results_two_routing) > 100:
            cycle_count = 6
        elif np.max(results_one_routing) > 100:
            cycle_count = 12
        else:
            cycle_count = None

        if self._cache_cycles_per_distillation:
            self._cycles_per_distillation_cache[current_optimization_key] = cycle_count

        return cycle_count

    def _cycles_per_distillation_remap(
            self,
            physical_qubits_online: NDArray[np.bool_],
        ) -> tuple[int | None, NDArray[np.int_] | None]:
        
        best_cycle_count = None
        best_mapping = None

        for chosen_wide_column in range(self.patch_indices.shape[1]):
            wide_columns = [False]*self.patch_indices.shape[1]
            wide_columns[chosen_wide_column] = True
            _, physical_qubits_from_patch_idx = self._get_phys_qubit_patch_mapping([True]+[False]*(self.patch_indices.shape[1]-1))

            patches_online = np.zeros(self.num_patches, bool)
            for p in range(self.num_patches):
                # this is currently the performance bottleneck...
                patches_online[p] = np.all(physical_qubits_online[physical_qubits_from_patch_idx[p]])
            online_patch_indices = np.where(patches_online)[0]
            online_wide_patches = [i for i in self.patch_indices[:,chosen_wide_column] if patches_online[i]]

            self.cache_queries += 1
            current_optimization_key = (tuple(wide_columns), tuple(patches_online))
            if current_optimization_key in self._cycles_per_distillation_cache:
                self.cache_hits += 1
                cycle_count = self._cycles_per_distillation_cache[(tuple(wide_columns), tuple(patches_online))]
                if cycle_count is not None and (best_cycle_count is None or cycle_count < best_cycle_count):
                    best_cycle_count = cycle_count
                    best_mapping = None
            else:
                for chosen_q0 in online_wide_patches:
                    for chosen_other_qubits in itertools.combinations([i for i in online_patch_indices if i != chosen_q0], 4):
                        chosen_logical_qubits = np.array([chosen_q0] + list(chosen_other_qubits), int)
                        communication_paths = [[] for _ in range(self.applied_rotations.shape[1])]

                        # make sure we can perform each rotation
                        can_perform_rotations = np.zeros(self.applied_rotations.shape[1], bool)
                        search_array = patches_online[self.patch_indices]
                        for q in chosen_logical_qubits:
                            search_array[self.patch_coords_from_idx[q][0], self.patch_coords_from_idx[q][1]] = False
                        all_pairs_paths = boolean_array_all_pairs_BFS(search_array)
                        for i,combo in enumerate(self.applied_rotations.T):
                            qubits = np.where(combo)[0]
                            qubit_coords = self.patch_coords_from_idx[chosen_logical_qubits[qubits]]
                            # routing space must connect to either top or bottom of
                            # each; we will check all combinations until we find one
                            # that works
                            # TODO: if a qubit is in a wide column, it can
                            # optionally be rotated and use left/right instead of
                            # top/bottom
                            for top_bot_assignment in itertools.product([-1, 1], repeat=len(qubits)):
                                routing_coords = qubit_coords + np.array([list(top_bot_assignment), [0]*len(qubits)]).T
                                if np.any(routing_coords < 0) or np.any(routing_coords[:,0] >= self.patch_indices.shape[0]) or np.any(routing_coords[:,1] >= self.patch_indices.shape[1]):
                                    continue
                                communication_path = set()
                                can_perform_rotations[i] = True
                                for q0 in range(len(qubits)):
                                    if tuple(routing_coords[q0]) in all_pairs_paths:
                                        paths = all_pairs_paths[tuple(routing_coords[q0])]
                                        for q1 in range(q0, len(qubits)):
                                            if tuple(routing_coords[q1]) in paths:
                                                shortest_path = np.array(paths[tuple(routing_coords[q1])])
                                                communication_path |= set(self.patch_indices[shortest_path[:,0], shortest_path[:,1]])
                                            else:
                                                can_perform_rotations[i] = False
                                                break
                                    else:
                                        can_perform_rotations[i] = False
                                        break
                                    if not can_perform_rotations[i]:
                                        break
                                if can_perform_rotations[i]:
                                    assert len(communication_path) > 0, (i, all_pairs_paths[tuple(routing_coords[0])], communication_path)
                                    communication_paths[i].append(communication_path)
                        
                        if not np.all(can_perform_rotations):
                            continue
                        else:
                            num_layers = 0
                            unscheduled_rotations = list(range(self.initial_rotations.shape[1]))
                            rotation_order = []
                            rotation_layers = []
                            did_initial_rotations = False
                            did_final_rotations = False

                            while not did_final_rotations:
                                if len(unscheduled_rotations) == 0:
                                    if did_initial_rotations:
                                        did_final_rotations = True
                                        break
                                    else:
                                        did_initial_rotations = True
                                        unscheduled_rotations = list(range(self.initial_rotations.shape[1], self.applied_rotations.shape[1]))
                                # schedule a layer of rotations
                                num_layers += 1
                                completed_rotations = []
                                # We want to minimize number of layers, so we will
                                # check every combination of rotations, attempting
                                # larger numbers of parallel rotations first.
                                for rot_combo in itertools.chain.from_iterable([itertools.combinations(unscheduled_rotations, n) for n in range(5,0,-1)]):
                                    all_rots_can_be_done = False
                                    for path_combo in itertools.product(*[communication_paths[rot] for rot in rot_combo]):
                                        # all paths must not intersect each other
                                        paths_sum = sum([len(path) for path in path_combo])
                                        paths_set = set.union(*path_combo)
                                        if paths_sum == len(paths_set):
                                            all_rots_can_be_done = True
                                            break
                                    if all_rots_can_be_done:
                                        completed_rotations = list(rot_combo)
                                        for rot in completed_rotations:
                                            rotation_order.append(self.applied_rotations.T[rot])
                                        rotation_layers.extend([num_layers]*len(completed_rotations))
                                        break
                                assert len(completed_rotations) > 0
                                unscheduled_rotations = [x for x in unscheduled_rotations if x not in completed_rotations]
                            if best_cycle_count is None or num_layers < best_cycle_count:
                                best_cycle_count = num_layers
                                best_mapping = chosen_logical_qubits
                        if best_cycle_count < 6:
                            raise Exception('Cycle time is less than 6.')
                        if best_cycle_count == 6:
                            # we know we can't do better than this
                            if self._cache_cycles_per_distillation:
                                self._cycles_per_distillation_cache[current_optimization_key] = best_cycle_count
                            return best_cycle_count, best_mapping
                if self._cache_cycles_per_distillation:
                    self._cycles_per_distillation_cache[current_optimization_key] = best_cycle_count
        return best_cycle_count, best_mapping
