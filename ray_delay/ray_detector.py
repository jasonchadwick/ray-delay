"""TODO
"""
import numpy as np
from stim_surface_code.patch import Qubit, MeasureQubit
from dataclasses import dataclass
from typing import Callable
from numpy.typing import NDArray
import scipy
import mpmath
from ray_delay.noise_model import CosmicRayParams
from ray_delay.noise_model_patch import NoiseModelPatch

class RayDetectorSpec:
    """Encodes information about ray model and detector.
    
    Attributes:
        detector_spatial_window_size: Edge length of the square region to
            consider when detecting cosmic rays, in units of device indices.
            Must be smaller than width and height of device.
        detector_temporal_window_size: The number of syndrome measurement
            rounds to consider when detecting cosmic rays.
        ray_model_type: The type of ray model to use.
        ray_radius: The radius of the ray model, in units of device indices.
        ray_max_strength: The strength at the center of the ray.
        detection_distances: The distances from the center of the ray at which
            we have computed the detection chance.
        signal_rates: The detection chance at each distance from
            the center of the ray. Must be the same length as
            detection_distances.
        ray_halflife: The time it takes for the ray to decay to half of its
            original strength, in seconds.
    """
    detector_spatial_window_size: int
    detector_temporal_window_size: int
    ray_params: CosmicRayParams
    detection_distances: NDArray[np.float_]
    signal_rates: NDArray[np.float_]
    ray_halflife: float

    def __init__(
            self,
            detector_spatial_window_size: int,
            detector_temporal_window_size: int,
            ray_params: CosmicRayParams,
            detection_distances: NDArray[np.float_],
            times_after_ray_impact: NDArray[np.float_],
            first_distillation_signal_rates: NDArray[np.float_],
            decaying_signal_rates: NDArray[np.float_],
            baseline_signal_rate: float = 0.0,
            ray_halflife: float = 30e-3,
        ):
        """Initialize the ray detector spec.

        Args:
            detector_spatial_window_size: Edge length of the square region to
                consider when detecting cosmic rays, in units of device indices.
                Must be smaller than width and height of device.
            detector_temporal_window_size: The number of syndrome measurement
                rounds to consider when detecting cosmic rays.
            ray_params: CosmicRayParams object describing the ray model.
            detection_distances: The distances from the center of the ray at
                which we have computed the detection chance.
            times_after_ray_impact: The times after ray impact at which we have
                computed the detection chance.
            first_distillation_signal_rates: For each cycle in the first
                distillation after the ray impact, the chance that a qubit at
                distance d is in a window that signals a ray. Must be of shape
                (detector_temporal_window_size, len(detection_distances)).
            decaying_signal_rates: For each time in times_after_ray_impact, the
                chance that a qubit at distance d is in a window that signals a
                ray. Must be of shape (len(times_after_ray_impact),
                len(detection_distances)).
            baseline_signal_rate: The average chance of detection for a qubit
                when no ray is present (i.e. the false positive rate).
            ray_halflife: The time it takes for the ray to decay to half of its
                original strength, in seconds.
        """
        self.detector_spatial_window_size = detector_spatial_window_size
        self.detector_temporal_window_size = detector_temporal_window_size
        self.ray_params = ray_params
        self.detection_distances = detection_distances
        self.times_after_ray_impact = times_after_ray_impact
        assert first_distillation_signal_rates.shape == (detector_temporal_window_size, len(detection_distances))
        assert decaying_signal_rates.shape == (len(times_after_ray_impact), len(detection_distances))
        self.first_distillation_signal_rates = first_distillation_signal_rates
        self.decaying_signal_rates = decaying_signal_rates
        self.baseline_signal_rate = baseline_signal_rate
        self.ray_halflife = ray_halflife

        first_distillation_signal_rates_float = first_distillation_signal_rates.astype(float)

        self.first_distillation_interpolators = [
            lambda x: np.interp(
                x,
                detection_distances,
                first_distillation_signal_rates_float[cycle],
            )
            for cycle in range(detector_temporal_window_size)
        ]
        self.decaying_interpolator = scipy.interpolate.RegularGridInterpolator(
            (times_after_ray_impact, detection_distances), 
            decaying_signal_rates.astype(float), 
            bounds_error=False, 
            fill_value=baseline_signal_rate,
        )
    
    def detection_chance_function(
            self, 
            distance_from_center: float | NDArray[np.float_], 
            cycles_after_ray_impact: int | None = None,
            time_after_ray_impact: float = 0.0,
        ) -> float | NDArray[np.float_]:
        """A function that takes in distance(s) from center of ray amd number of
        rounds since ray impact, and returns the chance of the ray being
        detected by a stabilizer(s) at the given distance(s).

        Only one of cycles_after_ray_impact and time_after_ray_impact should be
        specified. If cycles_after_ray_impact is not -1, we assume that the ray
        has just occurred and the window does not 
        
        Args:
            distance_from_center: Distance(s) from center of ray, in units of
                device indices.
            cycles_after_ray_impact: Number of rounds since ray impact. Used to
                calculate detection chance soon after ray impact. If -1, assume
                that ray impact occurred before the start of the window. Only
                meaningful if between 0 and self.detector_temporal_window_size,
                although it will work outside of those bounds.
            time_after_ray_impact: Time since ray impact, in seconds. Used to
                calculate residual detection chance while ray slowly decays
                away. If cycles_after_ray_impact is not None, this value is
                ignored.
        """
        if cycles_after_ray_impact is None:
            return self.decaying_interpolator((time_after_ray_impact, distance_from_center))
        else:
            temporal_idx = min(cycles_after_ray_impact, self.detector_temporal_window_size-1)
            return self.first_distillation_interpolators[temporal_idx](distance_from_center)

class RayImpactSimulator:
    """Calculates statistics for ray detection performance (as opposed to
    simulating randomly-sampled ray detections, which is done by RayDetector).
    """
    def __init__(
            self, 
            patch: NoiseModelPatch,
            spatial_window_size: int,
            only_full_windows: bool = False,
        ):
        """Initialize the ray detection simulator.

        Args:
            ray_detector_spec: The ray detector spec to use.
            device: The layout of the device, in the form of a 2D list of Qubit
                objects.    
            spatial_window_size: Number of qubits in each dimension of the
                square spatial window.
            only_full_windows: If True, only return windows that are completely
                filled with qubits. If False, return all windows that are at
                least partially filled with qubits (such as windows on the edges
                of a device patch).
        """
        self._patch = patch
        self._spatial_window_size = spatial_window_size
        self._windows = self._initialize_windows(
            spatial_window_size, 
            only_full_windows
        )
        
    def generate_detector_spec(
            self,
            window_fpr: float | mpmath.mpf,
            cycles_per_distillation: int,
            temporal_window_size: int | None = None,
            max_simulate_time: float = 30e-3,
            decay_nsteps: int = 10,
            save_detector_spec: bool = False,
        ):
        """Generate a RayDetectorSpec object for the given patch by simulating a
        cosmic ray impact at the center of the patch and calculating the chances
        that each qubit lies within a triggering detection window.

        Args:
            window_fpr: The desired false positive rate for every
                window. If type is mpmath.mpf, this type will be used throughout
                the computation. This will be more accurate for very low
                probabilities, but will be slower to calculate.
            cycles_per_distillation: The number of syndrome measurement rounds
                per distillation cycle.
            temporal_window_size: The number of syndrome measurement rounds to
                consider when detecting cosmic rays. If None, use
                cycles_per_distillation.
            max_simulate_time: The maximum time to simulate the tail decay of
                the ray.
            decay_nsteps: The number of evenly-spaced steps to use when
                simulating the decay of the ray.
            save_detector_spec: If True, save RayDetectorSpec object to a file
                via pickle.
        """
        # patch should be sufficiently large so that ray does not reach boundaries
        assert len(self._patch.patch.device) > 2*self._patch.noise_params.cosmic_ray_params.max_radius + 2*self._spatial_window_size
        assert len(self._patch.patch.device[0]) > 2*self._patch.noise_params.cosmic_ray_params.max_radius + 2*self._spatial_window_size

        # simulate ray impact and decay
        self._patch.reset()
        baseline_fractions = np.mean(self._patch.patch.count_detection_events(1e6, return_full_data=True)[0], axis=0)

        center_coords = (len(self._patch.patch.device) // 2, 
                         len(self._patch.patch.device[0]) // 2)

        # self._patch.force_cosmic_ray_by_coords(center_coords)
        # ray_fractions = []
        # timestep = max_simulate_time / (decay_nsteps-1)
        # for i in range(decay_nsteps):
        #     ray_fractions.append(np.mean(self._patch.patch.count_detection_events(1e6, return_full_data=True)[0], axis=0))
        #     self._patch.step(timestep)
        # self._patch.reset()
        ray_fractions = []
        for time in np.linspace(0, max_simulate_time, decay_nsteps):
            self._patch.reset()
            self._patch.force_cosmic_ray_by_coords(center_coords)
            self._patch.step(time)
            ray_fractions.append(np.mean(self._patch.patch.count_detection_events(1e6, return_full_data=True)[0], axis=0))

        syndrome_qubits = self._patch.patch.get_syndrome_qubits()
        baseline_fractions_labeled = {q.idx: baseline_fractions[i] for i,q in enumerate(syndrome_qubits)}
        ray_fractions_labeled = [{q.idx:r[i] for i,q in enumerate(syndrome_qubits)} for r in ray_fractions]
        
        # calculate distance from center for each qubit
        ancilla_distances = np.zeros(len(self._patch.patch.ancilla), dtype=float)
        for i,qubit in enumerate(self._patch.patch.ancilla):
            coords = qubit.coords
            ancilla_distances[i] = np.sqrt((coords[0]-center_coords[0])**2 + (coords[1]-center_coords[1])**2)

        unique_distances = []
        qubits_per_distance = []
        for i,dst in enumerate(ancilla_distances):
            qubit = self._patch.patch.ancilla[i].idx
            dst_rounded = np.round(dst, 2)
            if dst_rounded not in unique_distances:
                unique_distances.append(dst_rounded)
                qubits_per_distance.append([qubit])
            else:
                qubits_per_distance[unique_distances.index(dst_rounded)].append(qubit)
        qubits_per_distance = np.array(qubits_per_distance, dtype=object)[np.argsort(unique_distances)]
        unique_distances = np.sort(unique_distances)

        # calculate detection chances within first distillation, for each cycle.
        # Within this time frame, the temporal detection windows will not fully
        # contain the ray.
        # TODO: lots of code re-use between this section and the next section;
        # could probably be refactored into one function that takes different
        # sets of detector means.
        # TODO: would it be better to work in log space for floating point
        # accuracy? 
        if temporal_window_size is None:
            temporal_window_size = cycles_per_distillation
        dtype = type(window_fpr)
        window_signal_rates = np.zeros((cycles_per_distillation, len(self._windows)), dtype=dtype)
        for cycle in range(cycles_per_distillation):
            for k,window in enumerate(self._windows):
                baseline_mean = np.mean([baseline_fractions_labeled[q] for q in window]) # mean per qubit per round
                
                # take first ray data point (error values at moment of ray impact)
                ray_mean = np.mean([ray_fractions_labeled[0][q] for q in window])

                # threshold of detections where we say it is outside of the baseline regime
                detection_threshold = scipy.stats.binom.ppf(float(1-window_fpr), len(window)*temporal_window_size, baseline_mean)
            
                if cycle <= temporal_window_size:
                    # some part of window is at baseline rate (before ray has hit)
                    windowed_syndrome_rate = baseline_mean*(1-cycle/temporal_window_size) + ray_mean*(cycle/temporal_window_size)
                else:
                    windowed_syndrome_rate = ray_mean

                detection_prob = 1-scipy.stats.binom.cdf(detection_threshold, len(window)*temporal_window_size, windowed_syndrome_rate)
                window_signal_rates[cycle,k] = dtype(detection_prob)

        qubit_no_signal_rates = np.ones((cycles_per_distillation, len(self._patch.patch.all_qubits)), dtype=dtype)
        for cycle in range(cycles_per_distillation):
            for k,window in enumerate(self._windows):
                signal_rate = window_signal_rates[cycle,k]
                for q in window:
                    qubit_no_signal_rates[cycle,q] *= (1-signal_rate)

        # FPR of a non-ray qubit = average signal rate of all qubits before ray
        # has entered temporal detection windows.
        false_positive_rate = np.mean(1-qubit_no_signal_rates[0])

        first_distillation_signal_rate_by_distance = np.zeros((cycles_per_distillation, len(unique_distances)), dtype=dtype)
        for k,dst in enumerate(unique_distances):
            for q in qubits_per_distance[k]:
                first_distillation_signal_rate_by_distance[:,k] += (1-qubit_no_signal_rates[:,q])
            first_distillation_signal_rate_by_distance[:,k] /= len(qubits_per_distance[k])

            # lower bound by average false positive rate
            for cycle in range(cycles_per_distillation):
                first_distillation_signal_rate_by_distance[cycle,k] = np.maximum(first_distillation_signal_rate_by_distance[cycle,k], false_positive_rate)
        
        # calculate detection chances within the tail decay of the ray.
        decay_signal_rates = np.zeros((decay_nsteps, len(self._windows)), dtype=dtype)
        for j in range(decay_nsteps):
            for k,window in enumerate(self._windows):
                windowed_syndrome_rate = np.mean([ray_fractions_labeled[j][q] for q in window])

                # threshold of detections where we say it is outside of the baseline regime
                detection_threshold = scipy.stats.binom.ppf(float(1-window_fpr), len(window)*temporal_window_size, baseline_mean)

                detection_prob = 1-scipy.stats.binom.cdf(detection_threshold, len(window)*temporal_window_size, windowed_syndrome_rate)
                decay_signal_rates[j,k] = dtype(detection_prob)

        qubit_no_signal_rates = np.ones((decay_nsteps, len(self._patch.patch.all_qubits)), dtype=dtype)
        for j in range(decay_nsteps):
            for k,window in enumerate(self._windows):
                signal_rate = decay_signal_rates[j,k]
                for q in window:
                    qubit_no_signal_rates[j,q] *= (1-signal_rate)

        decaying_signal_rate_by_distance = np.zeros((decay_nsteps, len(unique_distances)), dtype=dtype)
        for k,dst in enumerate(unique_distances):
            for q in qubits_per_distance[k]:
                decaying_signal_rate_by_distance[:,k] += (1-qubit_no_signal_rates[:,q])
            decaying_signal_rate_by_distance[:,k] /= len(qubits_per_distance[k])

            # lower bound by average false positive rate
            for j in range(decay_nsteps):
                decaying_signal_rate_by_distance[j,k] = np.maximum(decaying_signal_rate_by_distance[j,k], false_positive_rate)

        ray_detector_spec = RayDetectorSpec(
            detector_spatial_window_size = self._spatial_window_size,
            detector_temporal_window_size = temporal_window_size,
            ray_params = self._patch.noise_params.cosmic_ray_params,
            detection_distances = unique_distances,
            times_after_ray_impact = np.linspace(0, max_simulate_time, decay_nsteps),
            first_distillation_signal_rates = first_distillation_signal_rate_by_distance,
            decaying_signal_rates = decaying_signal_rate_by_distance,
            baseline_signal_rate = float(false_positive_rate),
        )

        if save_detector_spec:
            with open(f'ray_detector_spec_threshold_1e{int(np.round(np.log10(float(1-threshold))))}.pkl', 'wb') as f:
                pickle.dump(ray_detector_spec, f)

        return ray_detector_spec

    def _initialize_windows(
            self,
            spatial_window_size: int,
            only_full_windows: bool = False,
        ) -> list[list[int]]:
        """Initialize spatial windows that we will use to detect cosmic rays.
        
        Args:
            device: The layout of the device, in the form of a 2D list of Qubit
                objects.
            spatial_window_size: Number of qubits in each dimension of the
                square spatial window.
            only_full_windows: If True, only return windows that are completely
                filled with qubits. If False, return all windows that are at
                least partially filled with qubits (such as windows on the edges
                of a device patch).
        
        Returns:
            A list of lists of qubit indices, where each inner list contains the
            indices of the qubits in one spatial window.
        """
        assert spatial_window_size < len(self._patch.patch.device) and spatial_window_size < len(self._patch.patch.device[0])
        window_rows = (len(self._patch.patch.device) - spatial_window_size)//2 + 1
        window_cols = (len(self._patch.patch.device[0]) - spatial_window_size)//2 + 1

        min_qubit_count = spatial_window_size**2 if only_full_windows else 1

        all_windows = []
        for wr in range(window_rows):
            for wc in range(window_cols):
                window_qubits = []
                for r in range(wr, wr + spatial_window_size):
                    for c in range(wc, wc + spatial_window_size):
                        qb = self._patch.patch.device[2*r][2*c]
                        if isinstance(qb, MeasureQubit):
                            window_qubits.append(qb.idx)
                if len(window_qubits) >= min_qubit_count:
                    all_windows.append(window_qubits)
        return all_windows

class RayDetector:
    """Detects cosmic ray impacts on a surface code patch using syndrome
    measurements.
    """
    def __init__(
            self, 
            device: list[list[Qubit | None]],
            baseline_temporal_window_size: int,
            temporal_window_size: int,
            spatial_window_size: int,
            trigger_confidence: float,
            auto_clean_data_on_detection: bool = True,
        ):
        """Initialize the ray detector.

        Args:
            device: The layout of the device, in the form of a 2D list of Qubit
                objects. 
            baseline_temporal_window_size: The number of syndrome measurement
                rounds to consider when calculating baseline syndrome rates.
            temporal_window_size: The number of syndrome measurement rounds to
                consider when detecting cosmic rays.
            spatial_window_size: Edge length of the square region to consider
                when detecting cosmic rays, in units of device indices. Must be
                smaller than width and height of device.
            trigger_confidence: The confidence threshold for detecting a cosmic
                ray, based on the binomial distribution PPF. Must be between 0
                and 1. A higher value decreases false positive chance but
                increases detection cycle.
            auto_clean_data_on_detection: If True, remove last window_size
                syndrome measurement rounds upon detection of suspected cosmic
                rays. This is useful for preventing false positives for future
                rounds of prediction.
        """
        self.device = device
        self.baseline_temporal_window_size = baseline_temporal_window_size
        self.temporal_window_size = temporal_window_size
        self.measurements_to_store = 2*self.temporal_window_size + self.baseline_temporal_window_size
        self.spatial_window_size = spatial_window_size
        self.trigger_confidence = trigger_confidence
        self.auto_clean_data_on_detection = auto_clean_data_on_detection

        self._windows = self._initialize_windows()
        self._observed_syndrome_data: dict[int, list[bool]] = {
            qubit.idx: [] 
            for device_row in device for qubit in device_row
            if isinstance(qubit, MeasureQubit)
        }
        self._current_data_size = 0

        self._baseline_rates = [np.nan for _ in range(len(self._windows))]
        self._observed_rates = [np.nan for _ in range(len(self._windows))]

    def update_and_predict(self, syndrome_data: dict[int, bool]) -> list[int]:
        """Update the ray detector and return a prediction of qubits that are
        currently affected by a cosmic ray.

        Modifies the values stored in self.window_baseline_rates and
        self.window_observed_rates. A ray can only be detected after we have
        seen at least self.temporal_window_size rounds of syndrome measurement.
        These first few rounds are assumed to be error-free. If
        self.auto_clean_data_on_detection is True, we will call
        self.clean_recent_data() upon detection of a cosmic ray.

        Args:
            syndrome_data: The syndrome data to update the ray detector with.
                Keys are qubit indices of ancilla qubits and values are 0 if the
                detector did not fire and 1 if it did.

        Returns:
            A list of qubit indices that are currently affected by a cosmic ray
            (typically empty, unless we detect a ray on that round).
        """
        # add new syndrome data
        for qubit_idx, data in syndrome_data.items():
            self._observed_syndrome_data[qubit_idx].append(data)
        
        # only keep required number of syndrome measurements
        if self._current_data_size > self.measurements_to_store:
            self._observed_syndrome_data = {
                q: measurements[-self.measurements_to_store:]
                for q,measurements in self._observed_syndrome_data.items()
            }

        # predict cosmic ray
        ray_qubits = []
        if self._current_data_size > self.baseline_temporal_window_size + self.temporal_window_size:
            for i,window in enumerate(self._windows):
                start = -(self.baseline_temporal_window_size + self.temporal_window_size)
                end = -self.temporal_window_size
                self._baseline_rates[i] = float(np.mean(
                    [self._observed_syndrome_data[q][start:end] for q in window]
                ))
                self._observed_rates[i] = float(np.mean(
                    [self._observed_syndrome_data[q][end:] for q in window]
                ))
                if self._observed_rates[i] > self._baseline_rates[i]:
                    ray_qubits.extend(window)
                
        if self.auto_clean_data_on_detection and len(ray_qubits) > 0:
            self.clean_recent_data()
        return ray_qubits

    def clean_recent_data(self):
        """Remove the last window_size syndrome measurement rounds from saved
        syndromes to filter out a cosmic ray event.

        Must have observed at least self.temporal_window_size rounds of syndrome
        measurements.
        """
        assert self._current_data_size >= self.temporal_window_size

        self._observed_syndrome_data = {
            q: self._observed_syndrome_data[q][:-self.temporal_window_size]
            for q in self._observed_syndrome_data
        }
        self._current_data_size -= self.temporal_window_size

    def _initialize_windows(
            self,
            only_full_windows: bool = False,
        ) -> list[list[int]]:
        """Initialize spatial windows that we will use to detect cosmic rays.
        
        Args:
            device: The layout of the device, in the form of a 2D list of Qubit
                objects.
        """
        assert self.spatial_window_size < len(self.device) and self.spatial_window_size < len(self.device[0])
        window_rows = (len(self.device) - self.spatial_window_size)//2 + 1
        window_cols = (len(self.device[0]) - self.spatial_window_size)//2 + 1

        min_qubit_count = self.spatial_window_size**2 if only_full_windows else 1

        all_windows = []
        for wr in range(window_rows):
            for wc in range(window_cols):
                window_qubits = []
                for r in range(wr, wr + self.spatial_window_size):
                    for c in range(wc, wc + self.spatial_window_size):
                        qb = self.device[2*r][2*c]
                        if isinstance(qb, MeasureQubit):
                            window_qubits.append(qb.idx)
                if len(window_qubits) >= min_qubit_count:
                    all_windows.append(window_qubits)
        return all_windows