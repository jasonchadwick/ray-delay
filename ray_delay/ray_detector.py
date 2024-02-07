"""TODO
"""
import numpy as np
from stim_surface_code.patch import Qubit, MeasureQubit
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from numpy.typing import NDArray
import scipy

class RayModelType(Enum):
    """TODO"""
    CONSTANT = 0
    LINEAR_ERR = 1
    LINEAR_T1 = 2

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
    ray_model_type: RayModelType
    ray_radius: float
    ray_max_strength: float
    detection_distances: NDArray[np.float_]
    signal_rates: NDArray[np.float_]
    ray_halflife: float

    def __init__(
            self,
            detector_spatial_window_size: int,
            detector_temporal_window_size: int,
            ray_model_type: RayModelType,
            ray_radius: float,
            ray_max_strength: float,
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
            ray_model_type: The type of ray model to use.
            ray_radius: The radius of the ray model, in units of device indices.
            ray_max_strength: The strength at the center of the ray.
            detection_distances: The distances from the center of the ray at
                which we have computed the detection chance.
            times_after_ray_impact: The times after ray impact at which we have
                computed the detection chance.
            
            ray_halflife: The time it takes for the ray to decay to half of its
                original strength, in seconds.
        """
        self.detector_spatial_window_size = detector_spatial_window_size
        self.detector_temporal_window_size = detector_temporal_window_size
        self.ray_model_type = ray_model_type
        self.ray_radius = ray_radius
        self.ray_max_strength = ray_max_strength
        self.detection_distances = detection_distances
        self.times_after_ray_impact = times_after_ray_impact
        assert first_distillation_signal_rates.shape == (detector_temporal_window_size, len(detection_distances))
        assert decaying_signal_rates.shape == (len(times_after_ray_impact), len(detection_distances))
        self.first_distillation_signal_rates = first_distillation_signal_rates
        self.decaying_signal_rates = decaying_signal_rates
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
            ray_detector_spec: RayDetectorSpec,
            device: list[list[Qubit | None]],
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
        self.ray_detector_spec = ray_detector_spec
        self._device = device
        self._windows = self._initialize_windows(
            spatial_window_size, 
            only_full_windows
        )
        
    def generate_detector_model(
            self,

        ):
        raise NotImplementedError

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
        assert spatial_window_size < len(self._device) and spatial_window_size < len(self._device[0])
        window_rows = (len(self._device) - spatial_window_size)//2 + 1
        window_cols = (len(self._device[0]) - spatial_window_size)//2 + 1

        min_qubit_count = spatial_window_size**2 if only_full_windows else 1

        all_windows = []
        for wr in range(window_rows):
            for wc in range(window_cols):
                window_qubits = []
                for r in range(wr, wr + spatial_window_size):
                    for c in range(wc, wc + spatial_window_size):
                        qb = self._device[2*r][2*c]
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
                increases detection latency.
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