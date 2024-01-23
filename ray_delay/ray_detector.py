"""TODO
"""
import numpy as np
from stim_surface_code.patch import Qubit, MeasureQubit
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from numpy.typing import NDArray

class RayModelType(Enum):
    """TODO"""
    CONSTANT = 0
    LINEAR_ERR = 1
    LINEAR_T1 = 2

@dataclass
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
        detection_chance_function: A function that takes in distance(s) from
            center of ray amd number of rounds since ray impact, and returns the
            chance of the ray being detected by a stabilizer(s) at the given
            distance(s).
    """
    detector_spatial_window_size: int
    detector_temporal_window_size: int
    ray_model_type: RayModelType
    ray_radius: float
    ray_max_strength: float
    detection_chance_function: Callable[[NDArray[np.float_] | float, int], NDArray[np.float_] | float]

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

        self._windows = self._initialize_windows(device)
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
            device: list[list[Qubit | None]],
        ) -> list[list[int]]:
        """Initialize spatial windows that we will use to detect cosmic rays.
        
        Args:
            device: The layout of the device, in the form of a 2D list of Qubit
                objects.
        """
        assert self.spatial_window_size < len(device) and self.spatial_window_size < len(device[0])
        window_rows = len(device) - self.spatial_window_size
        window_cols = len(device[0]) - self.spatial_window_size

        all_windows = []
        for wr in range(window_rows):
            for wc in range(window_cols):
                window_qubits = []
                for r in range(wr, wr + self.spatial_window_size + 1):
                    for c in range(wc, wc + self.spatial_window_size + 1):
                        qb = device[r][c]
                        if isinstance(qb, MeasureQubit):
                            window_qubits.append(qb.idx)
                if len(window_qubits) > 0:
                    all_windows.append(window_qubits)
        return all_windows