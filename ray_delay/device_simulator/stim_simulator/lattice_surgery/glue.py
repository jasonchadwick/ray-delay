from ray_delay.device_simulator.stim_simulator.patch import SurfaceCodePatch, MeasureQubit, DataQubit
import numpy as np
import stim
from enum import Enum

class BoundaryType(Enum):
    LEFT = 'X'
    RIGHT = 'X'
    TOP = 'Z'
    BOTTOM = 'Z'

    def __init__(self, basis: str):
        self.basis = basis
    
    @property
    def compatible(self):
        compatabilities = {
            BoundaryType.LEFT: BoundaryType.RIGHT,
            BoundaryType.RIGHT: BoundaryType.LEFT,
            BoundaryType.TOP: BoundaryType.BOTTOM,
            BoundaryType.BOTTOM: BoundaryType.TOP
        }
        return compatabilities[self]


class GluePatch(SurfaceCodePatch):
    '''
    A single line of stabilizers between two surface code patches involved in a lattice surgery operation.
    This allows all surface code patches to have the same orientation.
    '''
    def __init__(
            self, 
            patch1: SurfaceCodePatch, 
            patch1_boundary: BoundaryType, 
            patch2: SurfaceCodePatch, 
            patch2_boundary: BoundaryType
        ) -> None:
        '''
        Args:
            patch1: First surface code patch
            patch1_boundary: The boundary being merged in patch1
            patch2: Second surface code patch
            patch2_boundary: The boundary being merged in patch2

        Asserts:
            Boundaries are compatible
            Merged attributes are identical (i.e. code distance, gate time, measurement time, etc.)
        '''
        self.patch1 = patch1
        self.patch1_boundary = patch1_boundary
        self.patch2 = patch2
        self.patch2_boundary = patch2_boundary
        assert self.patch1_boundary.compatible == self.patch2_boundary
        if self.patch1_boundary.basis == 'Z':
            dz = self.patch1.dz
            dx = 1
        else:
            dz = 1
            dx = self.patch1.dx
        assert patch1.dm == patch2.dm
        assert patch1.gate1_time == patch2.gate1_time
        assert patch1.gate2_time == patch2.gate2_time
        assert patch1.mr_time == patch2.mr_time
        assert patch1.apply_idle_during_gates == patch2.apply_idle_during_gates
        assert patch1.save_stim_circuit == patch2.save_stim_circuit

        super().__init__(dx, dz, patch1.dm, 
                         patch1.gate1_time, patch1.gate2_time,
                         patch1.mr_time, patch1.apply_idle_during_gates,
                         patch1.save_stim_circuit)

        # Align new stabilizers
        if self.patch1_boundary.basis == 'Z':
            self.device = np.fliplr(self.device) 
        else:
            self.device = np.flipud(self.device)

    def merge(self) -> None:
        '''
        Updates stabilizers to reflect a merged boundary between patch1 and patch2.
        '''
        raise NotImplementedError
    
    def split(self) -> None:
        '''
        Updates stabilizers to reflect original, split boundary between patch1 and patch2
        Updates patch1, patch2 observables with necessary measurement record indices
        '''
        raise NotImplementedError

    def get_stim(self, dm: int = None) -> stim.Circuit:
        '''
        Returns a stim circuit of dm rounds of a syndrome circuit with
        the current state of patch1, patch2, and glue stabilizers
        '''
        if not dm:
            dm = self.dm
        raise NotImplementedError


