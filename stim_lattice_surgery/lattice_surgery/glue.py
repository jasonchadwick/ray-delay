from stim_surface_code.patch import SurfaceCodePatch, MeasureQubit, DataQubit
from stim_lattice_surgery.lattice_surgery.zxxz_patch import ZXXZPatch
import numpy as np
import stim
from enum import Enum

class BoundaryType(Enum):
    LEFT = 'X', 0
    RIGHT = 'X', 1
    TOP = 'Z', 2
    BOTTOM = 'Z', 3

    def __init__(self, basis: str, _: int):
        self.basis = basis
        if len(self.__class__):
            all = list(self.__class__)
            if all[-1].basis == self.basis:
                self.compatible = all[-1]
                all[-1].compatible = self
            


class GluePatch(ZXXZPatch):
    '''
    A single line of stabilizers between two surface code patches involved in a lattice surgery operation.
    This allows all surface code patches to have the same orientation.
    '''
    def __init__(
            self, 
            patch1: SurfaceCodePatch, 
            patch1_boundary: BoundaryType, 
            patch2: SurfaceCodePatch, 
            patch2_boundary: BoundaryType,
            **kwargs
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
            dx = 0
        else:
            dz = 0
            dx = self.patch1.dx
        assert patch1.dm == patch2.dm
        assert patch1.gate1_time == patch2.gate1_time
        assert patch1.gate2_time == patch2.gate2_time
        assert patch1.meas_time == patch2.meas_time
        assert patch1.apply_idle_during_gates == patch2.apply_idle_during_gates
        assert patch1.save_stim_circuit == patch2.save_stim_circuit

        self.patch1_boundary_stabilizers = self._get_boundary_stabilizers(patch1, patch1_boundary)
        self.patch2_boundary_stabilizers = self._get_boundary_stabilizers(patch2, patch2_boundary)

        super().__init__(dx, dz, patch1.dm, 
                         patch1.gate1_time, patch1.gate2_time,
                         patch1.meas_time, patch1.apply_idle_during_gates,
                         patch1.save_stim_circuit,
                         **kwargs)
        
        self.boundary_data = {measure_q: measure_q.data_qubits.copy() for measure_q in self.patch1_boundary_stabilizers + self.patch2_boundary_stabilizers}

        self.init_reset_h_z = []
        self.init_reset_h_x = []
        for meas in self.z_ancilla:
            for i, data in enumerate(meas.data_qubits):
                if data in self.data and data not in self.init_reset_h_z and (i == 1 or i == 2):
                    self.init_reset_h_z += [data]
        for meas in self.x_ancilla:
            for i, data in enumerate(meas.data_qubits):
                if data in self.data and data not in self.init_reset_h_x and (i == 1 or i == 2):
                    self.init_reset_h_x += [data]
        if len(self.init_reset_h_z) == 0:
            for data in self.data:
                if data not in self.init_reset_h_x:
                    self.init_reset_h_z += [data]
        elif len(self.init_reset_h_x) == 0:
            for data in self.data:
                if data not in self.init_reset_h_z:
                    self.init_reset_h_x += [data]
        self.merged = False
    
    def _get_anti_basis_data(self) -> DataQubit:
        '''
        Data qubit in merged observable of opposite basis whose result is assigned to patch2's observable
        '''
        for data in self.data:
            if self.patch1_boundary.basis == 'X':
                if data.coords[0] == list(self.patch1.logical_z_qubits)[0].coords[0]:
                    return data
            elif self.patch1_boundary.basis == 'Z':
                if data.coords[1] == list(self.patch1.logical_x_qubits)[0].coords[1]:
                    return data

    def _get_merged_product(self) -> list[MeasureQubit]:
        '''
        Returns list of ancilla who's product is the merged observable of interest
        '''
        def coord_range(patch: SurfaceCodePatch, patch_boundary: BoundaryType) -> tuple[tuple[int, int], tuple[int, int]]:
            x_col = list(patch.logical_x_qubits)[0].coords[1]
            z_row = list(patch.logical_z_qubits)[0].coords[0]
            if patch_boundary == BoundaryType.LEFT:
                patch_range = ((0, 2 * patch.dz + 1), (0, x_col))
            elif patch_boundary == BoundaryType.RIGHT:
                patch_range = ((0, 2 * patch.dz + 1), (x_col + 1, 2 * patch.dx + 1))
            elif patch_boundary == BoundaryType.TOP:
                patch_range = ((0, z_row), (0, 2 * patch.dx + 1))
            elif patch_boundary == BoundaryType.BOTTOM:
                patch_range = ((z_row + 1, 2 * patch.dz + 1), (0, 2 * patch.dx + 1))
            return patch_range
        
        def ancilla_in_range(patch: SurfaceCodePatch, range: tuple[tuple[int, int], tuple[int, int]], basis: str) -> list[MeasureQubit]:
            all_ancilla = (
                patch.x_ancilla if basis == 'X' else patch.z_ancilla
            )
            ancilla = []
            for measure_q in all_ancilla:
                if measure_q.coords[0] >= range[0][0] and measure_q.coords[0] < range[0][1] \
                    and measure_q.coords[1] >= range[1][0] and measure_q.coords[1] < range[1][1]:

                    ancilla.append(measure_q)
            return ancilla

        patch1_range = coord_range(self.patch1, self.patch1_boundary)
        patch2_range = coord_range(self.patch2, self.patch2_boundary)
        
        patch1_ancilla = ancilla_in_range(self.patch1, patch1_range, self.patch1_boundary.basis)
        patch2_ancilla = ancilla_in_range(self.patch2, patch2_range, self.patch2_boundary.basis)

        return patch1_ancilla + patch2_ancilla + (
            self.x_ancilla if self.patch1_boundary.basis == 'X' else 
            self.z_ancilla
        )

    def _get_boundary_stabilizers(self, patch: SurfaceCodePatch, boundary: BoundaryType) -> list[MeasureQubit]:
        if boundary == BoundaryType.TOP:
            return [qbit for qbit in patch.device[0] if qbit]
        elif boundary == BoundaryType.BOTTOM:
            return [qbit for qbit in patch.device[-1] if qbit]
        elif boundary == BoundaryType.LEFT:
            return [row[0] for row in patch.device if row[0]]
        elif boundary == BoundaryType.RIGHT:
            return [row[-1] for row in patch.device if row[-1]]
    
    def place_data(
            self,
            id_offset: int
        ) -> list[DataQubit]:
        '''
        Create the device object that will hold all physical qubits, and
        place data qubits within it.
        
        Returns:
            list of DataQubit objects.
        '''
        if self.dx == 0:
            data: list[DataQubit] = [
                DataQubit((self.dz*row + col) + id_offset, (2*row, 2*col+1)) 
                for col in range(self.dz) for row in range(1)]
        elif self.dz == 0:
            data: list[DataQubit] = [
                DataQubit((row + col) + id_offset, (2*row+1, 2*col)) 
                for col in range(1) for row in range(self.dx)]
        for data_qubit in data:
            self.device[data_qubit.coords[0]][data_qubit.coords[1]] = data_qubit
        return data
    
    def _find_data(self, coords: tuple[int, int], patch: SurfaceCodePatch) -> DataQubit:
        row = coords[0]
        col = coords[1]
        return (
            patch.device[row][col] if row >= 0 and row < len(patch.device) and col >= 0 and col < len(patch.device[0]) else
            None
        )

    def place_ancilla(self, id_offset: int) -> None:
        """Place ancilla (non-data) qubits in the patch. Must be run *after*
        place_data.
        """

        # number of qubits already placed (= index of next qubit)
        q_count = len(self.data) + id_offset
        basis = self.patch1_boundary.basis
        self.x_ancilla = []
        self.z_ancilla = []

        if basis == 'Z':
            top_patch = self.patch1 if self.patch1_boundary == BoundaryType.BOTTOM else self.patch2
            bottom_patch = self.patch2 if self.patch2_boundary == BoundaryType.TOP else self.patch1
            for col in range(self.dz + 1):
                if col % 2 == 1:
                    data_list = [
                        self._find_data((len(top_patch.device) - 2, 2 * col - 1), top_patch),
                        self._find_data((len(top_patch.device) - 2, 2 * col + 1), top_patch),
                        self._find_data((0, 2 * col - 1), self),
                        self._find_data((0, 2 * col + 1), self)
                    ]
                    z_ancilla = MeasureQubit(q_count, (2 * top_patch.dx, 2 * col), data_list, 'Z')
                    top_patch.device[2 * top_patch.dx][2 * col] = z_ancilla
                    self.z_ancilla.append(z_ancilla)
                    q_count += 1
                else:
                    data_list = [
                        self._find_data((0, 2 * col - 1), self),
                        self._find_data((0, 2 * col + 1), self),
                        self._find_data((1, 2 * col - 1), bottom_patch),
                        self._find_data((1, 2 * col + 1), bottom_patch)
                    ]
                    z_ancilla = MeasureQubit(q_count, (0, 2 * col), data_list, 'Z')
                    bottom_patch.device[0][2 * col] = z_ancilla
                    self.z_ancilla.append(z_ancilla)
                    q_count += 1
        elif basis == 'X':
            left_patch = self.patch1 if self.patch1_boundary == BoundaryType.RIGHT else self.patch2
            right_patch = self.patch2 if self.patch2_boundary == BoundaryType.LEFT else self.patch1
            for row in range(self.dx + 1):
                if row % 2 == 0:
                    data_list = [
                        self._find_data((2 * row - 1, len(left_patch.device[0]) - 2), left_patch),
                        self._find_data((2 * row + 1, len(left_patch.device[0]) - 2), left_patch),
                        self._find_data((2 * row - 1, 0), self),
                        self._find_data((2 * row + 1, 0), self)
                    ]
                    x_ancilla = MeasureQubit(q_count, (2 * row, 2 * left_patch.dz), data_list, 'X')
                    left_patch.device[2 * row][2 * left_patch.dz] = x_ancilla
                    self.x_ancilla.append(x_ancilla)
                    q_count += 1
                else:
                    data_list = [
                        self._find_data((2 * row - 1, 0), self),
                        self._find_data((2 * row + 1, 0), self),
                        self._find_data((2 * row - 1, 1), right_patch),
                        self._find_data((2 * row + 1, 1), right_patch)
                    ]
                    x_ancilla = MeasureQubit(q_count, (2 * row, 0), data_list, 'X')
                    right_patch.device[2 * row][0] = x_ancilla
                    self.x_ancilla.append(x_ancilla)
                    q_count += 1

    def merge(self) -> None:
        '''
        Turn weight 2 boundary stabilizers into merged weight 4 stabilizers
        '''
        offsets = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)] if self.patch1_boundary.basis == 'X' else [(-1, -1), (+1, -1), (-1, +1), (+1, +1)]
        for measure_q in self.patch1_boundary_stabilizers + self.patch2_boundary_stabilizers:
            for i, data_q in enumerate(measure_q.data_qubits):
                if not data_q:
                    glue_coord = (
                        (0, measure_q.coords[1] + offsets[i][1]) if self.patch1_boundary.basis == 'Z' else
                        (measure_q.coords[0] + offsets[i][0], 0)
                    )
                    measure_q.data_qubits[i] = self._find_data(glue_coord, self)
        self.merged = True
        
    
    def split(self) -> None:
        '''
        Updates stabilizers to reflect original, split boundary between patch1 and patch2
        '''
        for measure_q in self.patch1_boundary_stabilizers + self.patch2_boundary_stabilizers:
            measure_q.data_qubits = self.boundary_data[measure_q].copy()
        
        self.merged = False

    def get_stim(self, dm: int = None) -> stim.Circuit:
        '''
        Returns a stim circuit of dm rounds of a syndrome circuit with
        the current state of patch1, patch2, and glue stabilizers
        '''
        if not dm:
            dm = self.dm
        raise NotImplementedError


