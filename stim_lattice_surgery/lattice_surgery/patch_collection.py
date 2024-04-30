from stim_lattice_surgery.lattice_surgery.glue import GluePatch, BoundaryType
from stim_lattice_surgery.lattice_surgery.zxxz_patch import ZXXZPatch
from stim_surface_code.patch import SurfaceCodePatch, DataQubit, Qubit
from stim_surface_code.memory import MemoryPatch

import numpy as np
import stim

class PatchCollection(MemoryPatch):
    '''
    A collection of SurfaceCodePatches operating simultaneously
    '''
    
    def __init__(
            self,
            patches: list[SurfaceCodePatch],
            glue_patches: list[GluePatch],
            gate1_time: float = 25e-9, # params from Google scaling logical qubit paper
            gate2_time: float = 34e-9, 
            meas_time: float = 500e-9, 
            reset_time: float = 250e-9, # Multi-level reset from McEwen et al. (2021)
            apply_idle_during_gates: bool = True,
            save_stim_circuit: bool = False,
            **kwargs
        ) -> None:
        '''
        Collect all qubits from many individual patches into a single patch object
        Args:
            patches: Collection of SurfaceCodePatches
        
        '''

        self.glue_patches = glue_patches
        self.patches = patches
        self.patch_tree = {
                            patch: {
                                BoundaryType.BOTTOM: None,
                                BoundaryType.LEFT: None,
                                BoundaryType.RIGHT: None,
                                BoundaryType.TOP: None,
                            }
                            for patch in self.patches
                          }
        for glue in self.glue_patches:
            self.patch_tree[glue.patch1][glue.patch1_boundary] = (glue, glue.patch2)
            self.patch_tree[glue.patch2][glue.patch2_boundary] = (glue, glue.patch1)
            glue.product_ancilla = glue._get_merged_product()
        
        super().__init__(0, 0, 0, gate1_time, gate2_time, meas_time, reset_time, apply_idle_during_gates, save_stim_circuit, **kwargs)

        self.init_reset_h_z = []
        self.init_reset_h_x = self.data

        for patch in self.patches + self.glue_patches:
            for error_key, error_dict in patch.error_vals.items():
                for q, val in error_dict.items():
                    self.error_vals[error_key][q] = val
           
    def place_qubits(self) -> None:
        '''
        Place qubits in collective device
        '''
        patch_locations = {}
        curr_location = (0,0) # row, col
        def place_patch(patch):
            nonlocal curr_location
            patch_locations[patch] = curr_location
            for boundary, glue_tuple in self.patch_tree[patch].items():
                if not glue_tuple or glue_tuple[1] in patch_locations:
                    continue
                glue, neighbor_patch = glue_tuple
                if boundary == BoundaryType.BOTTOM:
                    patch_locations[glue] = (curr_location[0] + len(patch.device), curr_location[1])
                    curr_location = (curr_location[0] + len(patch.device) + len(glue.device), curr_location[1])
                elif boundary == BoundaryType.LEFT:
                    patch_locations[glue] = (curr_location[0], curr_location[1] - len(glue.device[0]))
                    curr_location = (curr_location[0], curr_location[1] - len(glue.device[0]) - len(neighbor_patch.device[0]))
                elif boundary == BoundaryType.RIGHT:
                    patch_locations[glue] = (curr_location[0], curr_location[1] + len(patch.device[0]))
                    curr_location = (curr_location[0], curr_location[1] + len(patch.device[0]) + len(glue.device[0]))
                elif boundary == BoundaryType.TOP:
                    patch_locations[glue] = (curr_location[0] - len(glue.device), curr_location[1])
                    curr_location = (curr_location[0] - len(glue.device) - len(neighbor_patch.device), curr_location[1])
                else:
                    raise Exception(f'Invalid boundary between {patch} and {neighbor_patch}')
                place_patch(neighbor_patch)
        
        place_patch(self.patches[0])

        row_range = [0, 0]
        col_range = [0, 0]
        for patch, loc in patch_locations.items():
            if loc[0] < row_range[0]:
                row_range[0] = loc[0]
            if loc[0] + len(patch.device) > row_range[1]:
                row_range[1] = loc[0] + len(patch.device)
            if loc[1] < col_range[0]:
                col_range[0] = loc[1]
            if loc[1] + len(patch.device[0]) > col_range[1]:
                col_range[1] = loc[1] + len(patch.device[0])
        self.device: list[list[Qubit | None]] = [
            [None for _ in range(col_range[1] - col_range[0])] for _ in range(row_range[1] - row_range[0])]
        for patch, location in patch_locations.items():
            row_start = location[0] - row_range[0] # ensure positive indices
            col_start = location[1] - col_range[0]
            # Copy qubits and update coords
            for row_idx in range(len(patch.device)):
                for col_idx in range(len(patch.device[0])):
                    qbit = patch.device[row_idx][col_idx]
                    if qbit:
                        qbit.coords = (row_start + row_idx, col_start + col_idx)
                        self.device[row_start + row_idx][col_start + col_idx] = qbit
    
    def place_data(self, id_offset: int) -> list[DataQubit]:
        self.place_qubits()
        data_qubits = []
        for patch in self.patches + self.glue_patches:
            data_qubits += patch.data
        return data_qubits

    def place_ancilla(self, id_offset: int) -> None:
        self.x_ancilla = []
        self.z_ancilla = []
        for patch in self.patches + self.glue_patches:
            self.x_ancilla += patch.x_ancilla
            self.z_ancilla += patch.z_ancilla

    def set_logical_operators(self) -> tuple[set[DataQubit], set[DataQubit]]:
        logical_x_qubits = []
        logical_z_qubits = []
        for patch in self.patches:
            logical_x_qubits += patch.logical_x_qubits
            logical_z_qubits += patch.logical_z_qubits
        return logical_x_qubits, logical_z_qubits
    
    def repeated_syndrome_round(
            self, 
            circ: stim.Circuit, 
            num_rounds: int = 1,
            **kwargs
        ) -> stim.Circuit:
        """Add stim code for num_rounds syndrome rounds to input circuit. Uses
        self.active_qubits to decide which qubits to apply operations to.

        Args:
            circ: Stim circuit to add to.
            num_rounds: Number of syndrome rounds to be repeated
        
        Returns:
            Modified Stim circuit with a repeat block of num_rounds syndrome rounds appended.
        """
        num_measures = len([measure.idx for measure in self.ancilla
                               if self.qubits_active[measure.idx]])
        
        circ.append(stim.CircuitRepeatBlock(num_rounds, self.syndrome_round(stim.Circuit(), **kwargs)))

        for round in self.meas_record[-2::-1]:
            for q, idx in round.items():
                round[q] = idx - num_measures * (num_rounds - 1) # Already measured once in self.syndrome_round
        
        return circ

