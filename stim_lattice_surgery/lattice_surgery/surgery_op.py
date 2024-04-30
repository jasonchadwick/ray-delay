from stim_lattice_surgery.lattice_surgery.glue import BoundaryType, GluePatch
from stim_lattice_surgery.lattice_surgery.patch_collection import PatchCollection
from stim_surface_code.patch import SurfaceCodePatch
import stim

class SurgeryOp():

    def __init__(
            self,
            data_patches: list[SurfaceCodePatch],
            routing_patches: list[SurfaceCodePatch],
            glue_pairs: list[tuple[SurfaceCodePatch, BoundaryType, SurfaceCodePatch, BoundaryType, int]],
            **kwargs
        ) -> None:
        '''
        Args:
            data_patches: List of data patches involved in lattice surgery operation
            routing_patches: List of routing patches involved in lattice surgery operation
            glue_pairs: List of pairs of patches to be merged with corresponding boundaries w/ id_offset for glue patch
        
        Asserts:
            All patches have same temporal dimension, dm
        '''
        self.data_patches = data_patches
        self.routing_patches = routing_patches
        self.glue_patches = []
        for patch1, boundary1, patch2, boundary2, id_offset in glue_pairs:
            self.glue_patches += [GluePatch(patch1, boundary1, patch2, boundary2, id_offset=id_offset)]
        self.dm = data_patches[0].dm
        for patch in data_patches + routing_patches:
            assert patch.dm == self.dm
        
        self.patch_collection = PatchCollection(self.data_patches + self.routing_patches, self.glue_patches, **kwargs)
        self.anti_basis_datas = {glue: glue._get_anti_basis_data() for glue in self.glue_patches}
        self.merge_patches()
        self.update_qubit_pairs()
        self.split_patches()

        
    def update_qubit_pairs(self) -> None:
        # Update qubit pairs
        for i in range(4):
            for measure_x in self.patch_collection.x_ancilla:
                dqi = measure_x.data_qubits[i]
                if dqi != None:
                    self.patch_collection.qubit_pairs.append((measure_x.idx, dqi.idx))
            for measure_z in self.patch_collection.z_ancilla:
                dqi = measure_z.data_qubits[i]
                if dqi != None:
                    self.patch_collection.qubit_pairs.append((dqi.idx, measure_z.idx))

    def merge_patches(self) -> None:
        '''
        Merges all patches

        Decide corner cases, update patch collection operators**
        '''
        for glue in self.glue_patches:
            glue.merge()
        for patch in self.routing_patches + self.glue_patches:
            for qbit in patch.all_qubits:
                self.patch_collection.qubits_active[qbit.idx] = True

    def split_patches(self) -> None:
        '''
        Splits all patches
        '''
        for glue in self.glue_patches:
            glue.split()
        for patch in self.routing_patches + self.glue_patches:
            for qbit in patch.all_qubits:
                self.patch_collection.qubits_active[qbit.idx] = False
        
    
    def get_stim(self, init_basis: str = 'Z', observable_basis: str = 'Z', dm: int = None, expect_bell: bool = False) -> stim.Circuit:
        '''
        Returns a stim circuit representing the lattice surgery operation.
        Patches are merged for dm rounds then split.
        After splitting, routing patches are measured out and the logical observables 
        for data patches are updated with necessary measurement results
        '''
        if not dm:
            dm = self.dm

        init_ancilla = (
            self.patch_collection.z_ancilla if init_basis == 'Z' else self.patch_collection.x_ancilla
        )

        circ = stim.Circuit()

        # Coords
        for qubit in self.patch_collection.all_qubits:
            circ.append('QUBIT_COORDS', qubit.idx, qubit.coords)

        init_h = []
        for data_patch in self.data_patches:
            init_h += (data_patch.init_reset_h_x if init_basis == 'X' else data_patch.init_reset_h_z)
        
        self.patch_collection.apply_reset(circ, [q.idx for data_patch in self.data_patches for q in data_patch.data])
        self.patch_collection.apply_1gate(circ, 'H', [q.idx for q in init_h if q])

        # Init stabilizers
        self.patch_collection.syndrome_round(
            circ,
            deterministic_detectors=[a.idx for a in init_ancilla],
            inactive_detectors=[a.idx for a in self.patch_collection.ancilla 
                                if a not in init_ancilla],
        )

        # Merge + Init merged patch
        self.merge_patches()

        new_ancilla = []

        glue_init_h = []
        glue_data = []
        for glue in self.glue_patches:
            new_ancilla += glue.ancilla + glue.patch1_boundary_stabilizers + glue.patch2_boundary_stabilizers
            glue_data += glue.data
            glue_init_h += (glue.init_reset_h_x if glue.patch1_boundary.basis == 'Z' else glue.init_reset_h_z)
        
        self.patch_collection.apply_reset(circ, [q.idx for q in glue_data])
        self.patch_collection.apply_1gate(circ, 'H', [q.idx for q in glue_init_h if q])

        new_ancilla_data = {
            anc: [q for q in anc.data_qubits] for anc in new_ancilla
        }

        self.patch_collection.syndrome_round(
            circ,
            deterministic_detectors=[],
            inactive_detectors=[a.idx for a in new_ancilla],
        )

        # Weight 2 stabilizers are unchanged due to glue data initialization
        for glue in self.glue_patches:
            for measure in glue.patch1_boundary_stabilizers + glue.patch2_boundary_stabilizers:
                circ.append('DETECTOR', [self.patch_collection.get_meas_rec(-2, measure.idx), 
                                         self.patch_collection.get_meas_rec(-1, measure.idx)],
                                         measure.coords + (0,))

        self.patch_collection.repeated_syndrome_round(circ, dm - 1)
        
        # Split patches
        data_observable_changes = {data_patch: [] for data_patch in self.data_patches}

        if expect_bell:
            for glue in self.glue_patches:
                product_records = (glue.patch1_boundary.basis, self.patch_collection.meas_record[-1], glue._get_merged_product())
                data_observable_changes[glue.patch1].append(product_records)

        # Measure out glue data in opposite basis
        self.patch_collection.apply_1gate(circ, 'H', [q.idx for q in glue_init_h if q])
        self.patch_collection.apply_meas(circ, [q.idx for q in glue_data])

        self.split_patches()

        # Save anti basis glue data for final observable
        for glue in self.glue_patches:
            basis = 'Z' if glue.patch1_boundary.basis == 'X' else 'X'
            data_observable_changes[glue.patch2].append((basis, self.patch_collection.meas_record[-1], [self.anti_basis_datas[glue]]))

        self.patch_collection.syndrome_round(
            circ,
            deterministic_detectors=[],
            inactive_detectors=[a.idx for a in self.patch_collection.ancilla],
        )
        # Process detectors here to adjust meas_rec idx due to glue data measure out
        for ancilla in self.patch_collection.ancilla:
            if ancilla not in new_ancilla:
                circ.append(
                            'DETECTOR', 
                            [self.patch_collection.get_meas_rec(-1, ancilla.idx),
                            self.patch_collection.get_meas_rec(-3, ancilla.idx)],
                            ancilla.coords + (0,)
                        )

        # Product of weight 2 stabilizers + measured out glue data is unchanged
        for glue in self.glue_patches:
            for measure in glue.patch1_boundary_stabilizers + glue.patch2_boundary_stabilizers:
                circ.append('DETECTOR', [self.patch_collection.get_meas_rec(-3, measure.idx), 
                                         self.patch_collection.get_meas_rec(-1, measure.idx)] + 
                                         [self.patch_collection.get_meas_rec(-2, data.idx) for data in new_ancilla_data[measure] if data in glue_data],
                                         measure.coords + (0,))

        self.patch_collection.repeated_syndrome_round(circ, dm - 1)

        # Measure in observable basis
        meas_h = []
        for data_patch in self.data_patches:
            meas_h += (data_patch.init_reset_h_x if observable_basis == 'X' else data_patch.init_reset_h_z)
        
        self.patch_collection.apply_1gate(circ, 'H', [q.idx for q in meas_h if q])
        self.patch_collection.apply_meas(circ, [q.idx for q in self.patch_collection.data if q not in glue_data])
            
        # Check consistency of data qubit measurements with last stabilizer measurement
        for i, data_patch in enumerate(self.data_patches):
            observable_ancilla = data_patch.z_ancilla if observable_basis == 'Z' else data_patch.x_ancilla
            for measure in observable_ancilla:
                data_rec = [self.patch_collection.get_meas_rec(-1, data.idx) for data in measure.data_qubits if data is not None]
                circ.append('DETECTOR', data_rec + [self.patch_collection.get_meas_rec(-2, measure.idx)], measure.coords + (i,))
        
        # Calculate final logical observables
        
        if expect_bell:
            # Bell state observable check
            final_observable = []
            for i, data_patch in enumerate(self.data_patches):
                frame_changes = [stim.target_rec(observable_change[1][meas.idx]) 
                                for observable_change in data_observable_changes[data_patch]
                                for meas in observable_change[2] if observable_change[0] == observable_basis]
                measured_frame = [self.patch_collection.get_meas_rec(-1, data.idx) 
                                for data in (
                                    data_patch.logical_z_qubits if observable_basis == 'Z' 
                                    else data_patch.logical_x_qubits)]
                final_observable += frame_changes + measured_frame

            circ.append('OBSERVABLE_INCLUDE', final_observable, 0)
        else:
            for i, data_patch in enumerate(self.data_patches):
                frame_changes = [stim.target_rec(observable_change[1][meas.idx]) 
                                for observable_change in data_observable_changes[data_patch]
                                for meas in observable_change[2] if observable_change[0] == observable_basis]
                circ.append(
                'OBSERVABLE_INCLUDE', 
                frame_changes + [self.patch_collection.get_meas_rec(-1, data.idx) 
                    for data in (
                        data_patch.logical_z_qubits if observable_basis == 'Z' 
                        else data_patch.logical_x_qubits)], 
                i)
        
        return circ