from stim_surface_code.patch import SurfaceCodePatch
import stim

class ZXXZPatch(SurfaceCodePatch):
    '''
    ZXXZ Surface Code patch which has balanced noise channels for Z/X
    '''

    def __init__(self, *args, **kwargs) -> None:
        '''
        SurfaceCodePatch initilization
        '''
        super().__init__(*args, **kwargs)
        self.init_reset_h_z = []
        self.init_reset_h_x = []
        for meas in self.z_ancilla:
            self.init_reset_h_z += [meas.data_qubits[1], meas.data_qubits[2]]
        for meas in self.x_ancilla:
            self.init_reset_h_x += [meas.data_qubits[1], meas.data_qubits[2]]
    
    def syndrome_round(
        self, 
        circ: stim.Circuit, 
        deterministic_detectors: list[int] = [],
        inactive_detectors: list[int] = [],
    ) -> stim.Circuit:
        """Add stim code for one syndrome round to input circuit. Uses
        self.active_qubits to decide which qubits to apply operations to.

        Args:
            circ: Stim circuit to add to.
            deterministic_detectors: list of ancilla whose measurements are
                (ideally, with no errors) deterministic.
            inactive_detectors: list of detectors to NOT enforce this round (but
                stabilizer is still measured)
        
        Returns:
            Modified Stim circuit with a syndrome round appended.
        """
        self.apply_reset(circ, [measure.idx for measure in self.ancilla
                                if self.qubits_active[measure.idx]])
        
        self.apply_1gate(circ, 'H', [measure.idx for measure in self.ancilla])

        for i in range(4):
            if i == 1 or i == 3:
                self.apply_1gate(circ, 'H', [data.idx 
                                     for data in self.data
                                     if self.qubits_active[data.idx]])
            err_qubits = []
            for measure in self.x_ancilla:
                if self.qubits_active[measure.idx]:
                    dqi = measure.data_qubits[i]
                    if dqi != None:
                        err_qubits += [(measure.idx, dqi.idx)]
            for measure in self.z_ancilla:
                if self.qubits_active[measure.idx]:
                    dqi = measure.data_qubits[i]
                    if dqi != None:
                        err_qubits += [(dqi.idx, measure.idx)]
            self.apply_2gate(circ,'CZ',err_qubits)
        
        self.apply_1gate(circ, 'H', [measure.idx for measure in self.ancilla])

        # Measure
        self.apply_meas(circ, [measure.idx for measure in self.ancilla
                               if self.qubits_active[measure.idx]])

        for ancilla in self.ancilla:
            if (self.qubits_active[ancilla.idx] 
                and ancilla.idx not in inactive_detectors):
                if ancilla.idx in deterministic_detectors:
                    # no detector history to compare
                    circ.append(
                        'DETECTOR', 
                        self.get_meas_rec(-1, ancilla.idx), 
                        ancilla.coords + (0,)
                    )
                else:
                    # compare detector to a previous round
                    circ.append(
                        'DETECTOR', 
                        [self.get_meas_rec(-1, ancilla.idx),
                        self.get_meas_rec(-2, ancilla.idx)],
                        ancilla.coords + (0,)
                    )

        circ.append('TICK')
        circ.append('SHIFT_COORDS', [], [0.0, 0.0, 1.0])

        return circ
 