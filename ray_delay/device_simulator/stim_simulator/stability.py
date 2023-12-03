import stim
from .patch import SurfaceCodePatch, DataQubit, MeasureQubit
    
class StabilityPatch(SurfaceCodePatch):
    """Surface code patch that performs the memory experiment. See
    https://quantum-journal.org/papers/q-2022-08-24-786/ for details.
    """
    def __init__(
            self, 
            d: int, 
            gate1_time: float = 25e-9, 
            gate2_time: float = 34e-9, 
            mr_time: float = 500e-9,
        ) -> None:
        """Initialize.
        
        Args:
            d: Code distance.
            gate1_time: Duration of single-qubit gates.
            gate2_time: Duration of two-qubit gates.
            mr_time: Duration of readout+reset.
        """
        self.observable_basis = 'Z'
        self.boundary_basis = 'X'

        super().__init__(d, gate1_time, gate2_time, mr_time)

        self.observable_ancilla = (self.z_ancilla if self.observable_basis == 'Z' else self.x_ancilla)
        self.boundary_basis_ancilla = (self.x_ancilla if self.observable_basis == 'Z' else self.z_ancilla)

    def place_data(self) -> list[DataQubit]:
        """Create the device object that will hold all physical qubits, and
        place data qubits within it.
        
        Returns:
            list of DataQubit objects.
        """
        data = []
        q_count = 0
        for x in range(self.d):
            for y in range(self.d):
                if self.d % 2 == 0 or not ((y == 0 and x == self.d-1) or (y == self.d-1 and x == 0)):
                    data.append(DataQubit(q_count, (2*x+1, 2*y+1)))
                q_count += 1
        self.device = [[None for _ in range(2*self.d+1)] for _ in range(2*self.d+1)]
        for d in data:
            self.device[d.coords[0]][d.coords[1]] = d
        return data

    def place_ancilla(self) -> None:
        """Place ancilla (non-data) qubits in the patch. Must be run *after*
        place_data.
        """
        # number of qubits already placed (= index of next qubit)
        q_count = len(self.data)

        self.x_ancilla = []
        self.z_ancilla = []
        for x in range(self.d+1):
            for y in range(self.d+1):
                if (x + y) % 2 == 1 and not ((y == self.d and x == 0) or (y == 0 and x == self.d)):# is syndrome matching boundary basis
                    coords = (2*x, 2*y)
                    if y == 0: # Left edge
                        data_qubits = [None, self.device[coords[0] - 1][coords[1] + 1], None, self.device[coords[0] + 1][coords[1] + 1]]
                    elif y == self.d: # Right edge
                        data_qubits = [self.device[coords[0] - 1][coords[1] - 1], None, self.device[coords[0] + 1][coords[1] - 1], None]
                    elif x == 0: # Top Edge
                        data_qubits = [None, None, self.device[coords[0] + 1][coords[1] - 1], self.device[coords[0] + 1][coords[1] + 1]]
                    elif x == self.d: # Bottom Edge
                        data_qubits = [self.device[coords[0] - 1][coords[1] - 1], self.device[coords[0] - 1][coords[1] + 1], None, None]
                    elif x > 0 and y > 0 and x < self.d and y < self.d: # in the middle
                        data_qubits = [self.device[coords[0] - 1][coords[1] - 1], self.device[coords[0] - 1][coords[1] + 1], self.device[coords[0] + 1][coords[1] - 1], self.device[coords[0] + 1][coords[1] + 1]]
                    else:
                        # Empty position
                        continue
                    measure_q = MeasureQubit(q_count, coords, data_qubits, self.boundary_basis)
                    self.device[coords[0]][coords[1]] = measure_q
                    if self.boundary_basis == "Z":
                        self.z_ancilla.append(measure_q)
                    else:
                        self.x_ancilla.append(measure_q)
                    q_count += 1
                elif (x + y) % 2 == 0 and x > 0 and y > 0 and x < self.d and y < self.d:# is syndrome opposite boundary basis
                    coords = (2*x, 2*y)
                    data_qubits = [self.device[coords[0] - 1][coords[1] - 1], self.device[coords[0] + 1][coords[1] - 1], self.device[coords[0] - 1][coords[1] + 1], self.device[coords[0] + 1][coords[1] + 1]]
                    measure_q = MeasureQubit(q_count, coords, data_qubits, self.observable_basis)
                    self.device[coords[0]][coords[1]] = measure_q
                    if self.observable_basis == "Z":
                        self.z_ancilla.append(measure_q)
                    else:
                        self.x_ancilla.append(measure_q)
                    q_count += 1
    
    def get_stim(self) -> stim.Circuit:
        """Generate Stim code performing a stability experiment in desired basis.
        
        Args:
            observable_basis: Basis to prepare and measure in. Must be 'X', 'Y'
                or 'Z'. 
            ideal_init_and_meas: If True, perform ideal initialization and
                measurement instead of gate-based. Required if basis is 'Y'.
        
        Returns:
            Stim circuit implementing the logical stability experiment.
        """
        assert self.error_vals_initialized

        self.meas_record: list[dict[int, int]] = []
        circ = stim.Circuit()

        # Coords
        for data in self.data:
            circ.append("QUBIT_COORDS", data.idx, data.coords)
        for x_ancilla in self.x_ancilla:
            circ.append("QUBIT_COORDS", x_ancilla.idx, x_ancilla.coords)
        for z_ancilla in self.z_ancilla:
            circ.append("QUBIT_COORDS", z_ancilla.idx, z_ancilla.coords)

        # Syndrome rounds
        self.syndrome_round(
            circ, 
            deterministic_detectors=[q.idx for q in self.observable_ancilla], 
            inactive_detectors=[q.idx for q in self.boundary_basis_ancilla],
        )
        circ.append(stim.CircuitRepeatBlock(self.num_rounds - 1, self.syndrome_round(stim.Circuit())))

        # Measure in basis opposite of boundary basis
        self.reset_meas_qubits(circ, 'M', [data.idx for data in self.data])

        for ancilla in self.observable_ancilla:
            circ.append("DETECTOR", [self.get_meas_rec(-1, data.idx) for data in ancilla.data_qubits if data is not None] + [self.get_meas_rec(-2, ancilla.idx)],
                                     ancilla.coords + (1,))

        # Observable is parity of all stabilizers matching boundary basis
        circ.append("OBSERVABLE_INCLUDE", [self.get_meas_rec(-2, ancilla.idx) for ancilla in self.boundary_basis_ancilla], 0)
        
        return circ
    