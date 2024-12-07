from ray_delay.device_simulator.stim_simulator.lattice_surgery.glue import BoundaryType
from ray_delay.device_simulator.stim_simulator.patch import SurfaceCodePatch

class PatchCollection(SurfaceCodePatch):
    '''
    A collection of SurfaceCodePatches operating simultaneously
    '''
    
    def __init__(
            self,
            patches: list[SurfaceCodePatch]
        ) -> None:
        '''
        Collect all qubits from many individual patches into a single patch object

        Args:
            patches: Collection of SurfaceCodePatches
        '''
    
    def place_data(self) -> None:
        '''
        Place data qubits in collective device
        '''
    
    def place_ancilla(self) -> None:
        '''
        Place ancilla qubits in collective device
        '''
        