from ray_delay.device_simulator.stim_simulator.lattice_surgery.glue import BoundaryType, GluePatch
from ray_delay.device_simulator.stim_simulator.lattice_surgery.patch_collection import PatchCollection
from ray_delay.device_simulator.stim_simulator.patch import SurfaceCodePatch
import stim

class SurgeryOp():

    def __init__(
            self,
            data_patches: list[SurfaceCodePatch],
            routing_patches: list[SurfaceCodePatch],
            glue_pairs: list[tuple[SurfaceCodePatch, BoundaryType, SurfaceCodePatch, BoundaryType]]
        ) -> None:
        '''
        Args:
            data_patches: List of data patches involved in lattice surgery operation
            routing_patches: List of routing patches involved in lattice surgery operation
            glue_pairs: List of pairs of patches to be merged with corresponding boundaries
        
        Asserts:
            All patches have same temporal dimension, dm
        '''
        self.data_patches = data_patches
        self.routing_ptaches = routing_patches
        self.glue_patches = []
        for patch1, boundary1, patch2, boundary2 in glue_pairs:
            self.glue_patches += [GluePatch(patch1, boundary1, patch2, boundary2)]
        self.dm = data_patches[0].dm
        for patch in data_patches + routing_patches:
            assert patch.dm == self.dm

        self.patch_collection = PatchCollection(self.data_patches + self.routing_ptaches + self.glue_patches)
        
        
    def merge_patches(self) -> None:
        '''
        Merges all patches
        '''
        for glue in self.glue_patches:
            glue.merge()
    
    def split_patches(self) -> None:
        '''
        Splits all patches
        '''
        for glue in self.glue_patches:
            glue.split()
    
    def get_stim(self, dm: int = None) -> stim.Circuit:
        '''
        Returns a stim circuit representing the lattice surgery operation.
        Patches are merged for dm rounds then split.
        After splitting, routing patches are measured out and the logical observables 
        for data patches are updated with necessary measurement results
        '''
        if not dm:
            dm = self.dm
        raise NotImplementedError