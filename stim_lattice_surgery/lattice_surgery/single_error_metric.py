import sys
sys.path.append('../../')
sys.path.append('../../ray-delay')

from typing import Any

from stim_surface_code.patch import SurfaceCodePatch
from stim_surface_code.memory import MemoryPatch
from stim_lattice_surgery.lattice_surgery.zxxz_patch import ZXXZPatch
from stim_lattice_surgery.lattice_surgery.glue import GluePatch, BoundaryType
from stim_lattice_surgery.lattice_surgery.surgery_op import SurgeryOp
from ray_delay.noise_model_patch import NoiseModelPatch
from ray_delay.noise_model import CosmicRayNoiseParams, GoogleNoiseParamsNoRandomRays
import numpy as np

import sinter
import matplotlib.pyplot as plt

def initialize_ops(d_range: list[int]) -> list[SurgeryOp]:
    ops = []
    for d in d_range:
        # patch1 = ZXXZPatch(d, d, d, id_offset=0)
        # patch2 = ZXXZPatch(d, d, d, id_offset=2*d**2 - 1)
        patch1 = MemoryPatch(d, d, d)
        patch1.init_reset_h_x = patch1.data
        patch1.init_reset_h_z = []
        patch2 = MemoryPatch(d, d, d, id_offset=2*d**2 - 1)
        patch2.init_reset_h_x = patch2.data
        patch2.init_reset_h_z = []
        zz_op = SurgeryOp([patch1, patch2], [], [(patch1, BoundaryType.TOP, patch2, BoundaryType.BOTTOM, 4*d**2 - 2)])
        ops.append(zz_op)
    return ops

def _get_many_circ_code(
        init_basis: str, 
        observable_basis: str,
        meas_mode: str, 
        d_range: list[int], 
        noise_param_list: list[CosmicRayNoiseParams] = [], 
        op_list: list[list[tuple[SurgeryOp, NoiseModelPatch]]] = [],
    ):
    """TODO
    
    Args:
        TODO
        patch_list: If nonempty, must be a 2D nested list of shape
            (len(d_range), n).
    """
    assert len(d_range) > 0
    assert len(op_list) == 0 or len(d_range) == len(op_list)
    assert len(noise_param_list) == 0 or len(op_list) == 0, 'Cannot provide both noise_param_list and patch_list'

    if len(noise_param_list) > 0: # op_list = []
        base_op_list = initialize_ops(d_range)
        for i, noise_params in enumerate(noise_param_list):
            nmps = [(op, NoiseModelPatch(op.patch_collection, noise_params)) for op in base_op_list]
            op_list.append(nmps)

    assert len(d_range) == len(op_list)

    tasks = []
    for d,patches in zip(d_range, op_list):
        for patch_idx,(op, _) in enumerate(patches):
            circ = op.get_stim(init_basis=init_basis, observable_basis=observable_basis, meas_mode=meas_mode, expect_bell=True)
            
            tasks.append(sinter.Task(circuit=circ, json_metadata={'noise_idx': patch_idx, 'd': int(d), 'meas_mode': meas_mode, 'init_basis': init_basis, 'observable_basis': observable_basis}))
    return tasks

def _get_all_xxtl_code(
        d_range: list[int], 
        noise_param_list: list[CosmicRayNoiseParams] = [], 
        op_list: list[list[tuple[SurgeryOp, NoiseModelPatch]]] = [],
    ):
    return _get_many_circ_code('Z', 'Z', '01TL', d_range, noise_param_list=noise_param_list, op_list=op_list)

def _get_all_zl_code(
        d_range: list[int], 
        noise_param_list: list[CosmicRayNoiseParams] = [], 
        op_list: list[list[tuple[SurgeryOp, NoiseModelPatch]]] = [],
    ):
    return _get_many_circ_code('X', 'X', '', d_range, noise_param_list=noise_param_list, op_list=op_list)

def _post_process_results(results: list[sinter.TaskStats]):
    # for each p, d, combo, we want to store the error rates
    
    raw_TL = {}
    raw_ZL = {}
    processed = {}
    
    for res in results:
        d = res.json_metadata['d']
        noise_idx = res.json_metadata['noise_idx']
        meas_mode = res.json_metadata['meas_mode']
        
        # init_basis = res.json_metadata['init_basis']
        # observable_basis = res.json_metadata['observable_basis']
        key = (d, noise_idx)
        
        if meas_mode == '01TL':
            raw_TL[key] = {'shots': res.shots, 'errs': res.custom_counts}
        else:
            raw_ZL[key] = {'shots': res.shots, 'errs': res.custom_counts}
    
    
    # enumerate through both dicts and calculate the error rates
    for dp_pair in raw_TL.keys():
        d, noise_idx = dp_pair
        
        out = {}
        
        tl_data = raw_TL[dp_pair]
        tl_shots = tl_data['shots']
        
        zl_data = raw_ZL[dp_pair]
        zl_shots = zl_data['shots']
        
        out['tl'] = {k: v / tl_shots for k, v in tl_data['errs'].items()}
        out['zl'] = {k: v / zl_shots for k, v in zl_data['errs'].items()}
        
        tl_err_rate = 0
        
        # tl err rates
        for err_type, err_rate in tl_data['errs'].items():
            err_name = err_type.split('=')[1]
            
            # will be X0, X1, TL
            # first error is 100% bad, second error doesn't matter, third error matters half
            if err_name.startswith('E'):
                tl_err_rate += err_rate
            elif err_name.endswith('E'):
                tl_err_rate += err_rate / 2
        tl_err_rate = tl_err_rate / tl_shots
        
        out['tl_err_rate'] = tl_err_rate
        
        # zl err rates
        assert len(out['zl']) <= 1, out['zl']
        zl_err_rate = out['zl'].get('obs_mistake_mask=E', 0)
        out['zl_err_rate'] = zl_err_rate
        
        out['error_rate'] = 1 - (1 - tl_err_rate) * (1 - zl_err_rate)
        processed[dp_pair] = out
    
    return processed, raw_TL, raw_ZL

def get_processed_error_data(
        d_range: list[int], 
        noise_param_list: list[CosmicRayNoiseParams] = [], 
        op_list: list[list[tuple[SurgeryOp, NoiseModelPatch]]] = [], 
        **sinter_kwargs: dict[str, Any],
    ):
    xxtl_circs = _get_all_xxtl_code(d_range, noise_param_list=noise_param_list, op_list=op_list)
    zl_circs = _get_all_zl_code(d_range, noise_param_list=noise_param_list, op_list=op_list)
    circs = xxtl_circs + zl_circs
    
    print(f'Running {len(circs)} circuits')
    
    combined_sinter_kwargs: dict[str, Any] = {
        'num_workers': 6,
        'max_shots': 10_000_000,
        'max_errors': 100,
        'decoders': ['pymatching'],
        'count_observable_error_combos': True,
    }
    combined_sinter_kwargs.update(sinter_kwargs)

    results = sinter.collect(
        tasks=circs,
        **combined_sinter_kwargs
    )
    return _post_process_results(results)