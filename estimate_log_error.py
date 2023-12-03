# See https://docs.google.com/document/d/1v3C2-PpiuXBmunc4OYsGjhJZ-0GLeR0mhbHUiLMyUNA/edit
# for agenda / notes

"""
When estimating patch error rates, we need to set some broad categories of parameters:
- Patch parameters: these define the patch properties, including 
the distance of the patch, 
the number of cycles measured, 
whether the surface code is XZZX, etc.
- Cosmic ray parameters: these define the type of ray event that occurs; note that we employ
a noise injection approach, meaning that we purposefully inject a ray event into the patch.
Some parameters include 
the diameter of the cosmic ray, 
the duration, 
the severity, and 
whether the ray is oblong or circular. 

The experiment has three stages:
1. Initialization: for some time t_0, no perturbation occurs other than the typical noise model,
and the experiment continues.
2. Perturbation: At time t_0, a perturbation will occur, marking a shift in the noise model.
This perturbation may be before or after scheduled events.
3. Closure: At time t*, the experiment ends, whether or not the perturbation has ended. 
At this time, decoding is finalized and we assess whether the patch has failed or not.
"""

def estimate_error_X(
    d: int,
    ray_size: int,
    
):
    pass

    # initialization
    # need to create the circuit and repeat the relevant ticks until t0
    # we may also choose to perform some logical operation here or Idle
    # TODO: identify where to put this -- is this immediately after initialization?
    
    # perturbation
    # now, from min(t0 + duration, t*), we need to add the cosmic ray
    # so modify the noise model but keep the surface code circuit the same
    # and repeat until min(t0 + duration, t*)
    
    # closure
    # from min(t0 + duration, t*) to t*, resume the original noise model
    
    
    # finally, perform measurement and decoding
    # and assess the patch's error rate




