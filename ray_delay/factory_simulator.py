from ray_delay.ray_detector import RayDetectorSpec

class RayDelaySimulator:
    """Calculates statistics for turning-patch-offline method of mitigating
    cosmic rays in magic state factories.
    """
    def __init__(
            self, 
            ray_detector_spec: RayDetectorSpec,
            magic_state_factory: MagicStateFactory,
        ):
        raise NotImplementedError