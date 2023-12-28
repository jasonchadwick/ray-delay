"""TODO
"""
import time
import multiprocessing as mp
from typing import Any

class ParallelObject(mp.Process):
    """Class to wrap around an object and speed up repeated simulations
    by maintaining it in a consistent state in a parallel thread.
    
    Attributes:
        object: Object to parallelize.
        in_queue: Input queue of actions to be carried out by the patch
            simulator.
        out_queue: Output queue of results of patch simulator actions.
    """
    def __init__(self, object: Any, daemon: bool = True):
        """Initialize with an object.

        Args:
            object: Object to parallelize.
            daemon: Initialization argument to multiprocess.Process class.
        """
        super().__init__(daemon=daemon)
        self.object = object
        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()
    
    def run(self) -> None:
        """Start the process. The process continuously checks action queue for
        new actions, executes them, and adds the results to the output queue.
        """
        while True:
            while not self.in_queue.empty():
                (command, args, kwargs) = self.in_queue.get()
                try:
                    assert hasattr(self.object, command), "Command does not exist"
                    fun = getattr(self.object, command)
                    output = fun(*args, **kwargs)
                    self.out_queue.put(output)
                except Exception as e:
                    self.out_queue.put(e)
                    raise e

            time.sleep(0.01) # Avoid unnecessary clock cycles