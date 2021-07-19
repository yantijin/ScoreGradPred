from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import DistributionOutput
from typing import *

class score_fn_output(DistributionOutput):
    @validated()
    def __init__(self, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond): 
        return (cond,)

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)