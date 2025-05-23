import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel, 
    RowwiseParallel, 
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False