from rp.registry import CRITERION
from torch.nn import CrossEntropyLoss, MSELoss

CRITERION.register_module(name='MSELoss', module=MSELoss)
CRITERION.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
