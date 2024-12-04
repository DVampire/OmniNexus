from torch.nn import CrossEntropyLoss, MSELoss

from rp.registry import CRITERION

CRITERION.register_module(name='MSELoss', module=MSELoss)
CRITERION.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
