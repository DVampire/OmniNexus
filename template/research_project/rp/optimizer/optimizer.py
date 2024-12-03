from torch.optim import Adam
from transformers.optimization import AdamW

from rp.registry import OPTIMIZER

OPTIMIZER.register_module(name='AdamW', module=AdamW)
OPTIMIZER.register_module(name='Adam', module=Adam)
