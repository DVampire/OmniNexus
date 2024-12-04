from mmengine.registry import Registry

DATASET = Registry('dataset', locations=['rp.data'])
TRANSFORM = Registry('transform', locations=['rp.transform'])
LOGGER = Registry('logger', locations=['rp.logger'])

MODEL = Registry('model', locations=['rp.model'])
OPTIMIZER = Registry('optimizer', locations=['rp.optimizer'])
SCHEDULER = Registry('scheduler', locations=['rp.scheduler'])
CRITERION = Registry('criterion', locations=['rp.criterion'])
METRIC = Registry('metric', locations=['rp.metric'])
TRAINER = Registry('trainer', locations=['rp.trainer'])
