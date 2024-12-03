import importlib
import os.path
import sys
from pathlib import Path

import torch

root = Path(__file__).parents[1]
sys.path.append(str(root))


def test_logger():
    try:
        Logger = importlib.import_module('rp.logger').Logger
    except ImportError:
        raise ImportError('Logger not found')

    logger = Logger('test_logger')
    logger.info('This is an info message')
    logger.debug('This is a debug message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')


def test_dataset():
    try:
        ImageDataset = importlib.import_module('rp.data').ImageDataset
        from torchvision import transforms
    except ImportError:
        raise ImportError('Dataset not found')

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            ),  # Normalize to [-1, 1]
        ]
    )
    dataset = ImageDataset(
        image_dir=os.path.join(root, 'datasets', 'cat_and_dog', 'train'),
        transform=transform,
    )
    print(dataset)


def test_transforms():
    try:
        ImageTransforms = importlib.import_module('rp.data').ImageTransforms
    except ImportError:
        raise ImportError('Transforms not found')

    transform = ImageTransforms(mode='train')
    print(transform)


def test_wandb_logger():
    try:
        WandbLogger = importlib.import_module('rp.wandb').WandbLogger
    except ImportError:
        raise ImportError('WandbLogger not found')

    logger = WandbLogger('test_project', 'test_run')
    logger.initialize()
    logger.log({'loss': 0.1, 'accuracy': 0.9})
    logger.finish()


def test_criterion():
    try:
        Criterion = importlib.import_module('rp.criterion').MSELoss
    except ImportError:
        raise ImportError('Criterion not found')

    criterion = Criterion()
    print(criterion)


def test_model():
    try:
        ResNet34 = importlib.import_module('rp.model').ResNet34
    except ImportError:
        raise ImportError('Model not found')

    model = ResNet34(num_classes=2)
    print(model)


def test_optimizer():
    try:
        Optimizer = importlib.import_module('rp.optimizer').Adam
        ResNet34 = importlib.import_module('rp.model').ResNet34
    except ImportError:
        raise ImportError('Optimizer not found')

    model = ResNet34(num_classes=2)
    optimizer = Optimizer(model.parameters(), lr=1e-3)
    print(model)
    print(optimizer)


def test_scheduler():
    try:
        Optimizer = importlib.import_module('rp.optimizer').Adam
        ResNet34 = importlib.import_module('rp.model').ResNet34
        Scheduler = importlib.import_module('rp.scheduler').CosineWithWarmupScheduler
    except ImportError:
        raise ImportError('Scheduler not found')

    model = ResNet34(num_classes=2)
    optimizer = Optimizer(model.parameters(), lr=1e-3)
    scheduler = Scheduler(optimizer, num_warmup_steps=100, num_training_steps=1000)
    print(scheduler)


def test_metric():
    try:
        Accuracy = importlib.import_module('rp.metric').Accuracy
    except ImportError:
        raise ImportError('Metric not found')

    metric = Accuracy(topk=(1,))
    pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    target = torch.tensor([1, 0])
    print(metric(pred, target))


if __name__ == '__main__':
    test_logger()
    test_dataset()
    test_transforms()
    test_wandb_logger()
    test_criterion()
    test_model()
    test_optimizer()
    test_scheduler()
    test_metric()
