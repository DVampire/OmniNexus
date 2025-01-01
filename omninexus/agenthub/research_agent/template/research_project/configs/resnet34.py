from copy import deepcopy

# fixed parameters
workdir = 'workdir'
tag = 'resnet34'
exp_path = f'{workdir}/{tag}'
project_name = 'research_project'
wandb_path = 'wandb'
checkpoint_path = 'checkpoints'
num_classes = 2
dtype = 'float32'
num_workers = 4

# configurable parameters
seed = 42
lr = 1e-3
epochs = int(1e3)
batch_size = 32
num_warmup_epochs = 10


transform = dict(type='ImageTransform', mode='')

train_transform = deepcopy(transform)
train_transform['mode'] = 'train'

val_transform = deepcopy(transform)
val_transform['mode'] = 'valid'

test_transform = deepcopy(transform)
test_transform['mode'] = 'test'

dataset: dict[str, str] = dict(type='ImageDataset', image_dir='', transform='')

train_dataset = deepcopy(dataset)
train_dataset.update(
    {'image_dir': 'datasets/cat_and_dog/train', 'transform': train_transform}
)

val_dataset = deepcopy(dataset)
val_dataset.update(
    {'image_dir': 'datasets/cat_and_dog/test', 'transform': val_transform}
)

test_dataset = deepcopy(dataset)
test_dataset.update(
    {'image_dir': 'datasets/cat_and_dog/test', 'transform': test_transform}
)

model = dict(type='ResNet34', num_classes=num_classes)

criterion = dict(type='CrossEntropyLoss')

optimizer = dict(type='Adam', lr=lr, params=None)

scheduler = dict(
    type='CosineWithWarmupScheduler',
    optimizer=None,
    num_warmup_steps=None,
    num_training_steps=None,
)

metrics = [
    dict(type='Accuracy', topk=(1,)),
    dict(
        type='F1Score',
    ),
]

trainer = dict(
    type='Trainer',
    model=None,
    train_loader=None,
    valid_loader=None,
    test_loader=None,
    wandb_logger=None,
    accelerator=None,
    optimizer=None,
    scheduler=None,
    criterion=None,
    metrics=None,
    device=None,
    dtype=None,
    exp_path=None,
)
