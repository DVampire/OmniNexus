import argparse
import os

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from rp.config import build_config
from rp.registry import (
    CRITERION,
    DATASET,
    METRIC,
    MODEL,
    OPTIMIZER,
    SCHEDULER,
    TRAINER,
    TRANSFORM,
)
from rp.utils import assemble_project_path, to_torch_dtype
from rp.wandb import WandbLogger


def get_args_parser():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument(
        '--config',
        default=os.path.join('configs', 'resnet34.py'),
        help='config file path',
    )

    parser.add_argument('--workdir', type=str, default='workdir')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--wandb_path', type=str, default=None)
    parser.add_argument('--if_remove', action='store_true', default=False)

    parser.add_argument(
        '--device', default='cuda', help='device to use for training / testing'
    )

    args = parser.parse_args()

    return args


def main(args):
    # 1. build config
    config = build_config(assemble_project_path(args.config), args)

    # 2. set dtype
    dtype = to_torch_dtype(config.dtype)

    # 3. init accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        cpu=True if args.device == 'cpu' else False, kwargs_handlers=[ddp_kwargs]
    )

    # 4. get device
    device = accelerator.device

    # 5. init wandb
    wandb = WandbLogger(
        log_dir=config.wandb_path, project_name=config.project_name, run_name=config.tag
    )
    wandb.initialize()

    # 6. transforms
    train_transform = TRANSFORM.build(config.train_transform)
    val_transform = TRANSFORM.build(config.val_transform)
    test_transform = TRANSFORM.build(config.test_transform)

    # 7. dataset
    train_dataset_config = config.train_dataset
    train_dataset_config.update({'transform': train_transform})
    train_dataset = DATASET.build(config.train_dataset)
    val_dataset_config = config.val_dataset
    val_dataset_config.update({'transform': val_transform})
    val_dataset = DATASET.build(config.val_dataset)
    test_dataset_config = config.test_dataset
    test_dataset_config.update({'transform': test_transform})
    test_dataset = DATASET.build(config.test_dataset)

    # dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 8. model
    model = MODEL.build(config.model)

    # 9. criterion
    criterion = CRITERION.build(config.criterion)

    # 10. optimizer
    optimizer_config = config.optimizer
    optimizer_config.update({'params': model.parameters()})
    optimizer = OPTIMIZER.build(config.optimizer)

    # 11. scheduler
    schdueler_config = config.scheduler
    num_warmpup_steps = int(
        len(train_loader) / config.batch_size * config.num_warmup_epochs
    )
    num_training_steps = int(len(train_loader) / config.batch_size * config.epochs)
    schdueler_config.update(
        {
            'num_warmup_steps': num_warmpup_steps,
            'num_training_steps': num_training_steps,
            'optimizer': optimizer,
        }
    )
    scheduler = SCHEDULER.build(config.scheduler)

    # 12. metric
    metrics = [METRIC.build(metric) for metric in config.metrics]

    # 13. trainer
    trainer_config = config.trainer
    trainer_config.update(
        {
            'config': config,
            'model': model,
            'train_loader': train_loader,
            'valid_loader': val_loader,
            'test_loader': test_loader,
            'wandb_logger': wandb,
            'accelerator': accelerator,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'criterion': criterion,
            'metrics': metrics,
            'device': device,
            'dtype': dtype,
            'exp_path': config.exp_path,
        }
    )
    trainer = TRAINER.build(trainer_config)

    # 14. train
    trainer.train()

    # 15. test
    trainer.test()

    # 16. finish
    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()

    main(args)
