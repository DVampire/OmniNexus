import os
from collections import OrderedDict

import torch

from rp.logger import research_logger as logger
from rp.registry import TRAINER


@TRAINER.register_module(force=True)
class Trainer:
    def __init__(
        self,
        config,
        model,
        train_loader,
        valid_loader,
        test_loader,
        wandb_logger,
        accelerator,
        optimizer,
        scheduler,
        criterion,
        metrics,
        device,
        dtype,
        exp_path,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.wandb_logger = wandb_logger
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics  # list of metrics
        self.criterion = criterion
        self.device = device
        self.dtype = dtype
        self.exp_path = exp_path

        self._init_params()

    def _init_params(self):
        self.is_main_process = self.accelerator.is_local_main_process

        if self.is_main_process:
            logger.info('| Init parameters for trainer...')

        torch.set_default_dtype(self.dtype)

        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.valid_loader = self.accelerator.prepare(self.valid_loader)
        self.test_loader = self.accelerator.prepare(self.test_loader)

        self.model = self.accelerator.prepare(self.model)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)
        self.criterion = self.accelerator.prepare(self.criterion)

        self.checkpoint_path = self.config.checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.wandb_path = self.config.wandb_path
        os.makedirs(self.wandb_path, exist_ok=True)

        if self.is_main_process:
            logger.info(f'| Checkpoint path: {self.checkpoint_path}')
            logger.info(f'| Wandb path: {self.wandb_path}')

    def _train_epoch(self):
        total_loss = 0

        for i, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        info = OrderedDict({'loss': total_loss / len(self.train_loader)})

        return info

    def _valid_epoch(self):
        total_loss = 0
        correct = {}
        total = 0

        with torch.no_grad():
            for i, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                total += target.size(0)

                metrics = OrderedDict()
                for metric in self.metrics:
                    metrics.update(metric(output, target))

                for key, value in metrics.items():
                    if key not in correct:
                        correct[key] = 0
                    correct[key] += value * target.size(0)

        info = OrderedDict(
            {
                'loss': total_loss / len(self.valid_loader),
            }
        )
        info.update({f'{key}': correct[key] / total for key in correct})

        return info

    def train(self):
        if self.is_main_process:
            logger.info('| Start training...')

        for epoch in range(self.config.epochs):
            info = OrderedDict({'epoch': epoch})

            self.model.train()
            train_info = self._train_epoch()
            train_log_dict = {
                'train_' + key: value for key, value in train_info.items()
            }
            info.update(train_log_dict)
            if self.is_main_process:
                self.wandb_logger.log(train_log_dict)

            self.model.eval()
            valid_info = self._valid_epoch()
            valid_log_dict = {
                'valid_' + key: value for key, value in valid_info.items()
            }
            info.update(valid_log_dict)
            if self.is_main_process:
                self.wandb_logger.log(valid_log_dict)

            self.scheduler.step()

            if self.is_main_process:
                log_str = '| '
                for key, value in info.items():
                    log_str += f'{key}: {value:.4f} | '
                logger.info(log_str)

            if epoch % self.config.save_interval == 0:
                if self.is_main_process:
                    logger.info(f'| Saving checkpoint at epoch {epoch}...')
                    # Format the epoch number with leading zeros
                    checkpoint_filename = f'checkpoint_{epoch:06d}.pth'
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.checkpoint_path, checkpoint_filename),
                    )

        if self.is_main_process:
            logger.info('| Finish training...')

    def test(self):
        if self.is_main_process:
            logger.info('| Start testing...')

        self.model.eval()
        total_loss = 0
        correct = {}
        total = 0

        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                total += target.size(0)

                metrics = OrderedDict()
                for metric in self.metrics:
                    metrics.update(metric(output, target))

                for key, value in metrics.items():
                    if key not in correct:
                        correct[key] = 0
                    correct[key] += value * target.size(0)

        info = OrderedDict(
            {
                'loss': total_loss / len(self.test_loader),
            }
        )
        info.update({f'{key}': correct[key] / total for key in correct})

        if self.is_main_process:
            log_str = '| '
            for key, value in info.items():
                log_str += f'{key}: {value:.4f} | '
            logger.info(log_str)

        if self.is_main_process:
            self.wandb_logger.log(info)

        if self.is_main_process:
            logger.info('| Finish testing...')

        return info
