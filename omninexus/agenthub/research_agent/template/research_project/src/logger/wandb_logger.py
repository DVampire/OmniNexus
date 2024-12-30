import torch
import wandb

from src.logger import logger
from src.registry import LOGGER


@LOGGER.register_module(force=True)
class WandbLogger:
    def __init__(self, log_dir, project_name, run_name=None, config=None):
        """A class for managing wandb logging and experiment tracking.

        :param log_dir: The directory to save wandb logs.
        :param project_name: The name of the wandb project.
        :param run_name: Optional, the name of the specific run.
        :param config: Optional, a dictionary containing configuration parameters.
        """
        self.log_dir = log_dir
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.run = None

    def initialize(self):
        """Initialize a wandb run with the given parameters."""
        self.run = wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=self.config,
            dir=self.log_dir,
        )
        logger.info(f'Wandb run initialized: {self.run_name or self.run.name}')

    def log(self, metrics, step=None):
        """Log metrics to wandb.

        :param metrics: A dictionary of metrics to log (e.g., {'loss': 0.1, 'accuracy': 0.9}).
        :param step: Optional, the step at which the metrics are logged.
        """
        if self.run is None:
            raise RuntimeError(
                'Wandb run has not been initialized. Call `initialize()` first.'
            )
        wandb.log(metrics, step=step)

    def save_model(self, model, model_path):
        """Save a model file and upload it to wandb.

        :param model: The model to save.
        :param model_path: The local path to save the model.
        """
        if self.run is None:
            raise RuntimeError(
                'Wandb run has not been initialized. Call `initialize()` first.'
            )

        # Save the model locally
        torch.save(model.state_dict(), model_path)
        logger.info(f'Model saved locally at {model_path}')

        # Upload to wandb
        wandb.save(model_path)
        logger.info(f'Model uploaded to wandb: {model_path}')

    def finish(self):
        """Finish the current wandb run."""
        if self.run is not None:
            self.run.finish()
            logger.info('Wandb run finished.')
        else:
            logger.info('Wandb run was not initialized.')
