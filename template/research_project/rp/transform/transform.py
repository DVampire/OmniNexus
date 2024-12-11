from torchvision import transforms

from rp.logger import research_logger as logger
from rp.registry import TRANSFORM


@TRANSFORM.register_module(force=True)
class ImageTransform(transforms.Compose):
    def __init__(self, mode='train'):
        """A transform selector that dynamically constructs a transform pipeline based on the mode.
        :param mode: 'train', 'valid', or 'test'
        """
        if mode == 'train':
            logger.info('Using training transforms.')
            super().__init__(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                    ),  # Normalize to [-1, 1]
                ]
            )
        elif mode == 'valid':
            logger.info('Using validation transforms.')
            super().__init__(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                    ),  # Normalize to [-1, 1]
                ]
            )
        elif mode == 'test':
            logger.info('Using test transforms.')
            super().__init__(
                [
                    transforms.Resize((224, 224)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                    ),  # Normalize to [-1, 1]
                ]
            )
        else:
            raise ValueError(
                f"Unsupported mode: {mode}. Use 'train', 'valid', or 'test'."
            )
