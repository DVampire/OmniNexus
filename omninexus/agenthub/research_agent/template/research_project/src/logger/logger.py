import logging

from src.registry import LOGGER


@LOGGER.register_module(force=True)
class Logger(logging.Logger):
    def __init__(self, name='logger', level=logging.INFO):
        """A custom logger class for enhanced console logging with colored output.

        :param name: Name of the logger.
        :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        super().__init__(name, level)

        # Formatter for log messages
        formatter = logging.Formatter(
            fmt='\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        # Console handler for logging to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        # Add the handler to the logger instance
        self.addHandler(console_handler)
        self.propagate = (
            False  # Prevent duplicate logs from propagating to the root logger
        )


logger = Logger(name='logger')
