import logging
import sys
from typing import Optional

import datasets
import transformers
from transformers import TrainingArguments

# Global variable to track if logging has been set up
_logging_setup_done = False
_level = logging.INFO


def get_logger(name: str) -> logging.Logger:
    global _logging_setup_done
    if not _logging_setup_done:
        setup_logging()
    
    logger = logging.getLogger(name)
    logger.setLevel(_level)
    return logger

def setup_logging(output_file: Optional[str] = None) -> logging.Logger:
    """
    Set up basic logging configuration for the project.

    Args:
        log_level (str): The logging level for the project.
        output_file (Optional[str]): Path to a file where logs should be written. If None, logs are only streamed to stdout.
        should_log (bool): Whether or not the current process should produce log.
    """
    global _logging_setup_done

    if _logging_setup_done:
        raise ValueError("Logging already setup. Call setup_logging first.")

    # Convert string log level to corresponding logging constant
    # numeric_level = getattr(logging, log_level.upper(), None)
    # if not isinstance(numeric_level, int):
    #     raise ValueError(f"Invalid log level: {log_level}")

    # Create handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if output_file:
        handlers.append(logging.FileHandler(output_file, mode='a'))

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers,
        # level=numeric_level,
    )
    _logging_setup_done = True


def setup_transformers_logging(transformers_log_level: str = "INFO", should_log: bool = True) -> logging.Logger:
    """
    Set up basic logging configuration for the project.

    Args:
        transformers_log_level (str): The logging level for the transformers library.
        output_file (Optional[str]): Path to a file where logs should be written. If None, logs are only streamed to stdout.
        should_log (bool): Whether or not the current process should produce log.
    """
    if should_log:
        # The default of log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    
    # Configure transformers logging
    transformers.utils.logging.set_verbosity(transformers_log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Configure datasets logging
    datasets.utils.logging.set_verbosity(transformers_log_level)

    global _level
    _level = transformers_log_level