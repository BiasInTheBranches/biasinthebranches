"""Implementation of the logging module for the evaluation workflow."""

from __future__ import annotations

import logging

import datasets

# Setup datasets logging level
datasets.logging.set_verbosity_error()
datasets.disable_progress_bars()


def setup_logging() -> logging.Logger:
    """Set up logging for the evaluation workflow."""
    logger = logging.getLogger('harness')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


# Initialize logger once when the module is imported
logger = setup_logging()
