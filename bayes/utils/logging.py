import logging
from pathlib import Path
import sys


def _setup_logger() -> logging.Logger:
    """Create (or return existing) logger with project-wide format."""
    logger = logging.getLogger("Bayes")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


log: logging.Logger = _setup_logger()
