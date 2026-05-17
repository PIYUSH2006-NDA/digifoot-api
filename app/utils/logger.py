"""
Structured logging utility.
Provides a pre-configured logger for the entire application.
"""

import logging
import sys

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
