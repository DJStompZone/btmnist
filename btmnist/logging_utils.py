"""
Lightweight logging utilities: timestamped logger and moving average tracker.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass


def make_logger(name: str = "btmnist", level: int = logging.INFO) -> logging.Logger:
    """
    Create a simple stdout logger with timestamps and levels.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level (e.g., logging.INFO).

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


@dataclass
class MovingAvg:
    """
    Exponential moving average tracker for scalar metrics.

    Attributes
    ----------
    beta : float
        Smoothing factor in [0,1). Closer to 1.0 -> smoother.
    value : float
        Current EMA value.
    initialized : bool
        Whether the EMA has seen a value.
    """
    beta: float = 0.98
    value: float = 0.0
    initialized: bool = False

    def update(self, x: float) -> float:
        """
        Update the EMA with a new observation.

        Parameters
        ----------
        x : float
            New measurement.

        Returns
        -------
        float
            Updated EMA value.
        """
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * x
        return self.value
