"""Seeded RNG factories — single source for numpy, torch, and Python random."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed numpy, Python random, and torch (if available) from one int."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def numpy_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
