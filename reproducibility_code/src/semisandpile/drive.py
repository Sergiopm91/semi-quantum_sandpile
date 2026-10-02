from __future__ import annotations

import numpy as np

from .config import DriveConfig


def generate_drive(size: int, steps: int, seed: int, drive: DriveConfig):
    """Generate the external drive with NumPy Generator(PCG64)."""
    drive.validate()
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = rng.integers(0, size, size=steps, dtype=np.int32)
    cols = rng.integers(0, size, size=steps, dtype=np.int32)
    if drive.mode == "classical":
        increments = np.ones(steps, dtype=np.float64)
    elif drive.mode == "fixed_beta":
        increments = np.full(steps, drive.beta_abs**2, dtype=np.float64)
    elif drive.mode == "uniform_p":
        increments = rng.uniform(drive.p_low, drive.p_high, size=steps).astype(np.float64)
    elif drive.mode == "beta_p":
        increments = rng.beta(drive.beta_a, drive.beta_b, size=steps).astype(np.float64)
    elif drive.mode == "bernoulli_p":
        increments = (rng.random(steps) < drive.bernoulli_prob).astype(np.float64)
    else:
        raise ValueError(drive.mode)
    return rows, cols, increments
