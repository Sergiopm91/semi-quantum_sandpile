import numpy as np

from semisandpile.config import DriveConfig
from semisandpile.drive import generate_drive


def test_drive_is_seed_reproducible():
    a = generate_drive(20, 100, 123, DriveConfig(mode="beta_p", beta_a=2, beta_b=5))
    b = generate_drive(20, 100, 123, DriveConfig(mode="beta_p", beta_a=2, beta_b=5))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
