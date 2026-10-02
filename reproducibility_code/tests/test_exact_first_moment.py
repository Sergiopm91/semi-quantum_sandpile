import numpy as np
from semisandpile.analysis import exact_mean_topplings_uniform_drive


def test_exact_mean_topplings_small_lattices():
    assert np.isclose(exact_mean_topplings_uniform_drive(1), 0.25, atol=1e-14)
    assert np.isclose(exact_mean_topplings_uniform_drive(2), 0.5, atol=1e-14)
    assert np.isclose(exact_mean_topplings_uniform_drive(4), 29.0/24.0, atol=1e-13)
