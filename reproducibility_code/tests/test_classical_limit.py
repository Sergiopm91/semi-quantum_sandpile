from dataclasses import replace
import numpy as np

from semisandpile.config import DriveConfig, OutputConfig, SimulationConfig
from semisandpile.simulation import simulate


def test_semquantum_beta_one_equals_classical():
    common = dict(
        size=16,
        initial_height=3.0,
        steps=500,
        seed=42,
        engine="numba",
        output=OutputConfig(directory="results/test", make_plots=False),
    )
    classical = simulate(SimulationConfig(**common, drive=DriveConfig(mode="classical")))
    semi = simulate(SimulationConfig(**common, drive=DriveConfig(mode="fixed_beta", beta_abs=1.0)))
    assert np.array_equal(classical.rows, semi.rows)
    assert np.array_equal(classical.cols, semi.cols)
    assert np.array_equal(classical.avalanche_size, semi.avalanche_size)
    assert np.array_equal(classical.avalanche_area, semi.avalanche_area)
    assert np.array_equal(classical.dissipated, semi.dissipated)
    assert np.array_equal(classical.final_lattice, semi.final_lattice)
