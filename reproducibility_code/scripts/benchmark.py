from dataclasses import replace
import time

from semisandpile.config import DriveConfig, OutputConfig, SimulationConfig
from semisandpile.simulation import simulate

base = SimulationConfig(
    size=100,
    initial_height=3.0,
    steps=10_000,
    burn_in=0,
    seed=1234,
    engine="numba",
    drive=DriveConfig(mode="classical"),
    output=OutputConfig(directory="results/benchmark", make_plots=False),
)

for engine in ("numba", "numpy"):
    cfg = replace(base, engine=engine)
    result = simulate(cfg)
    print(f"{engine:>6}: {result.runtime_seconds:.3f} s, topplings={result.avalanche_size.sum()}")

print("CuPy is intentionally not benchmarked automatically; use engine=cupy on a CUDA machine.")
