from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DriveConfig:
    mode: str = "classical"  # classical | fixed_beta | uniform_p | beta_p | bernoulli_p
    beta_abs: float = 1.0
    p_low: float = 0.0
    p_high: float = 1.0
    beta_a: float = 2.0
    beta_b: float = 2.0
    bernoulli_prob: float = 0.5

    def validate(self) -> None:
        if self.mode not in {"classical", "fixed_beta", "uniform_p", "beta_p", "bernoulli_p"}:
            raise ValueError(f"Unsupported drive.mode={self.mode!r}")
        if not 0.0 <= self.beta_abs <= 1.0:
            raise ValueError("drive.beta_abs must be in [0, 1]")
        if not (0.0 <= self.p_low <= self.p_high <= 1.0):
            raise ValueError("uniform_p requires 0 <= p_low <= p_high <= 1")
        if self.beta_a <= 0 or self.beta_b <= 0:
            raise ValueError("beta_p shape parameters must be > 0")
        if not 0.0 <= self.bernoulli_prob <= 1.0:
            raise ValueError("bernoulli_prob must be in [0,1]")


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "results/run"
    save_csv: bool = True
    save_npz: bool = True
    save_lattice: bool = True
    make_plots: bool = True


@dataclass(frozen=True)
class SimulationConfig:
    size: int = 100
    initial_height: float = 3.0
    threshold: float = 4.0
    steps: int = 100_000
    burn_in: int = 0
    seed: int = 20260905
    engine: str = "numba"  # numba | numpy | cupy | auto
    boundary: str = "open"
    epsilon: float = 1e-12
    redistribution: str = "discrete"  # discrete | dirichlet
    redistribution_seed: int = 20260906
    drive: DriveConfig = field(default_factory=DriveConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.size < 2:
            raise ValueError("size must be >= 2")
        if not 0.0 <= self.initial_height < self.threshold:
            raise ValueError("initial_height must satisfy 0 <= initial_height < threshold")
        if abs(self.threshold - 4.0) > 1e-15:
            raise ValueError("The implementation uses the square-lattice threshold 4.")
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if not 0 <= self.burn_in < self.steps:
            raise ValueError("burn_in must satisfy 0 <= burn_in < steps")
        if self.engine not in {"numba", "numpy", "cupy", "auto"}:
            raise ValueError("engine must be one of: numba, numpy, cupy, auto")
        if self.boundary != "open":
            raise ValueError("Only open/dissipative boundaries are implemented")
        if self.epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if self.redistribution not in {"discrete", "dirichlet"}:
            raise ValueError("redistribution must be discrete or dirichlet")
        if self.redistribution == "dirichlet" and self.engine not in {"numba", "auto"}:
            raise ValueError("dirichlet redistribution is implemented in the Numba engine")
        self.drive.validate()


def _deep_get(data: dict[str, Any], key: str, default: Any) -> Any:
    return data.get(key, default)


def load_config(path: str | Path) -> SimulationConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    lattice = raw.get("lattice", {})
    sim = raw.get("simulation", {})
    drive_raw = raw.get("drive", {})
    output_raw = raw.get("output", {})
    drive = DriveConfig(
        mode=_deep_get(drive_raw, "mode", "classical"),
        beta_abs=float(_deep_get(drive_raw, "beta_abs", 1.0)),
        p_low=float(_deep_get(drive_raw, "p_low", 0.0)),
        p_high=float(_deep_get(drive_raw, "p_high", 1.0)),
        beta_a=float(_deep_get(drive_raw, "beta_a", 2.0)),
        beta_b=float(_deep_get(drive_raw, "beta_b", 2.0)),
        bernoulli_prob=float(_deep_get(drive_raw, "bernoulli_prob", 0.5)),
    )
    output = OutputConfig(
        directory=str(_deep_get(output_raw, "directory", "results/run")),
        save_csv=bool(_deep_get(output_raw, "save_csv", True)),
        save_npz=bool(_deep_get(output_raw, "save_npz", True)),
        save_lattice=bool(_deep_get(output_raw, "save_lattice", True)),
        make_plots=bool(_deep_get(output_raw, "make_plots", True)),
    )
    cfg = SimulationConfig(
        size=int(_deep_get(lattice, "size", 100)),
        initial_height=float(_deep_get(lattice, "initial_height", 3.0)),
        threshold=float(_deep_get(lattice, "threshold", 4.0)),
        steps=int(_deep_get(sim, "steps", 100_000)),
        burn_in=int(_deep_get(sim, "burn_in", 0)),
        seed=int(_deep_get(sim, "seed", 20260905)),
        engine=str(_deep_get(sim, "engine", "numba")),
        boundary=str(_deep_get(lattice, "boundary", "open")),
        epsilon=float(_deep_get(sim, "epsilon", 1e-12)),
        redistribution=str(_deep_get(sim, "redistribution", "discrete")),
        redistribution_seed=int(_deep_get(sim, "redistribution_seed", 20260906)),
        drive=drive,
        output=output,
    )
    cfg.validate()
    return cfg
