from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from .analysis import fit_powerlaw_discrete, paper_style_ols, save_plots, save_summary
from .simulation import SimulationResult


def save_result(result: SimulationResult) -> Path:
    out_dir = Path(result.config.output.directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    sl = result.retained_slice

    fit = fit_powerlaw_discrete(result.avalanche_size[sl])
    summary = result.summary()
    summary["paper_style_area_ols"] = paper_style_ols(result.avalanche_area[sl])
    save_summary(summary, fit, out_dir)

    if result.config.output.save_csv:
        with (out_dir / "avalanche_stats.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", "row", "col", "drive_p", "size", "area", "dissipated", "parallel_rounds"])
            for i in range(result.config.steps):
                rounds = "" if result.parallel_rounds is None else int(result.parallel_rounds[i])
                writer.writerow([
                    i,
                    int(result.rows[i]),
                    int(result.cols[i]),
                    float(result.increments[i]),
                    int(result.avalanche_size[i]),
                    int(result.avalanche_area[i]),
                    int(result.dissipated[i]),
                    rounds,
                ])

    if result.config.output.save_npz:
        np.savez_compressed(
            out_dir / "simulation_arrays.npz",
            rows=result.rows,
            cols=result.cols,
            increments=result.increments,
            avalanche_size=result.avalanche_size,
            avalanche_area=result.avalanche_area,
            dissipated=result.dissipated,
            parallel_rounds=np.array([]) if result.parallel_rounds is None else result.parallel_rounds,
        )

    if result.config.output.save_lattice:
        np.save(out_dir / "final_lattice.npy", result.final_lattice)

    if result.config.output.make_plots:
        save_plots(result.avalanche_size[sl], out_dir, fit)
        _save_paper_area_plot(result.avalanche_area[sl], out_dir)

    return out_dir


def _save_paper_area_plot(areas: np.ndarray, out_dir: Path) -> None:
    """Compatibility plot matching the manuscript's apparent avalanche-area metric."""
    import matplotlib.pyplot as plt
    from .analysis import frequency_table, paper_style_ols

    values, counts = frequency_table(areas)
    fit = paper_style_ols(areas)
    if values.size == 0 or fit is None:
        return
    lx = np.log(values)
    lyhat = fit["slope"] * lx + fit["intercept"]
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.loglog(values, counts, marker="o", linestyle="none", markersize=3, label="Avalanche area frequency")
    ax.loglog(values, np.exp(lyhat), linestyle="--", label=f"OLS slope={fit['slope']:.4f}")
    ax.set_xlabel("Avalanche area (distinct toppled sites)")
    ax.set_ylabel("Frequency")
    ax.set_title("Paper-style log-log replication")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "paper_replication_avalanche_area.png", dpi=180)
    plt.close(fig)
