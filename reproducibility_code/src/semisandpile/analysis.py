from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import zeta


@dataclass
class PowerLawFit:
    alpha: float
    xmin: int
    ks: float
    n_tail: int

    def to_dict(self):
        return {"alpha": self.alpha, "xmin": self.xmin, "ks": self.ks, "n_tail": self.n_tail}


def frequency_table(sizes: np.ndarray):
    x = np.asarray(sizes, dtype=np.int64)
    x = x[x > 0]
    if x.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.unique(x, return_counts=True)


def paper_style_ols(metric: np.ndarray):
    """Legacy raw-frequency log-log regression, retained only for draft replication."""
    values, counts = frequency_table(metric)
    if values.size < 2:
        return None
    lx = np.log(values.astype(float)); ly = np.log(counts.astype(float))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2)); ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = lx.size
    se = float(np.sqrt((ss_res/(n-2))/np.sum((lx-lx.mean())**2))) if n > 2 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "slope_se": se, "n_points": int(n)}


def log_binned_pdf_fit(metric: np.ndarray, lower: int = 10, upper: int = 10_000, bins: int = 24):
    """Logarithmically binned PDF and descriptive scaling-window slope.

    Returns the PDF density (count / [N * bin width]) and a least-squares slope
    on nonempty bins whose geometric centers lie in [lower, upper].  The fitted
    tau_eff is -slope; it is an effective finite-size exponent, not a universal
    exponent estimate.
    """
    x = np.asarray(metric, dtype=np.int64)
    x = x[x > 0]
    if x.size == 0:
        return None
    lo = max(1, lower)
    hi = min(int(x.max()) + 1, upper + 1)
    if hi <= lo:
        return None
    edges = np.geomspace(lo, hi, bins + 1)
    edges = np.unique(np.maximum.accumulate(np.floor(edges).astype(int)))
    if edges[0] > lo:
        edges = np.insert(edges, 0, lo)
    if edges[-1] <= upper:
        edges = np.append(edges, upper + 1)
    edges = np.unique(edges)
    counts, e = np.histogram(x, bins=edges)
    widths = np.diff(e).astype(float)
    centers = np.sqrt(e[:-1].astype(float) * e[1:].astype(float))
    density = counts / (x.size * widths)
    mask = (counts > 0) & (centers >= lower) & (centers <= upper)
    if mask.sum() < 3:
        return None
    lx = np.log(centers[mask]); ly = np.log(density[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope*lx + intercept
    resid = ly-pred
    ss_res = float(np.sum(resid**2)); ss_tot = float(np.sum((ly-ly.mean())**2))
    r2 = 1-ss_res/ss_tot if ss_tot>0 else float('nan')
    se = float(np.sqrt((ss_res/(len(lx)-2))/np.sum((lx-lx.mean())**2))) if len(lx)>2 else float('nan')
    return {
        "tau_eff": float(-slope), "slope": float(slope), "slope_se": se,
        "tau_se": se, "intercept": float(intercept), "r2": r2,
        "lower": int(lower), "upper": int(upper), "n_events": int(x.size),
        "centers": centers.tolist(), "density": density.tolist(), "counts": counts.tolist(),
        "fit_mask": mask.tolist(), "edges": e.tolist(),
    }


def height_probabilities(lattice: np.ndarray):
    """Integer-height probabilities for a classical stable lattice."""
    z = np.rint(np.asarray(lattice)).astype(int)
    if not np.allclose(lattice, z, atol=1e-10):
        raise ValueError("height_probabilities requires an integer-valued classical lattice")
    counts = np.bincount(z.ravel(), minlength=4)[:4]
    probs = counts / counts.sum()
    return {str(i): float(probs[i]) for i in range(4)}


def _fit_alpha_discrete_exact(x: np.ndarray, xmin: int) -> float:
    tail = x[x >= xmin].astype(np.float64)
    sum_log = float(np.log(tail).sum())
    n = tail.size
    def nll(alpha):
        if alpha <= 1.0:
            return np.inf
        return alpha * sum_log + n * np.log(zeta(alpha, xmin))
    opt = minimize_scalar(nll, bounds=(1.0001, 8.0), method="bounded")
    if not opt.success:
        raise RuntimeError("Power-law MLE optimization failed")
    return float(opt.x)


def _ks_discrete(x: np.ndarray, xmin: int, alpha: float) -> float:
    tail = np.sort(x[x >= xmin].astype(np.int64))
    n = tail.size
    values, counts = np.unique(tail, return_counts=True)
    empirical_cdf = np.cumsum(counts) / n
    model_cdf = 1.0 - zeta(alpha, values + 1) / zeta(alpha, xmin)
    return float(np.max(np.abs(empirical_cdf - model_cdf)))


def fit_powerlaw_discrete(sizes: np.ndarray, xmin: int | None = None,
                          min_tail: int = 100, max_candidates: int = 80):
    """Legacy pure power-law tail diagnostic retained for reproducibility only."""
    x = np.asarray(sizes, dtype=np.int64); x = x[x > 0]
    if x.size < min_tail:
        return None
    if xmin is not None:
        if np.sum(x >= xmin) < min_tail:
            return None
        a = _fit_alpha_discrete_exact(x, int(xmin)); ks = _ks_discrete(x, int(xmin), a)
        return PowerLawFit(a, int(xmin), ks, int(np.sum(x >= xmin)))
    unique = np.unique(x)
    valid = np.array([u for u in unique if np.sum(x >= u) >= min_tail], dtype=np.int64)
    if valid.size == 0:
        return None
    if valid.size > max_candidates:
        valid = valid[np.unique(np.linspace(0, valid.size-1, max_candidates).astype(int))]
    best = None
    for cand in valid:
        a = _fit_alpha_discrete_exact(x, int(cand)); ks = _ks_discrete(x, int(cand), a)
        f = PowerLawFit(a, int(cand), ks, int(np.sum(x >= cand)))
        if best is None or f.ks < best.ks:
            best = f
    return best



def exact_mean_topplings_uniform_drive(size: int) -> float:
    """Exact finite-lattice BTW mean topplings per uniformly placed unit drive.

    On an LxL square with open boundaries, let Delta be the reduced toppling
    matrix (4 on the diagonal and -1 for in-domain nearest neighbours).
    Dhar's Green-function relation gives the expected odometer for a drive at
    site x as Delta^{-1} e_x. Averaging the total odometer over uniformly
    selected x yields

        <s> = (1/L^2) 1^T Delta^{-1} 1.

    The sparse linear solve evaluates this exact finite-graph expectation to
    floating-point precision; it is not a Monte Carlo estimate.
    """
    from scipy.sparse import diags, eye, kron
    from scipy.sparse.linalg import spsolve

    L = int(size)
    if L < 1:
        raise ValueError("size must be positive")
    off = np.ones(max(L - 1, 0), dtype=float)
    adjacency_1d = diags([off, off], offsets=[-1, 1], shape=(L, L), format="csr")
    I = eye(L, format="csr")
    delta = 4.0 * eye(L * L, format="csr") - kron(I, adjacency_1d) - kron(adjacency_1d, I)
    u = spsolve(delta, np.ones(L * L, dtype=float))
    return float(np.mean(u))

def save_plots(sizes: np.ndarray, out_dir: str | Path, fit: PowerLawFit | None = None) -> None:
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    values, counts = frequency_table(sizes)
    if values.size == 0: return
    fig, ax = plt.subplots(figsize=(7,5)); ax.loglog(values, counts, 'o', ms=3)
    ax.set_xlabel("Avalanche size (total topplings)"); ax.set_ylabel("Frequency"); fig.tight_layout()
    fig.savefig(out_dir/"avalanche_frequency.png", dpi=180); plt.close(fig)


def save_summary(summary: dict, fit: PowerLawFit | None, out_dir: str | Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(summary); payload["powerlaw_fit"] = None if fit is None else fit.to_dict()
    with (out_dir/"summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
