# Semi-Quantum Sandpile — reproducibility package (R3)

This package accompanies **“A Semi-Quantum Approach to the Classical BTW Sandpile Model: Formalization, Reproducible Implementation, and Numerical Validation”**.

## Implemented models

The code separates the normalized incoming state

\[
|\psi_q\rangle=\alpha_q|0\rangle+\beta_q|1\rangle,
\qquad p_q=|\beta_q|^2,
\]

from the accumulated real site load `G`. A site is numerically unstable when `G + epsilon >= 4`, with `epsilon = 1e-12` in the paper experiments.

Drive modes:

- `classical`: `p=1`;
- `fixed_beta`: fixed `p=|beta|^2`;
- `uniform_p`: `p ~ U(p_low,p_high)`;
- `beta_p`: beta-distributed `p`;
- `bernoulli_p`: binary `p in {0,1}`.

Redistribution modes:

- `discrete`: subtract four and transfer one unit to each cardinal neighbor;
- `dirichlet`: for every **individual** toppling sample conservative weights `w ~ 4 Dirichlet(1,1,1,1)` and transfer `w_m` to the four directions, with out-of-domain weights dissipated.

The fixed-transfer model is Abelian/order independent. The realized Dirichlet process is **not pathwise Abelian** because random weights are consumed in an order-dependent sequence; therefore FIFO order, batched multiplicity, and one independent Dirichlet draw per individual toppling are part of the model definition.

## Reproducibility details

- external drive RNG: NumPy `Generator(PCG64(seed))`;
- spatial redistribution RNG: explicitly seeded SplitMix64 stream in the Numba kernel;
- CPU production queue: FIFO;
- boundary: open/dissipative;
- floating threshold tolerance: `1e-12`;
- small residuals in `(-epsilon, 0)` are clamped to zero;
- fractional mass-balance residuals are evaluated **after** this clamp;
- fixed-transfer stabilization is order independent because its real-load toppling operators are additive commuting translations.

## Main `L=100` experiments

Each principal scenario uses 100,000 burn-in additions followed by 1,000,000 retained additions:

- `configs/paper_classical_stationary.yaml`
- `configs/paper_uniform_fixed.yaml`
- `configs/paper_bernoulli_fixed.yaml`
- `configs/paper_uniform_dirichlet.yaml`

The measured R3 values are:

| Scenario | Nonzero avalanches | Mean nonzero size | Final mean load | Effective slope, 10<=s<=1e4 | Topplings / unit added load |
|---|---:|---:|---:|---:|---:|
| Classical `p=1`, fixed | 439,671 | 830.8 | 2.1144 | 1.142 +/- 0.016 | 365.289 |
| `p~U(0,1)`, fixed | 219,442 | 831.8 | 2.6008 | 1.143 +/- 0.017 | 365.369 |
| Bernoulli(1/2), fixed | 219,745 | 831.7 | 2.1180 | 1.141 +/- 0.016 | 365.256 |
| `p~U(0,1)`, Dirichlet | 211,214 | 864.1 | 2.5663 | 1.284 +/- 0.006 | 365.329 |

The manuscript rounds non-fit quantities more aggressively; the values above are retained here for reproducibility.

## Exact finite-lattice first moment

For the open-boundary classical BTW model under a uniformly placed unit drive,

\[
\langle s\rangle_L=\frac{1}{L^2}\mathbf 1^T\Delta_L^{-1}\mathbf 1.
\]

`semisandpile.analysis.exact_mean_topplings_uniform_drive(L)` evaluates this finite-graph expectation by a sparse solve. The paper benchmarks are:

| L | Exact mean topplings / drive |
|---:|---:|
| 32 | 40.5803183602 |
| 64 | 153.0431002280 |
| 100 | 365.5959945136 |
| 128 | 593.8932453546 |
| 256 | 2339.2977889006 |

The production `L=100` simulation value 365.289 differs from the exact expectation by 0.084%.

## Finite-size assessment and stationarity check

`../scripts/run_fss.py` reproduces the three-seed `L={32,64,128,256}` distributional assessment using the historical fixed protocol of 50,000 burn-in + 100,000 retained additions per seed.

Because this burn-in is only 0.76 additions/site at `L=256`, R3 adds an explicit mean-height stationarity diagnostic. `../scripts/run_stationarity_diagnostic.py` runs 150,000 additions for three seeds at every `L` and samples `mean(z)` every 1,000 additions. The post-burn three-seed mean has no sustained drift on the plotted scale; at `L=256`, it is approximately 2.124 at 50,000 additions and 2.120 at 150,000 additions. This supports the chosen retained window but is not claimed as a proof of mixing.

The four simulated finite-size means have a log-log slope 1.9530; the exact finite-size values over the same `L` range have slope 1.95037.

## Height validation

For the classical `L=100` run, the final height probabilities are `(0.0769, 0.1728, 0.3093, 0.4410)` with mean height `2.1144`, close to the exact/infinite-lattice benchmarks discussed in the manuscript.

## Tests

R3 has **8 passing tests** covering:

1. exact classical limit;
2. Numba/NumPy deterministic engine equivalence;
3. mass balance with open boundaries;
4. seeded drive reproducibility;
5. fractional threshold accumulation;
6. real-load fixed-transfer Abelian/order equivalence;
7. stochastic Dirichlet reproducibility and mass conservation;
8. exact finite-graph first-moment values on small lattices.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
pytest -q
semisandpile run --config configs/paper_classical_stationary.yaml
```

To rebuild the paper validation products from the package root:

```bash
python scripts/run_main_experiments.py
python scripts/run_fss.py --L 32 64 128 256
python scripts/run_stationarity_diagnostic.py
python scripts/exact_btw_first_moment.py
python scripts/make_figures.py
```

## GPU status

CuPy remains an optional synchronous backend, but the manuscript does not claim a GPU benchmark because CUDA hardware was not available in the validation environment.
