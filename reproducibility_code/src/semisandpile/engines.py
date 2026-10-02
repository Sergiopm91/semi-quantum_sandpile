from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class EngineOutput:
    final_lattice: np.ndarray
    avalanche_size: np.ndarray
    avalanche_area: np.ndarray
    dissipated: np.ndarray
    parallel_rounds: np.ndarray | None = None


@njit(cache=True)
def _simulate_numba_core(
    lattice: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    increments: np.ndarray,
    epsilon: float,
):
    """FIFO event-queue relaxation for fixed one-unit neighbor transfers."""
    n = lattice.shape[0]
    nsteps = rows.shape[0]
    size_out = np.zeros(nsteps, dtype=np.int64)
    area_out = np.zeros(nsteps, dtype=np.int32)
    diss_out = np.zeros(nsteps, dtype=np.float64)

    capacity = n * n
    queue = np.empty(capacity, dtype=np.int32)
    queued = np.zeros(capacity, dtype=np.uint8)
    visited_stamp = np.zeros(capacity, dtype=np.int64)

    for step in range(nsteps):
        x = rows[step]
        y = cols[step]
        lattice[x, y] += increments[step]

        head = 0
        tail = 0
        qcount = 0
        site0 = x * n + y
        # Only the driven site can become newly unstable immediately after the drive.
        if lattice[x, y] + epsilon >= 4.0:
            queue[tail] = site0
            tail = (tail + 1) % capacity
            qcount += 1
            queued[site0] = 1

        topple_count = 0
        area_count = 0
        dissipated = 0.0
        # Python/NumPy int64 overflow is practically unreachable in the reported runs.
        stamp = step + 1

        while qcount > 0:
            site = queue[head]
            head = (head + 1) % capacity
            qcount -= 1
            queued[site] = 0
            i = site // n
            j = site - i * n

            k = int(np.floor((lattice[i, j] + epsilon) / 4.0))
            if k <= 0:
                continue

            lattice[i, j] -= 4.0 * k
            if lattice[i, j] < 0.0 and lattice[i, j] > -epsilon:
                lattice[i, j] = 0.0

            topple_count += k
            if visited_stamp[site] != stamp:
                visited_stamp[site] = stamp
                area_count += 1

            if i > 0:
                ni, nj = i - 1, j
                lattice[ni, nj] += k
                ns = ni * n + nj
                if lattice[ni, nj] + epsilon >= 4.0 and queued[ns] == 0:
                    queue[tail] = ns; tail = (tail + 1) % capacity; qcount += 1; queued[ns] = 1
            else:
                dissipated += k

            if i + 1 < n:
                ni, nj = i + 1, j
                lattice[ni, nj] += k
                ns = ni * n + nj
                if lattice[ni, nj] + epsilon >= 4.0 and queued[ns] == 0:
                    queue[tail] = ns; tail = (tail + 1) % capacity; qcount += 1; queued[ns] = 1
            else:
                dissipated += k

            if j > 0:
                ni, nj = i, j - 1
                lattice[ni, nj] += k
                ns = ni * n + nj
                if lattice[ni, nj] + epsilon >= 4.0 and queued[ns] == 0:
                    queue[tail] = ns; tail = (tail + 1) % capacity; qcount += 1; queued[ns] = 1
            else:
                dissipated += k

            if j + 1 < n:
                ni, nj = i, j + 1
                lattice[ni, nj] += k
                ns = ni * n + nj
                if lattice[ni, nj] + epsilon >= 4.0 and queued[ns] == 0:
                    queue[tail] = ns; tail = (tail + 1) % capacity; qcount += 1; queued[ns] = 1
            else:
                dissipated += k

        size_out[step] = topple_count
        area_out[step] = area_count
        diss_out[step] = dissipated

    return lattice, size_out, area_out, diss_out


# A small, explicit PRNG for redistribution weights.  The external drive itself uses
# NumPy Generator(PCG64); this SplitMix64 stream is separate so that random toppling
# weights do not depend on NumPy/Numba RNG implementation details.
@njit(cache=True)
def _splitmix64_next(state):
    state = np.uint64(state + np.uint64(0x9E3779B97F4A7C15))
    z = state
    z = np.uint64((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
    z = np.uint64((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
    z = z ^ (z >> np.uint64(31))
    # Strictly inside (0,1), using the upper 53 bits.
    u = (float(z >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)
    return state, u


@njit(cache=True)
def _simulate_numba_dirichlet_core(
    lattice: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    increments: np.ndarray,
    epsilon: float,
    redistribution_seed: int,
):
    """FIFO relaxation with spatially delocalized outgoing load.

    Each individual toppling redistributes four units as
        w = 4 * Dirichlet(1,1,1,1),  sum(w)=4.
    This is a stochastic continuous-load sandpile, not a unitary quantum map.
    """
    n = lattice.shape[0]
    nsteps = rows.shape[0]
    size_out = np.zeros(nsteps, dtype=np.int64)
    area_out = np.zeros(nsteps, dtype=np.int32)
    diss_out = np.zeros(nsteps, dtype=np.float64)

    capacity = n * n
    queue = np.empty(capacity, dtype=np.int32)
    queued = np.zeros(capacity, dtype=np.uint8)
    visited_stamp = np.zeros(capacity, dtype=np.int64)
    rng_state = np.uint64(redistribution_seed) ^ np.uint64(0xD1B54A32D192ED03)

    for step in range(nsteps):
        x = rows[step]
        y = cols[step]
        lattice[x, y] += increments[step]

        head = 0
        tail = 0
        qcount = 0
        site0 = x * n + y
        if lattice[x, y] + epsilon >= 4.0:
            queue[tail] = site0
            tail = (tail + 1) % capacity
            qcount += 1
            queued[site0] = 1

        topple_count = 0
        area_count = 0
        dissipated = 0.0
        stamp = step + 1

        while qcount > 0:
            site = queue[head]
            head = (head + 1) % capacity
            qcount -= 1
            queued[site] = 0
            i = site // n
            j = site - i * n

            k = int(np.floor((lattice[i, j] + epsilon) / 4.0))
            if k <= 0:
                continue

            lattice[i, j] -= 4.0 * k
            if lattice[i, j] < 0.0 and lattice[i, j] > -epsilon:
                lattice[i, j] = 0.0

            topple_count += k
            if visited_stamp[site] != stamp:
                visited_stamp[site] = stamp
                area_count += 1

            # Each toppling receives its own normalized spatial wave weights.
            for _ in range(k):
                rng_state, u0 = _splitmix64_next(rng_state)
                rng_state, u1 = _splitmix64_next(rng_state)
                rng_state, u2 = _splitmix64_next(rng_state)
                rng_state, u3 = _splitmix64_next(rng_state)
                e0 = -np.log(u0); e1 = -np.log(u1); e2 = -np.log(u2); e3 = -np.log(u3)
                den = e0 + e1 + e2 + e3
                w0 = 4.0 * e0 / den
                w1 = 4.0 * e1 / den
                w2 = 4.0 * e2 / den
                w3 = 4.0 * e3 / den

                if i > 0:
                    ni, nj = i - 1, j
                    lattice[ni, nj] += w0
                    ns = ni * n + nj
                    if lattice[ni, nj] + epsilon >= 4.0 and queued[ns] == 0:
                        queue[tail] = ns; tail = (tail + 1) % capacity; qcount += 1; queued[ns] = 1
                else:
                    dissipated += w0

                if i + 1 < n:
                    ni, nj = i + 1, j
                    lattice[ni, nj] += w1
                    ns = ni * n + nj
                    if lattice[ni, nj] + epsilon >= 4.0 and queued[ns] == 0:
                        queue[tail] = ns; tail = (tail + 1) % capacity; qcount += 1; queued[ns] = 1
                else:
                    dissipated += w1

                if j > 0:
                    ni, nj = i, j - 1
                    lattice[ni, nj] += w2
                    ns = ni * n + nj
                    if lattice[ni, nj] + epsilon >= 4.0 and queued[ns] == 0:
                        queue[tail] = ns; tail = (tail + 1) % capacity; qcount += 1; queued[ns] = 1
                else:
                    dissipated += w2

                if j + 1 < n:
                    ni, nj = i, j + 1
                    lattice[ni, nj] += w3
                    ns = ni * n + nj
                    if lattice[ni, nj] + epsilon >= 4.0 and queued[ns] == 0:
                        queue[tail] = ns; tail = (tail + 1) % capacity; qcount += 1; queued[ns] = 1
                else:
                    dissipated += w3

        size_out[step] = topple_count
        area_out[step] = area_count
        diss_out[step] = dissipated

    return lattice, size_out, area_out, diss_out


def simulate_numba(
    initial_lattice: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    increments: np.ndarray,
    epsilon: float = 1e-12,
) -> EngineOutput:
    lattice = np.asarray(initial_lattice, dtype=np.float64).copy()
    lattice, sizes, areas, diss = _simulate_numba_core(
        lattice,
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(increments, dtype=np.float64),
        float(epsilon),
    )
    return EngineOutput(lattice, sizes, areas, diss, None)


def simulate_numba_dirichlet(
    initial_lattice: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    increments: np.ndarray,
    epsilon: float = 1e-12,
    redistribution_seed: int = 1,
) -> EngineOutput:
    lattice = np.asarray(initial_lattice, dtype=np.float64).copy()
    lattice, sizes, areas, diss = _simulate_numba_dirichlet_core(
        lattice,
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(increments, dtype=np.float64),
        float(epsilon),
        int(redistribution_seed),
    )
    return EngineOutput(lattice, sizes, areas, diss, None)


def _relax_parallel_numpy(lattice: np.ndarray, epsilon: float):
    n = lattice.shape[0]
    total_topplings = 0
    dissipated = 0.0
    rounds = 0
    toppled_any = np.zeros_like(lattice, dtype=bool)

    while True:
        k = np.floor((lattice + epsilon) / 4.0).astype(np.int64)
        if not np.any(k):
            break
        rounds += 1
        toppled_any |= k > 0
        total_topplings += int(k.sum())
        dissipated += float(k[0, :].sum() + k[-1, :].sum() + k[:, 0].sum() + k[:, -1].sum())

        lattice -= 4.0 * k
        lattice[:-1, :] += k[1:, :]
        lattice[1:, :] += k[:-1, :]
        lattice[:, :-1] += k[:, 1:]
        lattice[:, 1:] += k[:, :-1]
        tiny_negative = (lattice < 0.0) & (lattice > -epsilon)
        lattice[tiny_negative] = 0.0

    return total_topplings, int(toppled_any.sum()), dissipated, rounds


def simulate_numpy(
    initial_lattice: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    increments: np.ndarray,
    epsilon: float = 1e-12,
) -> EngineOutput:
    lattice = np.asarray(initial_lattice, dtype=np.float64).copy()
    nsteps = len(rows)
    sizes = np.zeros(nsteps, dtype=np.int64)
    areas = np.zeros(nsteps, dtype=np.int32)
    diss = np.zeros(nsteps, dtype=np.float64)
    rounds = np.zeros(nsteps, dtype=np.int32)
    for step in range(nsteps):
        lattice[rows[step], cols[step]] += increments[step]
        s, a, d, r = _relax_parallel_numpy(lattice, epsilon)
        sizes[step] = s
        areas[step] = a
        diss[step] = d
        rounds[step] = r
    return EngineOutput(lattice, sizes, areas, diss, rounds)


def simulate_cupy(
    initial_lattice: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    increments: np.ndarray,
    epsilon: float = 1e-12,
) -> EngineOutput:
    """CuPy synchronous relaxation backend for fixed BTW redistribution."""
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "CuPy is not installed. Install the wheel matching your CUDA runtime, "
            "for example: pip install cupy-cuda12x"
        ) from exc

    lattice = cp.asarray(initial_lattice, dtype=cp.float64)
    rows_cp = cp.asarray(rows, dtype=cp.int32)
    cols_cp = cp.asarray(cols, dtype=cp.int32)
    inc_cp = cp.asarray(increments, dtype=cp.float64)
    nsteps = len(rows)
    sizes = np.zeros(nsteps, dtype=np.int64)
    areas = np.zeros(nsteps, dtype=np.int32)
    diss = np.zeros(nsteps, dtype=np.float64)
    rounds = np.zeros(nsteps, dtype=np.int32)

    for step in range(nsteps):
        lattice[rows_cp[step], cols_cp[step]] += inc_cp[step]
        toppled_any = cp.zeros(lattice.shape, dtype=cp.bool_)
        total_topplings = 0
        total_diss = 0.0
        r = 0
        while True:
            k = cp.floor((lattice + epsilon) / 4.0).astype(cp.int64)
            ksum = int(cp.sum(k).item())
            if ksum == 0:
                break
            r += 1
            total_topplings += ksum
            toppled_any |= k > 0
            total_diss += float(
                (cp.sum(k[0, :]) + cp.sum(k[-1, :]) + cp.sum(k[:, 0]) + cp.sum(k[:, -1])).item()
            )
            lattice -= 4.0 * k
            lattice[:-1, :] += k[1:, :]
            lattice[1:, :] += k[:-1, :]
            lattice[:, :-1] += k[:, 1:]
            lattice[:, 1:] += k[:, :-1]
            lattice = cp.where((lattice < 0.0) & (lattice > -epsilon), 0.0, lattice)
        sizes[step] = total_topplings
        areas[step] = int(cp.sum(topped_any).item())
        diss[step] = total_diss
        rounds[step] = r

    return EngineOutput(cp.asnumpy(lattice), sizes, areas, diss, rounds)
