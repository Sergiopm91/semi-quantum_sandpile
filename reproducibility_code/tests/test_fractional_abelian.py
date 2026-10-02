import numpy as np
from semisandpile.config import DriveConfig
from semisandpile.drive import generate_drive
from semisandpile.engines import simulate_numba, simulate_numpy


def test_real_load_fixed_transfer_is_order_independent_numerically():
    n=12
    rows,cols,inc=generate_drive(n,400,314159,DriveConfig(mode='uniform_p'))
    initial=np.full((n,n),3.0)
    q=simulate_numba(initial,rows,cols,inc,epsilon=1e-12)
    s=simulate_numpy(initial,rows,cols,inc,epsilon=1e-12)
    assert np.array_equal(q.avalanche_size,s.avalanche_size)
    assert np.array_equal(q.avalanche_area,s.avalanche_area)
    assert np.allclose(q.dissipated,s.dissipated,atol=1e-10)
    assert np.allclose(q.final_lattice,s.final_lattice,atol=1e-10)
