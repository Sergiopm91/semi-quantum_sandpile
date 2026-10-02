import numpy as np
from semisandpile.config import DriveConfig
from semisandpile.drive import generate_drive
from semisandpile.engines import simulate_numba_dirichlet


def test_dirichlet_redistribution_is_reproducible_and_mass_conserving():
    n=10
    rows,cols,inc=generate_drive(n,500,101,DriveConfig(mode='uniform_p'))
    initial=np.full((n,n),3.0)
    a=simulate_numba_dirichlet(initial,rows,cols,inc,1e-12,2026)
    b=simulate_numba_dirichlet(initial,rows,cols,inc,1e-12,2026)
    assert np.array_equal(a.avalanche_size,b.avalanche_size)
    assert np.array_equal(a.avalanche_area,b.avalanche_area)
    assert np.allclose(a.final_lattice,b.final_lattice,atol=1e-12)
    assert np.allclose(a.dissipated,b.dissipated,atol=1e-12)
    residual=initial.sum()+inc.sum()-a.dissipated.sum()-a.final_lattice.sum()
    assert abs(residual)<1e-8
    assert a.final_lattice.min()>=-1e-12
