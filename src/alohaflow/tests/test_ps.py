""" Tests phase space routines
"""
from alohaflow.config import DTYPE
import alohaflow.phasespace as ps
import numpy as np
import tensorflow as tf


def massless_volume(n, e):
    """Volume of massless phase space
    1 / 2 / (4*pi)^(2n-3) * E^(2n-4) / G(n)G(n-1)
    """
    gn = np.math.factorial(n - 1)
    gnm1 = np.math.factorial(n - 2)
    energy = pow(e, 2 * n - 4) / 2.0
    return energy / gn / gnm1 / pow(4 * np.pi, 2 * n - 3)


def auto_test_rambo_massless(n, sqrts, n_events=3, tol=1e-6):
    """Check that a massless phase space point
    produces the right weight and has the right shape
    """
    n_rand_dim = n * 4
    xrand = tf.random.uniform((n_events, n_rand_dim), dtype=DTYPE)
    all_p, w = ps.rambo(xrand, n, sqrts, masses=None)
    np.testing.assert_equal(all_p.shape, (n_events, n, 4))
    vol = massless_volume(n, sqrts)
    np.testing.assert_allclose(vol, w, rtol=tol)


def test_rambo(sqrts=7e3, max_n=8):
    """Check that rambo produces the right type of phase space"""
    for n in range(2, max_n):
        auto_test_rambo_massless(n, sqrts)

    # Check that it also accepts a variable input energy
    events = 13
    variable_sqrts = tf.random.uniform((13,), dtype=DTYPE) * sqrts
    auto_test_rambo_massless(n, variable_sqrts, n_events=events)


def test_PhaseSpaceGenerator(sqrts=7e3, nparticles=5, nevents=10000):
    """Check that the phase space generator and cuts work
    This is explicitly testing:

    1. The phase space is generating points
    2. The phase space can compute the PT
    3. The register_cut and clear_cuts methods of the phase space are doing something
    4. The cuts done with numpy and done with ps_gen are equivalent
    """
    ps_gen = ps.PhaseSpaceGenerator(nparticles, sqrts, algorithm="ramboflow")
    ps_gen.register_cut("pt", particle=3, min_val=60, max_val=300.0)
    dim = (nparticles - 2) * 4 + 2
    xrand = tf.random.uniform((nevents, dim), dtype=DTYPE)
    all_ps, w, x1, x2, idx = ps_gen(xrand)
    ps_gen.clear_cuts()
    full_ps, full_w, fx1, fx2, _ = ps_gen(xrand)
    # Check the original pt
    internal_pt = ps_gen.pt(full_ps[:, 3, :])
    full_np = full_ps.numpy()
    numpy_pt = np.sqrt(full_np[:, 3, 1] ** 2 + full_np[:, 3, 2] ** 2)
    np.testing.assert_allclose(numpy_pt, internal_pt)
    # Check that after the cuts we get the right ones
    mask = np.all([numpy_pt > 60.0, numpy_pt < 300.0], axis=0)
    np.testing.assert_allclose(all_ps, full_np[mask])


def test_fourmomenta(sqrts=7e3, nparticles=4, nevents=100, masses=[50.0, 125.0]):
    """Generate a few phase space points and compute some quantities"""
    ps_gen = ps.PhaseSpaceGenerator(nparticles, sqrts, masses=masses, algorithm="ramboflow")
    dim = (nparticles - 2) * 4 + 2
    xrand = tf.random.uniform((nevents, dim), dtype=DTYPE)
    all_ps, w, x1, x2, idx = ps_gen(xrand)
    # The initial particles should have mass == 0.0
    np.testing.assert_allclose(ps._invariant_mass(all_ps[:, 0, :]), 0.0)
    np.testing.assert_allclose(ps._invariant_mass(all_ps[:, 1, :]), 0.0)
    # And the others whatever is given by the mass
    np.testing.assert_allclose(ps._invariant_mass(all_ps[:, 2, :]), masses[0] ** 2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(ps._invariant_mass(all_ps[:, 3, :]), masses[1] ** 2, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    from time import time as tm

    start = tm()
    test_fourmomenta()
    print(f"Program done in {tm()-start} s")
