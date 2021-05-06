""" Tests phase space routines
"""
from alohaflow.config import DTYPE
from alohaflow.phasespace import rambo
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
    all_p, w = rambo(xrand, n, sqrts, masses=None)
    np.testing.assert_equal(all_p.shape, (n_events, n, 4))
    vol = massless_volume(n, sqrts)
    np.testing.assert_allclose(vol, w, rtol=tol)


def test_rambo(sqrts=7e3, max_n=8):
    """ Check that rambo produces the right type of phase space """
    for n in range(2, max_n):
        auto_test_rambo_massless(n, sqrts)

    # Check that it also accepts a variable input energy
    events = 13
    variable_sqrts = tf.random.uniform((13,), dtype=DTYPE)*sqrts
    auto_test_rambo_massless(n, variable_sqrts, n_events=events)

if __name__=='__main__':
    from time import time as tm
    start = tm()
    test_rambo()
    print(f"Program done in {tm()-start} s")
