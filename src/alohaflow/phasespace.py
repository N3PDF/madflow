"""
    PhaseSpace generation routines

    The output momenta has always the structure (E, px, py, pz)
    When a complete set of momenta is output the structure is (n_events, n_particles, n_dim)
"""

from .config import float_me, run_eager, DTYPE
run_eager(True)

import numpy as np
import tensorflow as tf

import logging

PI = float_me(np.pi)
logger = logging.getLogger(__name__)

# Helpers for rambo
tf.TensorSpec(shape=[None, 4], dtype=DTYPE),
def _gen_unconstrained_momenta(xrand):
    """
    Generates unconstrained 4-momenta

    Parameters
    ----------
        xrand: tensor(n_events, 4)
        random numbers to generate momenta

    Returns
    -------
        unconstrained momenta: tensor(n_events, 4)
    """
    costh = 2.0 * xrand[:, 0] - 1.0
    sinth = tf.sqrt(1.0 - costh ** 2)
    phi = 2 * PI * xrand[:, 1]
    energy = -1.0*tf.math.log(tf.reduce_prod(xrand[:, 2:4], axis=1))

    qx = energy * sinth * tf.math.sin(phi)
    qy = energy * sinth * tf.math.cos(phi)
    qz = energy * costh
    return tf.stack([energy, qx, qy, qz], axis=1)

def rambo(xrand, n_particles, sqrts, masses=None, check_physical=False):
    """
    Implementation of the RAMBO phase space generator in TensorFlow
    RAMBO: RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)
        a democratic multi-particle phase space generator
        based on Comput. Phys. Commun. 40 (1986) 359-373
    """
    if check_physical:
        if tf.executing_eagerly():
            if masses is not None and tf.reduce_sum(masses) > sqrts:
                raise ValueError(
                    f"Not enough energy ({sqrts}) to generate particles of mass: {masses}"
                )
        else:
            logger.warning("Graph-compiled functions assumes all imput is physical")

    sqrts = float_me(sqrts)
    if masses is not None:
        if isinstance(masses, list) and np.sum(masses) == 0:
            masses = None
        else:
            masses = float_me(masses)

    # Consume the random numbers into unconstrained momenta that will be later transformed
    all_q = [_gen_unconstrained_momenta(xrand[:, i*4:(i+1)*4]) for i in range(n_particles)]
    sum_q = tf.reduce_sum(all_q, axis=0) # (n_events, 4)
    sum_q2 = sum_q**2
    qmass = tf.sqrt(sum_q2[:,0:1] - tf.reduce_sum(sum_q2[:,1:], axis=1, keepdims=1))
    x = sqrts / qmass # (nevents, 1)
    bquad = -sum_q / qmass # (nevents, 4)
    bvec = bquad[:, 1:]
    gamma = -bquad[:, 0:1]
    a = 1.0 / (1.0 + gamma)

    # Perform the conformal transformation q -> p
    all_p = []
    for q in all_q:
        bq = tf.reduce_sum(q[:, 1:]*bvec, axis=1, keepdims=True)
        tmp = bq*a + q[:, 0:1]
        pvec = (q[:,1:] + bvec*tmp) # (n_events, 3)
        pnrg = q[:,0:1]*gamma + bq
        p = tf.concat([pnrg, pvec], axis=1)*x # (n_events, 4)
        all_p.append(p)
    all_p = tf.stack(all_p, axis=1) # (n_events, n_particles, 4)

    # Finally compute the weight for the phase space point
    wt = tf.math.log(PI/2.0)*(n_particles-1)
    wt -= 2.0*tf.math.lgamma(float_me(n_particles-1))
    wt -= tf.math.log(float_me(n_particles-1))
    wt += (2*n_particles - 4)*tf.math.log(sqrts)

    if masses is None:
        wt = tf.exp(wt)/tf.pow(2*PI, 3*n_particles-4)
        return all_p, tf.ones_like(x)*wt
