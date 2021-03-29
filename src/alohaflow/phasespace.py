"""
    PhaseSpace generation routines

    The output momenta has always the structure (E, px, py, pz)
    When a complete set of momenta is output the structure is (n_events, n_particles, n_dim)
"""

from .config import float_me, DTYPE, run_eager

import numpy as np
import tensorflow as tf

import logging

PI = float_me(np.pi)
ACC = float_me(1e-10)
logger = logging.getLogger(__name__)

# Helpers for rambo
events_signature = tf.TensorSpec(shape=[None, 1], dtype=DTYPE)
p_signature = tf.TensorSpec(shape=[None, 4], dtype=DTYPE)
ps_signature = tf.TensorSpec(shape=[None, None, 4], dtype=DTYPE)


@tf.function(
    input_signature=[
        events_signature,
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None, None], dtype=DTYPE),
    ]
)
def _massive_xfactor(sqrts, masses, massless_energies):
    """
    Takes as input the total energy of the system
    the masses of the external particles
    and the energies of the (massless) phase space
    computes the new energies and the necessary rescaling of the
    momenta

    This relies in an iterative process which is can't be naively parallelized
    over the number of events.
    Instead we loop until all points are below the desired accuracy or the maximum
    number of iterations is reached.
    The body of the loop in this case is simple enough that an impact in performance is
    not expected so no more complicated solutions are needed.

    Parameters
    ----------
        sqrts: float (nevents)
        masses: float (nparticles)
        massless_energies: (nevents, nparticles)

    Returns
    -------
        xfactor: float (nevents,)
        new_energies: float (nevents, nparticles)
    """
    total_mass = tf.reduce_sum(masses)
    e2 = tf.square(massless_energies)
    masses2 = tf.square(masses)

    @tf.function
    def while_body(_, xfactor, *arg):
        """Computation of the xfactor.
        As the computation converges for different events the xfactor
        stops being updated.
        A possible alternative is to continue updating the xfactor until
        _all_ events are under the threshold, but in practice this means
        different events would have had a different accuracy threshold
        """
        x2 = xfactor ** 2
        new_E = tf.sqrt(masses2 + e2 * x2)
        f0 = tf.reduce_sum(new_E, axis=1, keepdims=True) - sqrts
        g0 = tf.reduce_sum(e2 / new_E, axis=1, keepdims=True)
        next_xfactor = xfactor - f0 / (xfactor * g0)

        has_converged = f0 > ACC
        xfactor = tf.where(has_converged, next_xfactor, xfactor)
        return tf.reduce_all(has_converged), xfactor, new_E

    xfactor = tf.sqrt(1 - (total_mass / sqrts) ** 2)
    new_E = massless_energies

    _, xfactor, new_E = tf.while_loop(
        lambda x, *args: x,
        while_body,
        (True, xfactor, new_E),
        parallel_iterations=1,  # iterations are consecutive!
        maximum_iterations=10,
    )

    return xfactor, new_E


@tf.function(input_signature=[p_signature, p_signature])
def _conformal_transformation(input_q, bquad):
    """ Perform the conformal transformation q->p """
    bvec = bquad[:, 1:]
    gamma = -bquad[:, 0:1]
    a = 1.0 / (1.0 + gamma)
    bq = tf.reduce_sum(input_q[:, 1:] * bvec, axis=1, keepdims=True)
    tmp = bq * a + input_q[:, 0:1]
    pvec = input_q[:, 1:] + bvec * tmp  # (n_events, 3)
    pnrg = input_q[:, 0:1] * gamma + bq
    return tf.concat([pnrg, pvec], axis=1)  # (n_events, 4)


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
    energy = -1.0 * tf.math.log(tf.reduce_prod(xrand[:, 2:4], axis=1))

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
            if masses is not None and tf.reduce_sum(masses) > sqrts and isinstance(sqrts, float):
                raise ValueError(
                    f"Not enough energy ({sqrts}) to generate particles of mass: {masses}"
                )
        else:
            logger.warning("Graph-compiled functions assumes all imput is physical")

    if isinstance(sqrts, float):
        sqrts = float_me(sqrts) * tf.ones_like(xrand[:, 0])
    sqrts = tf.reshape(sqrts, (-1, 1))

    if masses is not None:
        if isinstance(masses, list) and np.sum(masses) == 0:
            masses = None
        else:
            masses = float_me(masses)

    # Consume the random numbers into unconstrained momenta that will be later transformed
    all_q = [_gen_unconstrained_momenta(xrand[:, i * 4 : (i + 1) * 4]) for i in range(n_particles)]
    sum_q = tf.reduce_sum(all_q, axis=0)  # (n_events, 4)
    sum_q2 = sum_q ** 2
    qmass = tf.sqrt(sum_q2[:, 0:1] - tf.reduce_sum(sum_q2[:, 1:], axis=1, keepdims=True))
    x = sqrts / qmass  # (nevents, 1)
    bquad = -sum_q / qmass  # (nevents, 4)

    # Perform the conformal transformation q -> p
    tmp_p = tf.stack(
        [_conformal_transformation(q, bquad) for q in all_q], axis=1
    )  # (n_events, n_particles, 4)
    all_p = tmp_p * tf.expand_dims(x, axis=-1)

    # Finally compute the weight for the phase space point
    wt = tf.math.log(PI / 2.0) * (n_particles - 1)
    wt -= 2.0 * tf.math.lgamma(float_me(n_particles - 1))
    wt -= tf.math.log(float_me(n_particles - 1))
    wt += (2 * n_particles - 4) * tf.math.log(sqrts[:, 0])

    if masses is None:
        wt = tf.exp(wt) / tf.pow(2 * PI, 3 * n_particles - 4)
        return all_p, wt

    # If dealing with massive particles, momenta needs to be rescaled
    xfactor, new_E = _massive_xfactor(sqrts, masses, all_p[:, :, 0])

    # Rescale the momenta
    wt2 = 1.0
    wt3 = 0.0
    v = []

    rescaled_pvec = all_p[:, :, 1:] * tf.expand_dims(xfactor, axis=-1)
    massive_p = tf.concat([tf.expand_dims(new_E, axis=-1), rescaled_pvec], axis=-1)

    # and weights
    v = all_p[:, :, 0] * xfactor
    wt2 = tf.reduce_prod(v / new_E, axis=1)
    wt3 = tf.reduce_sum(v ** 2 / new_E, axis=1)
    wt += (2 * n_particles - 3) * tf.math.log(xfactor[:, 0]) + tf.math.log(wt2 / wt3 * sqrts[:, 0])
    wt = tf.exp(wt) / tf.pow(2 * PI, 3 * n_particles - 4)
    return massive_p, wt
