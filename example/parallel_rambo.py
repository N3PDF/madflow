"""
    Extension of rambo to generate many N-particle phase space in one go.

    This is the first non-fortran version where the source code is not shouting at you at any point
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from vegasflow import float_me, run_eager
import numpy as np
import tensorflow as tf

PI = float_me(np.pi)


def gen_unconstrained_momenta(xrand):
    """
    Generates unconstrained 4-momenta
    """
    costh = 2.0 * xrand[:, 0] - 1.0
    sinth = tf.sqrt(1.0 - costh ** 2)
    phi = 2 * PI * xrand[:, 1]
    energy = -tf.math.log(tf.reduce_prod(xrand[:, 2:4], axis=1))

    qx = energy * sinth * tf.math.sin(phi)
    qy = energy * sinth * tf.math.cos(phi)
    qz = energy * costh
    return tf.stack([qx, qy, qz, energy], axis=1)


ACC = float_me(1e-8)

@tf.function
def massive_xfactor(sqrts, masses, massless_energies):
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
        new_E = tf.sqrt(masses2 + mul(e2, x2))
        f0 = tf.reduce_sum(new_E, axis=1) - sqrts
        g0 = tf.reduce_sum(e2 / new_E, axis=1)
        next_xfactor = xfactor - f0 / (xfactor * g0)

        has_converged = f0 > ACC
        xfactor = tf.where(has_converged, next_xfactor, xfactor)
        return tf.reduce_all(has_converged), xfactor, new_E

    xfactor = tf.sqrt(1 - (total_mass / sqrts) ** 2)*tf.ones_like(e2[:,0])
    f0 = 2.0 * ACC
    new_E = massless_energies

    _, xfactor, new_E = tf.while_loop(
        lambda x, *args: x,
        while_body,
        (True, xfactor, new_E),
        parallel_iterations=1,  # iterations are consecutive!
        maximum_iterations=10,
    )

    return xfactor, new_E

@tf.function
def mul(a, b):
    """Multiply tensors a and b where rank(a) > 1, rank(b) = 1
    and the first dimension has a size equal to the only dimension of b
    """
    tmp = tf.transpose(a) * b
    return tf.transpose(tmp)


@tf.function
def parallel_rambo(xrand, n, sqrts, masses=None, check_physical=False):
    """********************************************************************
    *                       RAMBO                                         *
    *    RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)                  *
    *                                                                     *
    *    a democratic multi-particle phase space generator                *
    *    authors:  S.D. Ellis,  R. Kleiss,  W.J. Stirling                 *
    *    -- adjusted by Hans Kuijf, weights are logarithmic (20-08-90)    *
    *    this is py version 1.0 -  written by O. Mattelaer                *
    *    this is parallel version 1.0 -  written by J. Cruz-Martinez      *
    *                                                                     *
    *    xrand = array of (array of) random numbers                       *
    *    n  = number of particles                                         *
    *    sqrts = total centre-of-mass energy                              *
    *    masses = particle masses                                         *
    *  return                                                             *
    *    p  = array of particle momenta (events, nparticles, {E,px,py,pz})*
    *    wt = array of weight of the events                               *
    ***********************************************************************"""
    # Note:
    # By default no check is performed on the mass of the particle and so it is the
    # caller responsability to ensure the requested point is physical.
    # Furthermore, when running in tf-compiled mode no check _can_ be performed.
    if check_physical and tf.executing_eagerly():
        if masses is not None and tf.reduce_sum(masses) > sqrts:
            raise ValueError(f"Not enough energy ({sqrts}) to generate particles of mass: {masses}")

    # Ensure the com energy has the correct type
    sqrts = float_me(sqrts)

    # Generate the unconstrained momenta Q
    all_q = []
    for i in range(n):
        q = gen_unconstrained_momenta(xrand[:, i * 4 : (i + 1) * 4])
        all_q.append(q)
    # all_q: (nparticles, nevents, 4:{x,y,z,E})

    # Compute the parameters for the conformal transformation
    sum_q = tf.reduce_sum(all_q, axis=0)  # (nevents, 4)
    sum_q2 = sum_q ** 2
    qmass = tf.sqrt(sum_q2[:, -1] - tf.reduce_sum(sum_q2[:, :-1], axis=1))
    x = sqrts / qmass  # (nevents,)
    bquad = mul(-sum_q, 1.0 / qmass)
    bvec = bquad[:, :-1]
    gamma = -bquad[:, -1]
    a = 1.0 / (1.0 + gamma)

    # Perform the conformal transformation q->p
    all_p = []
    for q in all_q:
        bq = tf.reduce_sum(q[:, :-1] * bvec, axis=1)
        tmp = bq * a + q[:, -1]
        pvec = mul(q[:, :-1] + mul(bvec, tmp), x)
        pnrg = tf.reshape((q[:, -1] * gamma + bq) * x, (-1, 1))
        # Note that the Madgraph ME requires the momenta to be (E, px, py, pz)
        p = tf.concat([pnrg, pvec], axis=1)
        all_p.append(p)
    all_p = tf.stack(all_p, axis=1)  # (nevents, nparticles, 4)

    # Compute the weight of the phase space point
    # (for n > 2)
    wt = tf.math.log(PI / 2)
    for i in range(2, n):
        wt += tf.math.log(PI / 2) - 2.0 * tf.math.log(float_me(i - 1)) - tf.math.log(float_me(i))
    wt += (2 * n - 4) * tf.math.log(sqrts)

    if masses is None:
        return all_p, tf.ones_like(x) * wt

    # Masses were given, start by ensuring they are the correct "type"
    masses = float_me(masses)

    # If dealing with massive particles, momenta needs to be rescaled
    xfactor, new_E = massive_xfactor(sqrts, masses, all_p[:, :, 0])

    # rescale the momenta
    wt2 = 1.0
    wt3 = 0.0
    v = []

    rescaled_pvec = mul(all_p[:, :, 1:], xfactor)
    rescaled_ener = tf.expand_dims(new_E, axis=-1)
    massive_p = tf.concat([rescaled_ener, rescaled_pvec], axis=-1)

    # and weight
    v = mul(all_p[:, :, 0], xfactor)
    wt2 = tf.reduce_prod(v / new_E, axis=1)
    wt3 = tf.reduce_sum(v ** 2 / new_E, axis=1)

    wt += (2 * n - 3) * tf.math.log(xfactor) + tf.math.log(wt2 / wt3 * sqrts)

    return massive_p, wt


if __name__ == "__main__":
    import random, argparse

    arger = argparse.ArgumentParser()
    arger.add_argument("-n", "--nparticles", help="Number of particles to be run", type=int, default=4)
    arger.add_argument("-s", "--seed", help="Seed", type=int, default=4)
    arger.add_argument("-t", "--trials", help="Trials with different seeds", type=int, default=1)
    arger.add_argument("-e", "--tol", help="tolerance", type=float, default=1e-4)
    args = arger.parse_args()

    n = args.nparticles
    tol = args.tol
    sqrts = 7e3
    ndim = n * 4
    tf_masses = [173, 173]

    # Add here the madgraph rambo module for comparison
    mad_rambo = "../../mg5amcnlo/madgraph/various/rambo.py"
    import importlib.util

    rambo_spec = importlib.util.spec_from_file_location("rambo", mad_rambo)
    rambo_modu = importlib.util.module_from_spec(rambo_spec)
    rambo_spec.loader.exec_module(rambo_modu)
    rambo = getattr(rambo_modu, "RAMBO")
    flist = getattr(rambo_modu, "FortranList")
    #####

    masses = flist(n)

    for i in range(n-2):
        tf_masses.insert(0,0)

    for i in range(args.trials):
        seed = args.seed + i
        print(f"Computing PS with seed {seed}")
        random.seed(seed)
        all_rand = []
        for _ in range(2):
            all_rand.append([random.uniform(0, 1) for _ in range(ndim)])
        xrand = float_me(all_rand)
        p, wt = parallel_rambo(xrand, n, sqrts)

        pmass = wtmass = None
        pmass, wtmass = parallel_rambo(xrand, n, sqrts, masses=tf_masses)

        all_p = [p, pmass]
        all_w = [wt, wtmass]


        for i, mode in enumerate(["massless", "massive"]):
            random.seed(seed)

            # For a check I want to reduce how much I mess with these fortran lists
            for idx, mass in enumerate(tf_masses):
                if mode == "massive":
                    masses[idx + 1] = mass
                elif mode == "massless":
                    masses[idx + 1] = 0.0

            p1r, wt1 = rambo(n, sqrts, masses)
            p1 = np.array(p1r).T
            p2r, wt2 = rambo(n, sqrts, masses)
            p2 = np.array(p2r).T

            print(f"Checking first {mode} ps point...")
            fortran_axis = [1, 2, 3, 0]
            np.testing.assert_allclose(all_p[i][0].numpy()[:, fortran_axis], p1, rtol=tol)
            print(f"Checking second {mode} ps point...")
            np.testing.assert_allclose(all_p[i][1].numpy()[:, fortran_axis], p2, rtol=tol)
            print("Checking weight...")
            np.testing.assert_allclose([wt1, wt2], all_w[i], rtol=tol)
            print("...")

        print("All good!")
