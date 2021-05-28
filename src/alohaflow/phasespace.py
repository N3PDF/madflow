"""
    PhaseSpace generation routines

    The output momenta has always the structure (E, px, py, pz)
    When a complete set of momenta is output the structure is (n_events, n_particles, n_dim)
"""

from .config import float_me, int_me, DTYPE, run_eager

import numpy as np
import copy
import tensorflow as tf

import logging

PI = float_me(np.pi)
ACC = float_me(1e-10)
logger = logging.getLogger(__name__)

# Helpers for rambo
events_signature = tf.TensorSpec(shape=[None, 1], dtype=DTYPE)
p_signature = tf.TensorSpec(shape=[None, 4], dtype=DTYPE)
ps_signature = tf.TensorSpec(shape=[None, None, 4], dtype=DTYPE)


@tf.function(input_signature=2 * [p_signature])
def _fourdot(f1, f2):
    ener = f1[:, 0] * f2[:, 0]
    pmom = tf.reduce_sum(f1[:, 1:] * f2[:, 1:], axis=1)
    return ener - pmom


@tf.function(input_signature=[p_signature])
def _invariant_mass(fm):
    return _fourdot(fm, fm)


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
    momenta.

    This relies in an iterative process which can't be naively parallelized
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
    """Perform the conformal transformation q->p"""
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


def _get_x1x2(xarr, shat_min, s_in):
    """Receives two random numbers and return the
    value of the invariant mass of the center of mass
    as well as the jacobian of the x1,x2 -> tau-y transformation
    and the values of x1 and x2.

    The xarr array is of shape (batch_size, 2)
    """
    taumin = shat_min / s_in
    taumax = float_me(1.0)
    # Pick tau
    delta_tau = taumax - taumin
    tau = xarr[:, 0] * delta_tau + taumin
    wgt = delta_tau
    x1 = tf.pow(tau, xarr[:, 1])
    x2 = tau / x1
    wgt *= -1.0 * tf.math.log(tau)
    shat = x1 * x2 * s_in
    return shat, wgt, x1, x2


def ramboflow(xrand, nparticles, com_sqrts, masses=None):
    """Takes as input an array of nevent x ndim random points and outputs
    an array of momenta (nevents x nparticles x 4) in the C.O.M.

    The two first particles are the two incoming particles which are always
    massless. The masses of the outgoing particles is given by the list ``masses``
    if not given all outgoing particles are taken as massless

    Parameters
    ----------
        xrand: tensor (nevents, (nparticles-2)*4+2)
            random numbers to generate momenta
        nparticles: int
            number of external particles
        com_sqrts: float
            center of mass energy
        masses: lust (nparticles-2)
            masses of the outgoing particles

    Return
    ------
        final_p: tensor (nevents, nparticles, 4)
            phase space point for all particles for all events (E, px, py, pz)
        wgt: tensor (nevents)
            weight of the phase space points
        x1: tensor (nevents)
            momentum fraction of parton 1
        x2: tensor (nevents)
            momentum fraction of parton 2
    """
    if masses is None:
        shat_min = float_me(0.0)
    else:
        shat_min = float_me(np.sum(masses) ** 2)

    # Sample the initial state
    shat, wgt, x1, x2 = _get_x1x2(xrand[:, :2], shat_min, com_sqrts ** 2)
    roots = tf.sqrt(shat)

    # Sample the outgoing states
    p_out, wtps = rambo(xrand[:, 2:], int(nparticles - 2), roots, masses=masses)
    wgt *= wtps

    # Now stack the input states on top
    zeros = tf.zeros_like(x1)
    ein = roots / 2.0
    pa = tf.expand_dims(tf.stack([ein, zeros, zeros, ein], axis=1), 1)
    pb = tf.expand_dims(tf.stack([ein, zeros, zeros, -ein], axis=1), 1)

    final_p = tf.concat([pa, pb, p_out], axis=1)

    # Add the flux factor
    wgt *= float_me(389379365.6)  # GeV to pb
    wgt /= 2 * shat

    return final_p, wgt, x1, x2


def _boost_to_lab(p_com, x1, x2):
    """Boost the momenta back from the COM frame of the initial partons
    to the lab frame

    Parameters
    ----------
        p_comp: tensor (nevents, nparticles, 4)
            phase space point for all particles for all events (E, px, py, pz)
            in the center of mass frame
        x1: tensor (nevents)
            momentum fraction of parton 1
        x2: tensor (nevents)
            momentum fraction of parton 2

    Return
    ------
        tensor (nevents, nparticles, 4)
            phase space point for all particles for all events (E, px, py, pz)
            in the center of mass frame
    """
    # Boost the momenta back from the COM of pa + pb
    eta = -0.5 * tf.math.log(x1 / x2)
    cth = tf.math.cosh(eta)
    sth = tf.math.sinh(eta)
    # Generate the boost matrix
    zeros = tf.zeros_like(x1)
    ones = tf.ones_like(x1)
    bE = tf.stack([cth, zeros, zeros, -sth], axis=1)
    bX = tf.stack([zeros, ones, zeros, zeros], axis=1)
    bY = tf.stack([zeros, zeros, ones, zeros], axis=1)
    bZ = tf.stack([-sth, zeros, zeros, cth], axis=1)

    bmat = tf.stack([bE, bX, bY, bZ], axis=1)
    # Apply boost
    return tf.keras.backend.batch_dot(p_com, bmat, axes=2)


class PhaseSpaceGenerator:
    """Phase space generator class
    able to not only generate momenta but also apply cuts

    Parameters
    ----------
        nparticles: int
            number of external particles
        com_sqrts: float
            com energy
        masses: list(float)
            mass of the outgoing particles
        algorithm: str
            algoirhtm to be used (by default ramboflow)
        com_output: bool
            whether the output should be on the com frame (default, true) or the lb frame (false)
    """

    def __init__(self, nparticles, com_sqrts, masses=None, com_output=True, algorithm="ramboflow"):
        if masses is None:
            masses = [0.0] * (nparticles - 2)
        if len(masses) != (nparticles - 2):
            raise ValueError(
                "Missmatch in PhaseSpaceGenerator between particles and masses"
                f" {len(masses)} given for {nparticles-2} outgoing particles"
            )
        self._sqrts = float_me(com_sqrts)
        self._masses = masses
        self._nparticles = nparticles
        self._cuts = []
        self._cuts_info = []
        self._com_output = com_output

        if algorithm == "ramboflow":
            self._ps_gen = ramboflow
        else:
            raise ValueError(f"PS algorithm {algorithm} not understood")

    def clear_cuts(self):
        """Clear all cuts, if not running on eager mode, we need to regenerate call"""
        self._cuts = []
        self._cuts_info = []
        orig_function = self.__call__.python_function
        orig_signature = self.__call__.input_signature
        self.__call__ = tf.function(orig_function, input_signature=orig_signature)

    @staticmethod
    def mt2(ps_point):
        """Transverse mass squared of the given ps point (nevents, 4)"""
        pt2 = PhaseSpaceGenerator.pt(ps_point) ** 2
        m2 = _invariant_mass(ps_point)
        return m2 + pt2

    @staticmethod
    def mt(ps_point):
        """Transverse mass of the given ps point"""
        return tf.math.sqrt(PhaseSpaceGenerator.mt2(ps_point))

    @staticmethod
    def pt(ps_point):
        """Compute the pt of the ps point (nevents,4)"""
        px = ps_point[:, 1]
        py = ps_point[:, 2]
        return tf.math.sqrt(px ** 2 + py ** 2)

    def register_cut(self, variable, particle=None, min_val=None, max_val=None):
        """Register the cut for the given variable for the given particle (if needed)
        The variables must be done as a string and be a valid method of this class
        the min and max values are the values between which the variable must be found

        Warning:
            at this point the user is trusted to use the argument 'particle'
            whenever the variable applies to

        The cut functions will return a tensor of the idx of the accepted events
        and a tensor of booleans

        Parameters
        ----------
            variable: str
                name of the variable to which the cut applies
            particle: int
                particle index to which the cut applies
            min_val: float
                minimum accepted value of the variable
            max_val: float
                maximum accepted value of the variable
        """
        try:
            fun = getattr(self, variable)
        except AttributeError:
            raise ValueError(f"{variable} is not implemented")

        if particle is not None and particle >= self._nparticles:
            raise ValueError(f"Cannot apply cuts to particle {particle}, python idx starts at 0!")
        if min_val is None and max_val is None:
            logger.warning(f"Cut for {variable} has no min or max val, ignoring")

        def cut_function(phase_space):
            if particle is not None:
                phase_space = phase_space[:, particle, :]
            variable_value = fun(phase_space)

            if min_val is not None:
                min_pass = variable_value > float_me(min_val)
                if max_val is None:
                    return min_pass
            if max_val is not None:
                max_pass = variable_value < float_me(max_val)
                if min_val is None:
                    return max_pass
            return tf.logical_and(min_pass, max_pass)

        if particle is None:
            cut_fun = tf.function(cut_function, input_signature=[p_signature])
            self._cuts_info.append(f"{min_val} < {variable}({particle}) < {max_val}")
        else:
            cut_fun = tf.function(cut_function, input_signature=[ps_signature])
            self._cuts_info.append(f"{min_val} < {variable} < {max_val}")
        self._cuts.append(cut_fun)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=DTYPE)])
    def __call__(self, xrand):
        """
            Generate phase space points according to the xrand random input

        Parameters
        ----------
            xrand: tensor (nevents, (nparticles-2)*4+2)
                random numbers to generate momenta

        Return  (if there are cuts, the output nevents =/= input nevents)
        ------
            final_p: tensor (nevents, nparticles, 4)
                phase space point for all particles for all events (E, px, py, pz)
            wgt: tensor (nevents)
                weight of the phase space points
            x1: tensor (nevents)
                momentum fraction of parton 1
            x2: tensor (nevents)
                momentum fraction of parton 2
            idx: tensor (nevents)
                if there are cuts, index of the passing events, (1,) otherwise
        """
        ps, wgt, x1, x2 = self._ps_gen(xrand, self._nparticles, self._sqrts, self._masses)

        # Apply all cuts
        if self._cuts:
            stripes = [cut(ps) for cut in self._cuts]
            passing_values = tf.math.reduce_all(stripes, axis=0)
            ps = tf.boolean_mask(ps, passing_values, axis=0)
            wgt = tf.boolean_mask(wgt, passing_values)
            x1 = tf.boolean_mask(x1, passing_values)
            x2 = tf.boolean_mask(x2, passing_values)
            idx = int_me(tf.where(passing_values))
        else:
            idx = float_me(1.0)

        if not self._com_output:
            ps = _boost_to_lab(ps, x1, x2)

        return ps, wgt, x1, x2, idx
