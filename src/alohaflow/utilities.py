"""
    Utilities and wrappers for quickstarting integration

    The main utility in this module is ``one_matrix_integration``
    which is a wrapper to a full Monte Carlo integration of a given matrix
    element generated with madgraph.
    
    For the example below the matrix element generated is
    ``g g > t t~``

    >>> from alohaflow.utilities import one_matrix_integration
    >>> from pdfflow import mkPDF
    >>> pdf = mkPDF("NNPDF31_nnlo_as_0118/0")
    >>> one_matrix_integration(matrix, model_params, pdf=pdf, flavours=(0,), out_masses=[173.0, 173.0])
    [INFO]  > Final results: 103.439 +/- 0.1147
"""
from .config import int_me, float_me
from .phasespace import rambo

import numpy as np
import tensorflow as tf
from vegasflow import vegas_wrapper


def _generate_luminosity(pdf, q):
    """ Generates a luminosity function """
    q2 = float_me(q ** 2)

    def luminosity_function(x1, x2, flavours):
        """Returns f(x1)*f(x2) for the given flavours"""
        q2array = tf.ones_like(x1) * q2
        hadron_1 = pdf.xfxQ2(flavours, x1, q2array)
        hadron_2 = pdf.xfxQ2(flavours, x2, q2array)
        return (hadron_1 * hadron_2) / x1 / x2

    return luminosity_function


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


def _get_ps(xrand, nparticles, com_sqrts, masses=None):
    """Takes as input an array of nevent x ndim random points and outputs
    an array of momenta (nevents x nparticles x 4) in the C.O.M.
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


def one_matrix_integration(
    matrix,
    model_params,
    sqrts=7e3,
    n_events=int(1e5),
    n_iter=5,
    q=91.46,
    pdf=None,
    flavours=None,
    out_masses=None,
):
    """Receives a matrix element from Madgraph"""
    nparticles = int(matrix.nexternal)
    if pdf is None:

        def luminosity_function(x, *args):
            return tf.ones_like(x)

    else:
        luminosity_function = _generate_luminosity(pdf, q)

    # Prepare the matrix element
    def matrix_wgt(all_p):
        return matrix.smatrix(all_p, *model_params)

    # Prepare the integrand
    def cross_section(xrand, **kwargs):
        all_ps, wts, x1, x2 = _get_ps(xrand, nparticles, sqrts, masses=out_masses)
        smatrix = matrix_wgt(all_ps)
        pdf_result = luminosity_function(x1, x2, int_me(flavours))
        return smatrix * pdf_result * wts

    ndim = (nparticles - 2) * 4 + 2
    tf.random.set_seed(4)
    return vegas_wrapper(cross_section, ndim, n_iter, n_events)
