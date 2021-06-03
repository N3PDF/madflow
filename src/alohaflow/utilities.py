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
from .phasespace import ramboflow

import numpy as np
import tensorflow as tf
from vegasflow import vegas_wrapper


def _generate_luminosity(pdf, q):
    """Generates a luminosity function"""
    q2 = float_me(q ** 2)

    def luminosity_function(x1, x2, flavours):
        """Returns f(x1)*f(x2) for the given flavours"""
        q2array = tf.ones_like(x1) * q2
        hadron_1 = pdf.xfxQ2(flavours, x1, q2array)
        hadron_2 = pdf.xfxQ2(flavours, x2, q2array)
        return (hadron_1 * hadron_2) / x1 / x2

    return luminosity_function


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
        all_ps, wts, x1, x2 = ramboflow(xrand, nparticles, sqrts, masses=out_masses)
        smatrix = matrix_wgt(all_ps)
        pdf_result = luminosity_function(x1, x2, int_me(flavours))
        return smatrix * pdf_result * wts

    ndim = (nparticles - 2) * 4 + 2
    tf.random.set_seed(4)
    return vegas_wrapper(cross_section, ndim, n_iter, n_events)
