"""
    Test for the integration routine
"""

import os
import numpy as np
from alohaflow.utilities import one_matrix_integration
from pdfflow import mkPDF

# For github actions, check whether we find a PDF set directorty
git_pdfs = os.environ.get("PDFDIR")

def test_integration():
    """ Regresion-style check to the integration routine using a predefined
    mockup matrix element.
    Checks that the result is within 3 sigmas of the true result
    """
    from .mockup_debug_me import Matrix_1_gg_ttx, model_params
    matrix = Matrix_1_gg_ttx()
    pdf = mkPDF("NNPDF31_nnlo_as_0118/0", dirname=git_pdfs)
    res, error = one_matrix_integration(matrix, model_params, pdf=pdf, flavours=(0,), out_masses=[173.0, 173.0])
    true_result = 103.4
    assert np.fabs(true_result - res) < 3*error
