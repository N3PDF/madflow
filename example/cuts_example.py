#!/usr/bin/env python3

"""
Example script that goes from the generation of a matrix element
to the integration with the corresponding cuts

The matrix element to be run is fixed to be: g g > t t~

```
    ~$ ./cuts_example.py
```
"""
import os
os.environ.setdefault("VEGASFLOW_LOG_LEVEL", "2")
import sys
import argparse
import tempfile
import subprocess as sp
from pathlib import Path

from vegasflow import vegas_wrapper
from pdfflow import mkPDF, float_me, int_me

from alohaflow.config import get_madgraph_path
from alohaflow.utilities import _get_ps
import tensorflow as tf

# Create some temporary directories and files 
# (won't be removed on output so they can be inspected)
out_path = Path(tempfile.mkdtemp())
script_path = Path(tempfile.mktemp())

# Note that if another process is run, the imports below
# must be changed accordingly, it can be made into options later on
madgraph_script = f"""generate g g > t t~
output pyout {out_path}"""

if __name__ == "__main__":
    arger = argparse.ArgumentParser(__doc__)
    arger.add_argument("-v", "--verbose", help="Print extra info", action="store_true")
    arger.add_argument("-p", "--pdf", help="PDF set", type=str, default="NNPDF31_nnlo_as_0118")

    args = arger.parse_args()

    # Run the process
    script_path.write_text(madgraph_script)
    mg5_path = get_madgraph_path()
    mg5_exe = mg5_path / "bin/mg5_aMC"
    if not mg5_exe.exists():
        raise RuntimeError(f"Could not find madgraph executable at {mg5_exe}")
    if args.verbose:
        print(f" > Running madgraph script at {script_path}")
        output = None
    else:
        output = sp.DEVNULL
    sp.run([mg5_exe, "-f", script_path], stdout=output)
    if args.verbose:
        print(f" > Madgraph output can be found at {out_path}")

    # And now bring the python files from the output folder
    sys.path.insert(0, out_path.as_posix())
    # Note: these imports must be changed if the process is changed
    from matrix_1_gg_ttx import Matrix_1_gg_ttx as matrix_element, import_ufo, get_model_param

    # Read the parameters of the model
    model_sm = mg5_path / "models/sm"
    model = import_ufo.import_model(model_sm.as_posix())
    model_params = get_model_param(model, (out_path / 'Cards/param_card.dat').as_posix())
    

    matrix = matrix_element()

    nparticles = int(matrix.nexternal)
    ndim = (nparticles - 2)*4 + 2
    sqrts = 7e3
    massive_particles = 2

    masses = model_params.get_masses()*massive_particles + [0]*(nparticles-massive_particles-2)
    model_params.freeze_alpha_s(0.118)

    if args.verbose:
        xrand = tf.random.uniform(shape=(10, ndim), dtype=tf.float64)
        ps, wgt, x1, x2 = _get_ps(xrand, nparticles, sqrts, masses)
        wgts = matrix.smatrix(ps, *model_params.evaluate(None))
        print(f"Weights: \n{wgts.numpy()}")

    q2 = float_me(91.46**2)
    pdf = mkPDF(args.pdf + "/0")
    def luminosity(x1, x2, flavours):
        """Returns f(x1)*f(x2) for the given flavours"""
        q2array = tf.ones_like(x1) * q2
        hadron_1 = pdf.xfxQ2(flavours, x1, q2array)
        hadron_2 = pdf.xfxQ2(flavours, x2, q2array)
        return (hadron_1 * hadron_2) / x1 / x2

    def cross_section(xrand, **kwargs):
        all_ps, wts, x1, x2 = _get_ps(xrand, nparticles, sqrts, masses)
        pdf_result = luminosity(x1, x2, int_me([21]))
        smatrix = matrix.smatrix(all_ps, *model_params.evaluate(None))
        return smatrix * pdf_result * wts

    _ = vegas_wrapper(cross_section, ndim, 5, int(1e5))
