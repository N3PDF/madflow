#!/usr/bin/env python3

"""
Example script that goes from the generation of a matrix element
to the integration with the corresponding cuts

The matrix element run by default is: g g > t t~

```
    ~$ ./cuts_example.py --madgraph_process "g g > g g"
```

It is possible to apply some mock cuts (pt > 60) with the option  `-c`
It is also possible to use a variable coupling with muF = muR = pt of the top with the option `-g`
"""
import re
import sys
import importlib
import argparse
import tempfile
import subprocess as sp
from pathlib import Path

from vegasflow import vegas_wrapper
from pdfflow import mkPDF, float_me, int_me, run_eager

from alohaflow.config import get_madgraph_path
from alohaflow.phasespace import ramboflow, PhaseSpaceGenerator
import tensorflow as tf

# Create some temporary directories and files
# (won't be removed on output so they can be inspected)
out_path = Path(tempfile.mkdtemp(prefix="mad"))
script_path = Path(tempfile.mktemp(prefix="mad_script"))

# Note that if another process is run, the imports below
# must be changed accordingly, it can be made into options later on
re_name = re.compile(r"\w{3,}")


def _import_module_from_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    arger = argparse.ArgumentParser(__doc__)
    arger.add_argument("-v", "--verbose", help="Print extra info", action="store_true")
    arger.add_argument("-p", "--pdf", help="PDF set", type=str, default="NNPDF31_nnlo_as_0118")
    arger.add_argument("-c", "--enable_cuts", help="Enable the cuts", action="store_true")
    arger.add_argument(
        "--madgraph_process",
        help="Set the madgraph process to be run",
        type=str,
        default="g g > t t~",
    )
    arger.add_argument(
        "-m", "--massive_particles", help="Number of massive particles", type=int, default=2
    )
    arger.add_argument("-g", "--variable_g", help="Use variable g_s", action="store_true")

    args = arger.parse_args()

    # Prepare the madgraph script
    madgraph_script = f"""generate {args.madgraph_process}
output pyout {out_path}"""

    # Run the process in madgraph and create the tensorized output
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
    sp.run([mg5_exe, "-f", script_path], stdout=output, check=True)
    if args.verbose:
        print(f" > Madgraph output can be found at {out_path}")

    # Import the matrix file from the output folder as a module
    sys.path.insert(0, out_path.as_posix())
    matrix_file = next(out_path.glob("matrix_*.py")).name
    matrix_name = re_name.findall(matrix_file)[0]
    matrix_module = _import_module_from_path(out_path / matrix_file, matrix_name)
    # Import specifically the matrix element
    matrix_element = getattr(matrix_module, matrix_name.capitalize())

    # Read the parameters of the model
    model_sm = mg5_path / "models/sm"
    model = matrix_module.import_ufo.import_model(model_sm.as_posix())
    model_params = matrix_module.get_model_param(
        model, (out_path / "Cards/param_card.dat").as_posix()
    )

    # Instantiate the matrix element
    matrix = matrix_element()

    # Set up the parameters of the process
    nparticles = int(matrix.nexternal)
    ndim = (nparticles - 2) * 4 + 2
    sqrts = 7e3
    massive_particles = args.massive_particles
    non_massive = nparticles - massive_particles - 2
    # Assume that the massive particles go first
    # and _if_ the number of masses is below the number of massive particle
    # assume they are all the same mass (usually the top anyway)
    param_masses = model_params.get_masses()
    if len(param_masses) < massive_particles:
        param_masses *= massive_particles

    masses = param_masses + [0.0] * non_massive

    if not args.variable_g:
        q2 = float_me(91.46 ** 2)
        model_params.freeze_alpha_s(0.118)

    if args.verbose:
        test_events = 5
        xrand = tf.random.uniform(shape=(test_events, ndim), dtype=tf.float64)
        ps, wgt, x1, x2 = ramboflow(xrand, nparticles, sqrts, masses)
        if args.variable_g:
            alpha_s = float_me([0.118]*test_events)
        wgts = matrix.smatrix(ps, *model_params.evaluate(alpha_s))
        print(f"Weights: \n{wgts.numpy()}")

    pdf = mkPDF(args.pdf + "/0")

    # Create the pase space and register the cuts
    phasespace = PhaseSpaceGenerator(nparticles, sqrts, masses)
    if args.enable_cuts:
        phasespace.register_cut("pt", particle=3, min_val=60.0)
        phasespace.register_cut("pt", particle=2, min_val=60.0)

    def luminosity(x1, x2, flavours, q2array):
        """Returns f(x1)*f(x2) for the given flavours"""
        hadron_1 = pdf.xfxQ2(flavours, x1, q2array)
        hadron_2 = pdf.xfxQ2(flavours, x2, q2array)
        return (hadron_1 * hadron_2) / x1 / x2

    def cross_section(xrand, **kwargs):
        """Compute the cross section"""
        # Generate the phase space point
        all_ps, wts, x1, x2, idx = phasespace(xrand)

        # Compute the value of muF==muR if needed
        if args.variable_g:
            q2array = phasespace.pt(all_ps[:,2,:]) ** 2
            alpha_s = pdf.alphasQ2(q2array)
        else:
            q2array = tf.ones_like(x1) * q2
            alpha_s = None

        # Get the luminosity per event
        pdf_result = luminosity(x1, x2, int_me([21]), q2array)

        # Compute the cross section
        smatrix = matrix.smatrix(all_ps, *model_params.evaluate(alpha_s))
        ret = smatrix * pdf_result * wts
        if args.enable_cuts:
            ret = tf.scatter_nd(idx, ret, shape=xrand.shape[0:1])
        return ret

    _ = vegas_wrapper(cross_section, ndim, 5, int(1e5))
