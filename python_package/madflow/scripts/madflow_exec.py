#!/usr/bin/env python3

"""
Example script that goes from the generation of a matrix element
to the integration with the corresponding cuts

The matrix element run by default is: g g > t t~

```
    ~$ madflow --madgraph_process "g g > t t~"
```

It is possible to apply some mock cuts (pt_min) with the option  `--pt_cut` (defaults to 30)
By default the PDF and the strong coupling is computed for muF = muR = sum(mT)/2 but
a fixed value of the scale can be given with --fixed_scale (defaults to 91.46).

LHE files can be produced with the `--histograms` flag.
"""
import re
import sys
import time
import itertools
import importlib
import argparse
import tempfile
import subprocess as sp
from pathlib import Path
import logging
import numpy as np

from madflow.config import get_madgraph_path, get_madgraph_exe, DTYPE, DTYPEINT, float_me, int_me, run_eager, guess_events_limit
from madflow.phasespace import PhaseSpaceGenerator
from madflow.lhe_writer import LheWriter

from vegasflow import VegasFlow
from pdfflow import mkPDF

import tensorflow as tf

DEFAULT_PDF = "NNPDF31_nnlo_as_0118"
logger = logging.getLogger(__name__)

# Note that if another process is run, the imports below
# must be changed accordingly, it can be made into options later on
_flav_dict = {"g": 21, "d": 1, "u": 2, "s": 3, "c": 4, "b": 5, "t": 6}


def _read_flav(flav_str):
    particle = _flav_dict.get(flav_str[0])
    if particle is None:
        raise ValueError(
            f"Could not understand the incoming flavour: {flav_str} "
            "You can skip this error by using --no_pdf"
        )
    if flav_str[-1] == "~":
        particle = -particle
    return particle


def _import_module_from_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generate_madgraph_process(process, output_folder):
    """Given a process string in the madgraph format
    ex: g g > t t~
    generate the madgraph process file in the appropriate folder
    """
    madgraph_script = f"""generate {process}
output pyout {output_folder}"""
    script_path = Path(tempfile.mktemp(prefix="mad_script_"))
    script_path.write_text(madgraph_script)
    logger.debug("Writing madgraph output script at %s", script_path)

    mg5_command = [get_madgraph_exe(), "-f", script_path]
    mg5_p = sp.run(mg5_command, capture_output=True, check=True)

    # madgraph is not very helpful when the plugin is not found
    # so let's "brute force" that information
    parsed_output = mg5_p.stdout.decode()
    if "initialize a new directory: pyout" in parsed_output:
        logger.error(parsed_output)
        raise ValueError(
            "It seems madgraph was not able to find the pyout plugin. "
            f"Please, ensure the plugin is linked to {get_madgraph_path()}/PLUGIN/ "
            "For more information visit https://github.com/N3PDF/madflow"
        )

    logger.debug(parsed_output)
    logger.info("Matrix files written to: %s", output_folder)


def _import_matrices(output_folder):
    """Given a folder with the pyout matrix_xxx.py files,
    import them all and instantiate the matrix element and model files"""
    sys.path.insert(0, output_folder.as_posix())
    re_name = re.compile(r"\w{3,}")
    matrices = []
    models = []
    for i, matrix_file in enumerate(output_folder.glob("matrix_*.py")):
        matrix_name = re_name.findall(matrix_file.name)[0]
        matrix_module = _import_module_from_path(matrix_file, matrix_name)
        # Import specifically the matrix element
        matrix_element = getattr(matrix_module, matrix_name.capitalize())
        matrices.append(matrix_element())

        # Read the parameters of the model, shared among matrix elements
        model_sm = get_madgraph_path() / "models/sm"
        model = matrix_module.import_ufo.import_model(model_sm.as_posix())
        # Instantiate matrix element and models
        model_params = matrix_module.get_model_param(
            model, (output_folder / "Cards/param_card.dat").as_posix()
        )
        models.append(model_params)

    return matrices, models


def _generate_initial_states(matrices):
    """Reads a list of matrices and outputs a list of tuples of initial states
    each element in the list will be a tuple ([flavours hadron 1, flavours hadron 2])
    for each matrix
    """
    initial_flavours = []
    for matrix in matrices:
        initials = matrix.initial_states
        flavs_1, flavs_2 = zip(*initials)
        if matrix.mirror_initial_states:
            m2, m1 = zip(*initials)
            flavs_1 += m1
            flavs_2 += m2
        initial_flavours.append((flavs_1, flavs_2))
    return initial_flavours


def madflow_main(args=None, quick_return=False):
    arger = argparse.ArgumentParser(__doc__)
    arger.add_argument("-v", "--verbose", help="Print extra info", action="store_true")
    arger.add_argument("-p", "--pdf", help="PDF set", type=str, default=DEFAULT_PDF)
    arger.add_argument(
        "--no_pdf", help="Don't use a PDF for the initial state", action="store_true"
    )
    arger.add_argument(
        "--madgraph_process",
        help="Set the madgraph process to be run",
        type=str,
        default="g g > t t~",
    )
    arger.add_argument(
        "-m", "--massive_particles", help="Number of massive particles", type=int, default=2
    )
    arger.add_argument(
        "-q",
        "--fixed_scale",
        help="Fix value of scale muR=muF (and alphas(q)), if this flag is not provided take dynamical scale q2 = sum(mT)/2",
        type=float,
        nargs="?",
        const=91.46,
    )
    arger.add_argument(
        "-c",
        "--pt_cut",
        help="Enable a pt cut for the outgoing particles",
        type=float,
        nargs="?",
        const=30.0,
    )
    arger.add_argument("--histograms", help="Generate LHE files/histograms", action="store_true")
    arger.add_argument(
        "-i", "--iterations", help="Iterations of vegasfow to run", type=int, default=10
    )
    arger.add_argument(
        "-f", "--frozen_iter", help="Iterations with frozen grid", type=int, default=0
    )

    args = arger.parse_args(args)
    if quick_return:
        return args, None, None

    out_path = Path(tempfile.mkdtemp(prefix="mad_"))
    _generate_madgraph_process(args.madgraph_process, out_path)
    matrices, models = _import_matrices(out_path)

    if args.no_pdf:
        initial_flavours = [None]
    else:
        pdf = mkPDF(args.pdf + "/0")
        initial_flavours = _generate_initial_states(matrices)
        # Prepare 1) the flavours we will ask pdfflow for
        # 2) the indexes for the gathers
        flavours_hadron_1, flavours_hadron_2 = zip(*initial_flavours)
        # These are passed to pdfflow
        hadron_1 = list(set(itertools.chain(*flavours_hadron_1)))
        hadron_2 = list(set(itertools.chain(*flavours_hadron_2)))
        gather_1 = []
        gather_2 = []
        for p1, p2 in initial_flavours:
            gather_1.append([hadron_1.index(i) for i in p1])
            gather_2.append([hadron_2.index(i) for i in p2])

    ### Set up some parameters for the process
    sqrts = 13e3
    # The number of particles is the same for all matrices
    nparticles = int(matrices[0].nexternal)
    ndim = (nparticles - 2) * 4 + 2
    massive_particles = args.massive_particles
    non_massive = nparticles - massive_particles - 2
    # For this script the massive particles go always first
    # as the output should always be to particles and not wrappers
    # _if_ the number of masses is below the number of massive particle
    param_masses = models[0].get_masses()
    if len(param_masses) < massive_particles:
        param_masses *= massive_particles
    param_masses = [i.numpy() for i in param_masses]
    masses = param_masses + [0.0] * non_massive
    logger.debug("Masses: %s", masses)
    ###################################################

    if args.fixed_scale is None:
        logger.info("Set variable muF=muR=sum(mT)/2")
    else:
        logger.info("Setting fixed muF=muR=%.2f GeV.", args.fixed_scale)
        q2 = float_me(args.fixed_scale ** 2)
        if args.no_pdf:
            alpha_s = 0.118
        else:
            alpha_s = np.squeeze(pdf.alphasQ2([q2]))
        logger.info("Setting alpha_s = %.4f.", alpha_s)
        # Fix all models
        for model in models:
            models.freeze_alpha_s(alpha_s)

    # Create the phase space
    phasespace = PhaseSpaceGenerator(nparticles, sqrts, masses, com_output=False)

    # Register the cuts with the phase space
    if args.pt_cut is not None:
        for i in range(2, nparticles):
            logger.info("Applying cut of pt > %.2f to particle %d", args.pt_cut, i)
            phasespace.register_cut("pt", particle=i, min_val=args.pt_cut)

    # Test the matrix elements
    test_events = 10
    test_xrand = tf.random.uniform(shape=(test_events, ndim), dtype=tf.float64)
    test_ps, test_wt, _, _, _ = phasespace(test_xrand)
    test_alpha = float_me([0.118] * len(test_wt))
    for matrix, model in zip(matrices, models):
        wgts = matrix.smatrix(test_ps, *model.evaluate(test_alpha))
        logger.info("Testing %s: %s", matrix, wgts.numpy())

    @tf.function(input_signature=3 * [tf.TensorSpec(shape=[None], dtype=DTYPE)])
    def luminosity_function(x1, x2, q2array):
        raw_proton_1 = pdf.xfxQ2(int_me(hadron_1), x1, q2array)
        raw_proton_2 = pdf.xfxQ2(int_me(hadron_2), x2, q2array)
        # Ensure they have the right shape, just in case
        proton_1 = tf.reshape(raw_proton_1, (-1, len(hadron_1)))
        proton_2 = tf.reshape(raw_proton_2, (-1, len(hadron_2)))
        return proton_1, proton_2

    def generate_integrand(lhewriter=None):
        """Generate a cross section with (or without) a LHE parser"""

        def cross_section(xrand, n_dim=ndim, weight=1.0):
            """Compute the cross section"""
            # Generate the phase space point
            all_ps, wts, x1, x2, idx = phasespace(xrand)

            # Compute the value of muF==muR if needed
            if args.fixed_scale is None:
                full_mt = tf.reduce_sum(phasespace.mt(all_ps[:, 2:nparticles, :]), axis=-1)
                q2array = (full_mt / 2.0) ** 2
                alpha_s = pdf.alphasQ2(q2array)
            else:
                q2array = tf.ones_like(x1) * q2
                alpha_s = None

            # Get the luminosity per event
            if args.no_pdf:
                luminosity = float_me(1.0)
            else:
                proton_1, proton_2 = luminosity_function(x1, x2, q2array)

            # Compute each matrix element
            ret = 0.0
            for i, (matrix, model) in enumerate(zip(matrices, models)):
                smatrix = matrix.smatrix(all_ps, *model.evaluate(alpha_s))
                if not args.no_pdf:
                    p1 = tf.gather(proton_1, gather_1[i], axis=1)
                    p2 = tf.gather(proton_2, gather_2[i], axis=1)
                    # Sum all input channels together for now
                    luminosity = tf.reduce_sum(p1 * p2, axis=1) / x1 / x2
                ret += luminosity * smatrix

            # Final cross section
            ret *= wts

            if lhewriter is not None:
                # Fill up the LHE grid
                if args.pt_cut is not None:
                    weight = tf.gather(weight, idx)[:, 0]
                tf.py_function(func=lhewriter.lhe_parser, inp=[all_ps, ret * weight], Tout=DTYPE)

            if args.pt_cut is not None:
                ret = tf.scatter_nd(idx, ret, shape=xrand.shape[0:1])

            return ret

        return cross_section

    events_per_iteration = int(1e6)
    events_limit = guess_events_limit(nparticles)
    frozen_limit = events_limit*2
    if nparticles >= 5 and args.frozen_iter == 0:
        logger.warning("With this many particles (> 5) it is recommended to run with frozen iterations")

    vegas = VegasFlow(ndim, events_per_iteration, events_limit=events_limit)
    integrand = generate_integrand()
    vegas.compile(integrand)

    if args.frozen_iter == 0:
        warmup_iterations = args.iterations // 2
    else:
        warmup_iterations = max(args.iterations - args.frozen_iter, 2)
    logger.info(
        "Running %d warm-up iterations of %d events each", warmup_iterations, events_per_iteration
    )
    vegas.run_integration(warmup_iterations)

    if args.frozen_iter > 0:
        vegas.events_per_run = frozen_limit
        vegas.freeze_grid()
        final_iterations = args.frozen_iter
    else:
        final_iterations = args.iterations // 2
    logger.info(
        "Running %d iterations of %d events each with the grid frozen",
        final_iterations,
        events_per_iteration,
    )

    if args.histograms:
        proc_name = args.madgraph_process.replace(" ", "_").replace(">", "to").replace("~", "b")
        with LheWriter(Path("."), proc_name, False, 0) as lhe_writer:
            integrand = generate_integrand(lhe_writer)
            vegas.compile(integrand)
            res, err = vegas.run_integration(final_iterations)
            lhe_writer.store_result((res, err))
            proc_folder = Path(f"Events/{proc_name}")
            logger.info("Written LHE file to %s", proc_folder)
    else:
        proc_folder = None
        res, err = vegas.run_integration(final_iterations)

    return args, (res, err), proc_folder


def main():
    flow_start = time.time()
    _ = madflow_main()
    flow_final = time.time()
    logger.info(f"> Madflow took: {flow_final-flow_start:.4}s")


if __name__ == "__main__":
    main()
