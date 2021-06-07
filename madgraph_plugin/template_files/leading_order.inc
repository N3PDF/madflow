#!/usr/bin/env python3
"""
   MADFLOW AUTOMATIC TEMPLATE

   Automatic template for the process: (process)
   at Leading Order accuracy
"""

# The following settings are the default for this template
PDF_NAME = "NNPDF31_nnlo_as_0118"
SQRTS = 13000  # TeV
GLOBAL_MODEL_PATH = "models/sm"
PARAM_CARD = "Cards/param_card.dat"
EVENTS = int(1e4)
ITERATIONS = 3

import sys
import itertools
from pathlib import Path
from madflow.config import get_madgraph_path, int_me, run_eager, DTYPE
from madflow.phasespace import PhaseSpaceGenerator
from madflow.lhe_writer import LheWriter
from madflow.utilities import generate_initial_states
from pdfflow import mkPDF
from vegasflow import VegasFlow
from matrix_1_gg_ttx import Matrix_1_gg_ttx, get_model_param as model_1_gg_ttx
%(matrix_element_imports)s

import tensorflow as tf

# Get import_ufo from madgraph
import models.import_ufo as import_ufo

# Read the UFO model from the madgraph folder
model_sm = get_madgraph_path() / GLOBAL_MODEL_PATH
if not model_sm.exists():
    print(f"No model sm found at {model_sm}, test cannot continue")
    sys.exit(0)
global_model = import_ufo.import_model(model_sm.as_posix())

tree_level = {%(tree_level_keys)s}

final_state_masses = {%(masses)s}

matrices = []
models = []
for key, (matrix, get_model) in tree_level.items():
    model = get_model(global_model, PARAM_CARD)
    matrices.append(matrix())
    models.append(model)
nparticles = int(matrices[0].nexternal)

# Assumption for the template: all final state masses are equal
for m in final_state_masses.values():
    masses = [models[0].parse_parameter(i) for i in m[2:]]


# Instantiate the PDF
pdf = mkPDF(f"{PDF_NAME}/0")
# Read up all initial states that can appear in these matrices
initial_flavours = generate_initial_states(matrices)
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

# Instantiate the phase space
phasespace = PhaseSpaceGenerator(nparticles, SQRTS, masses, com_output=False)
for i in range(2, nparticles):
    # Apply a PT cut of 30 GeV to all outgoing particles
    phasespace.register_cut("pt", particle=i, min_val=30.0)


def luminosity_function(x1, x2, q2array):
    raw_proton_1 = pdf.xfxQ2(int_me(hadron_1), x1, q2array)
    raw_proton_2 = pdf.xfxQ2(int_me(hadron_2), x2, q2array)
    # Ensure they have the right shape, just in case
    proton_1 = tf.reshape(raw_proton_1, (-1, len(hadron_1)))
    proton_2 = tf.reshape(raw_proton_2, (-1, len(hadron_2)))
    return proton_1, proton_2


def cross_section_generation(lhewriter=None):
    """
    Function that generates a cross section to be integrated with
    or without a LHE event output
    """

    def cross_section(xrand, n_dim=None, weight=1.0):
        """Compute the cross section at LO for (process)"""

        # Generate the phase space point
        all_ps, wts, x1, x2, idx = phasespace(xrand)

        # Compute the renormalization and factorization scale = sum(mt)/2
        full_mt = tf.reduce_sum(phasespace.mt(all_ps[:, 2:nparticles, :]), axis=-1)
        q2array = (full_mt / 2.0) ** 2

        # Compute alpha_s at renormalization scale
        alpha_s = pdf.alphasQ2(q2array)

        # Compute the PDF at the factorization scale
        proton_1, proton_2 = luminosity_function(x1, x2, q2array)

        # Compute the weight for each matrix element
        ret = 0.0
        for i, (matrix, model) in enumerate(zip(matrices, models)):
            smatrix = matrix.smatrix(all_ps, *model.evaluate(alpha_s))
            p1 = tf.gather(proton_1, gather_1[i], axis=1)
            p2 = tf.gather(proton_2, gather_2[i], axis=1)
            # Sum all input channels together for now
            luminosity = tf.reduce_sum(p1 * p2, axis=1) / x1 / x2
            ret += luminosity * smatrix

        # Final cross section
        ret *= wts

        if lhewriter is not None:
            # Fill up the LHE grid
            weight = tf.gather(weight, idx)[:, 0]
            tf.py_function(func=lhewriter.lhe_parser, inp=[all_ps, ret * weight], Tout=DTYPE)

        return tf.scatter_nd(idx, ret, shape=xrand.shape[0:1])

    return cross_section


xs = cross_section_generation(None)
# Do a warmup of the integrator grid
ndim = (nparticles - 2) * 4 + 2
vegas = VegasFlow(ndim, EVENTS)
# Generate a cross section with no LHE writer for the warmup
vegas.compile(xs)
vegas.run_integration(ITERATIONS)

# Now output an event file
proc_name = Path("test")
with LheWriter(Path("."), proc_name, False, 0) as lhe_writer:
    xs_events = cross_section_generation(lhe_writer)
    vegas.compile(xs_events)
    res, err = vegas.run_integration(ITERATIONS)
    lhe_writer.store_result((res, err))
    proc_folder = Path(f"Events/{proc_name}")
    print(f"Written LHE file to {proc_folder}")
