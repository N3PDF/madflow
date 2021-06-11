
from madflow.config import (
    int_me,
    float_me,
    DTYPE,
    DTYPEINT,
    run_eager,
    DTYPECOMPLEX,
    complex_tf,
    complex_me
)
from madflow.wavefunctions_flow import oxxxxx, ixxxxx, vxxxxx, sxxxxx
from madflow.parameters import Model

import os
import sys
import numpy as np

import tensorflow as tf
import collections

ModelParamTupleConst = collections.namedtuple("constants", [%(paramnames_const)s])
ModelParamTupleFunc = collections.namedtuple("functions", [%(paramnames_func)s])

root_path = '%(root_path)s'
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'madgraph'))
sys.path.insert(0, os.path.join(root_path, 'aloha', 'template_files'))

import models.import_ufo as import_ufo
import models.check_param_card as param_card_reader

%(aloha_imports)s


def get_model_param(model, param_card_path):
    param_card = param_card_reader.ParamCard(param_card_path)
    %(model_parameters)s
    constants = ModelParamTupleConst(%(paramtuple_const)s)
    functions = ModelParamTupleFunc(%(paramtuple_func)s)
    return Model(constants, functions)



smatrix_signature = [
        tf.TensorSpec(shape=[None,%(nexternal)d,4], dtype=DTYPE),
        %(paramsignature_const)s
        %(paramsignature_func)s
        ]


matrix_signature = [
        tf.TensorSpec(shape=[None,%(nexternal)d,4], dtype=DTYPE),
        tf.TensorSpec(shape=[%(nexternal)d], dtype=DTYPE),
        %(paramsignature_const)s
        %(paramsignature_func)s
        ]


class Matrix_%(process_string)s(object):
    nexternal = float_me(%(nexternal)d)
    ndiags = float_me(%(ndiags)d)
    ncomb = float_me(%(ncomb)d)
    initial_states = [%(initial_states)s]
    mirror_initial_states = %(mirror)s
    %(helicity_lines)s
    %(den_factor_line)s

    def __init__(self):
        """define the object"""
        self.clean()

    def clean(self):
        pass
        ##self.jamp = []

    def __str__(self):
        return "%(process_string)s"

    @tf.function(input_signature=smatrix_signature)
    def smatrix(self,all_ps,%(params)s):
        #  
        %(info_lines)s
        # 
        # MadGraph5_aMC@NLO StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        %(process_lines)s
        #  
        # Clean additional output
        #
        ###self.clean()
        # ----------
        # BEGIN CODE
        # ----------
        nevts = tf.shape(all_ps, out_type=DTYPEINT)[0]
        ans = tf.zeros(nevts, dtype=DTYPECOMPLEX)
        for hel in self.helicities:
            t = self.matrix(all_ps,hel,%(params)s)
            ans = ans + t

        return tf.math.real(ans)/self.denominator

    @tf.function(input_signature=matrix_signature)
    def matrix(self,all_ps,hel,%(params)s):
        #  
        %(info_lines)s
        #
        # Returns amplitude squared summed/avg over colors
        # for the point with external lines W(0:6,NEXTERNAL)
        #
        %(process_lines)s
        #  
        #  
        # Process parameters
        #  
        ngraphs = %(ngraphs)d
        nwavefuncs = %(nwavefuncs)d
        ncolor = %(ncolor)d
        ZERO = float_me(0.)
        #  
        # Color matrix
        #  
        %(color_matrix_lines)s
        #
        # Model parameters
        #
        # ----------
        # Begin code
        # ----------
        %(helas_calls)s

        %(jamp_lines)s

        ##%(amp2_lines)s
        return tf.einsum("ie, ij, je -> e", jamp, cf, tf.math.conj(jamp)/tf.reshape(denom, (ncolor, 1)))



if __name__ == "__main__":
    import sys, pathlib
    import numpy as np

    # Get the name of the matrix in this file
    matrix_name = pathlib.Path(sys.argv[0]).stem.capitalize()
    matrix_class = globals()[matrix_name]

    # Instantiate the matrix
    matrix = matrix_class()

    # Read up the model
    model_sm = pathlib.Path(root_path) / "models/sm"
    if not model_sm.exists():
        print(f"No model sm found at {model_sm}, test cannot continue")
        sys.exit(0)
    model = import_ufo.import_model(model_sm.as_posix())
    model_params = get_model_param(model, 'Cards/param_card.dat')

    # Define th phase space
    # The structure asked by the matrix elements is
    #   (nevents, ndimensions, nparticles)
    # the 4 dimensions of the 4-momentum is expected as
    #   (E, px, py, pz)
    ndim = 4
    npar = int(matrix.nexternal)
    nevents = 2
    max_momentum = 7e3

    par_ax = 1
    dim_ax = 2

    # Now generate random outgoing particles in a com frame (last_p carries whatever momentum is needed to sum 0 )
    shape = [nevents, 0, 0]
    shape[par_ax] = npar - 3
    shape[dim_ax] = ndim - 1
    partial_out_p = tf.random.uniform(shape, minval=-max_momentum, maxval=max_momentum, dtype=DTYPE)
    last_p = -tf.reduce_sum(partial_out_p, keepdims=True, axis=par_ax)
    out_p = tf.concat([partial_out_p, last_p], axis=par_ax)

    if "mdl_MT" in dir(model_params):
        # TODO fill in the mass according to the particles
        out_m = tf.reshape((npar - 2) * [model_params.mdl_MT], (1, -1, 1))
    else:
        out_m = 0.0
    out_e = tf.sqrt(tf.reduce_sum(out_p ** 2, keepdims=True, axis=dim_ax) + out_m ** 2)
    outgoing_4m = tf.concat([out_e, out_p], axis=dim_ax)

    # Put all incoming momenta in the z axis (TODO: for now assume massless input)
    ea = tf.reduce_sum(out_e, axis=par_ax, keepdims=True) / 2
    zeros = tf.zeros_like(ea)
    inc_p1 = tf.concat([ea, zeros, zeros, ea], axis=dim_ax)
    inc_p2 = tf.concat([ea, zeros, zeros, -ea], axis=dim_ax)

    all_ps = tf.concat([inc_p1, inc_p2, outgoing_4m], axis=par_ax)

    model_params.freeze_alpha_s(0.118)
    wgt_set = matrix.smatrix(all_ps, *model_params.evaluate(None))

    print("All good!")
    for i, (p, wgt) in enumerate(zip(all_ps, wgt_set)):
        print(f"\n#{i} ME value: {wgt.numpy():.3e} for P set:\n{p.numpy()}")
