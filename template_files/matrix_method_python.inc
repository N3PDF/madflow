
from alohaflow.config import (
    int_me,
    float_me,
    DTYPE,
    DTYPEINT,
    run_eager,
    DTYPECOMPLEX,
    complex_tf,
    complex_me
)
from alohaflow.wavefunctions_flow import oxxxxx, ixxxxx, vxxxxx, sxxxxx

import os
import sys

import tensorflow as tf
import collections

ModelParamTuple = collections.namedtuple("Model", [%(paramnames)s])

root_path = '%(root_path)s'
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'madgraph'))
sys.path.insert(0, os.path.join(root_path, 'aloha', 'template_files'))

import models.import_ufo as import_ufo

%(aloha_imports)s


def get_model_param(model):
    %(model_parameters)s
    return ModelParamTuple(%(paramtuple)s)



smatrix_signature = [
        tf.TensorSpec(shape=[None,None,4], dtype=DTYPE),
        %(paramsignature)s
        ]


matrix_signature = [
        tf.TensorSpec(shape=[None,None,4], dtype=DTYPE),
        tf.TensorSpec(shape=[4], dtype=DTYPE),
        %(paramsignature)s
        ]


class Matrix_%(process_string)s(object):
    nexternal = float_me(%(nexternal)d)
    ndiags = float_me(%(ndiags)d)
    ncomb = float_me(%(ncomb)d)
    %(helicity_lines)s
    %(den_factor_line)s

    def __init__(self):
        """define the object"""
        self.clean()

    def clean(self):
        pass
        ##self.jamp = []

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
        nexternal = self.nexternal
        nwavefuncs = %(nwavefuncs)d
        ncolor = %(ncolor)d
        ZERO = float_me(0.)
        nevts = tf.shape(all_ps, out_type=DTYPEINT)[0]
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
        amp = [None] * ngraphs
        w = [None] * nwavefuncs
        %(helas_calls)s

        %(jamp_lines)s

        ##%(amp2_lines)s
        matrix = tf.zeros(nevts, dtype=DTYPECOMPLEX)
        for i in range(ncolor):
            ztemp = tf.zeros(nevts, dtype=DTYPECOMPLEX)
            for j in range(ncolor):
                ztemp = ztemp + cf[i][j]*jamp[j]
            matrix = matrix + ztemp * tf.math.conj(jamp[i])/denom[i]   
        #self.jamp.append(jamp)

        return matrix



if __name__ == "__main__":
    matrix_elm_folder = "../../vegasflow_example/"
    model = None

    import sys
    import glob
    import copy
    import importlib.util
    import re
    from time import time as tm
    re_name = re.compile("\w{3,}")
    original_path = copy.copy(sys.path)
    sys.path.insert(0, matrix_elm_folder)
    matrix_file = f"{matrix_elm_folder}/matrix_1_gg_ttx.py"
    matrix_name = re_name.findall(matrix_file)[-1]
    class_name = matrix_name.capitalize()
    module_spec = importlib.util.spec_from_file_location(matrix_name, matrix_file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    
    matrix = getattr(module, class_name)()
    matrixflow = Matrixflow_1_gg_ttx()
    root_path = getattr(module, "root_path")
    import_ufo = getattr(module, "import_ufo")
    model = import_ufo.import_model("%(model_path)s") 
    model_params = get_model_param(model)

    # Clean the path
    sys.path = original_path
    # run_eager(True)

    import numpy as np
    COM_SQRTS = 7e3
    nevt = 100000
    ndim = 4*matrixflow.nexternal
    xrand = float_me(np.random.rand(nevt,ndim)*100)
    from parallel_rambo import parallel_rambo
    def phasespace_generator(xrand, nparticles):
        """ Takes as input an array of nevent x ndim random points and outputs
        an array of momenta (nevents x nparticles x 4)
        """
        return parallel_rambo(xrand, 4, COM_SQRTS)
    all_momenta, _ = phasespace_generator(xrand, 4)
    print("Tracing TGraph ...")
    print(f"Evaluating smatrix for {nevt} events")
    start = tm()
    matrixflow.smatrix(all_momenta[:1], *model_params)
    print(f"TGraph traced in {tm()-start} s")
    start = tm()
    resflow = matrixflow.smatrix(all_momenta, *model_params)
    print(f"Mg5Flow computed smatrix in: {tm()-start} s")
    start = tm()
    res = []
    for momenta in all_momenta.numpy():
        res.append(matrix.smatrix(momenta, model))
    res = np.stack(res) # [nevt,]
    print(f"Mg5 computed smatrix in: {tm()-start}")

    print("Checking smatrix results...")
    np.testing.assert_allclose(res, resflow, rtol=1e-4)
    print("All good!")
