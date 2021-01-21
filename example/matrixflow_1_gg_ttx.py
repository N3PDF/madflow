from vegasflow.configflow import int_me, float_me, DTYPE, DTYPEINT
import tensorflow as tf
import collections
ModelParamTuple = collections.namedtuple("Model", ["mdl_MT", "mdl_WT", "GC_10", "GC_11"])
from wavefunctions_flow import oxxxxx, ixxxxx, vxxxxx
from vertices_flow import FFV1_0, FFV1_1, FFV1_2, VVV1P0_1
from config import complex_tf, complex_me, DTYPECOMPLEX

def get_model_param(model):
    mdl_MT = model.get('parameter_dict')["mdl_MT"]
    mdl_WT = model.get('parameter_dict')["mdl_WT"]
    GC_10 = model.get('coupling_dict')["GC_10"]
    GC_11 = model.get('coupling_dict')["GC_11"]
    return ModelParamTuple(float_me(mdl_MT), float_me(mdl_WT),
                           complex_me(GC_10), complex_me(GC_11))


class Matrixflow_1_gg_ttx:
    # TODO: not sure if to use class attributes like this or instance attributes
    nexternal = int_me(4)
    ndiags = float_me(3)
    ncomb = float_me(16)
    helicities = float_me([
                           [-1,-1,-1,1],
                           [-1,-1,-1,-1],
                           [-1,-1,1,1],
                           [-1,-1,1,-1],
                           [-1,1,-1,1],
                           [-1,1,-1,-1],
                           [-1,1,1,1],
                           [-1,1,1,-1],
                           [1,-1,-1,1],
                           [1,-1,-1,-1],
                           [1,-1,1,1],
                           [1,-1,1,-1],
                           [1,1,-1,1],
                           [1,1,-1,-1],
                           [1,1,1,1],
                           [1,1,1,-1]
                         ])
    denominator = float_me(256)
    def __init__(self):
        # drop for the moment some list attributes not useful for this MWE:
        # self.jamp, self.helEvals, self.amp2
        self.clean()
        
    def clean(self):
        pass

    def smatrix(self, all_ps, mdl_MT, mdl_WT, GC_10, GC_11):
        """
        Given the vector of all phase space points and the relevant model
        parameters, this function returns the amplitude squared summed/avg over
        colors and helicities.

        Note: to automatically wrap and pass the model parameters, just call
            `get_model_param(model)` in the main file and pass its output as
            *args to this function like: `smatrix(all_ps, *args)`.

        Process: g g > t t~ WEIGHTED<=2 @1

        Parameters
        ----------
            all_ps: tf.tensor of shape [None, None, 4]
                query values of ps points (nevents x nparticles x 4)
            mdl_MT: tf.tensor of shape [1]
                model parameter
            mdl_WT: tf.tensor of shape [1]
                model parameter
            GC_10: tf.tensor of shape [1]
                model parameter
            GC_11: tf.tensor of shape [1]
                model parameter
        
        Returns
        -------
            tf.tensor of shape [None]
                tensor of smatrix real parts
        """
        # TODO: is it possible to skip the helicity for and parallelize it ?
        # TODO: check if the cast
        ans = complex_tf(0., 0.)
        for hel in self.helicities:
            t = self.matrix(all_ps, hel, mdl_MT, mdl_WT, GC_10, GC_11)
            ans += t
        return tf.math.real(ans)/self.denominator

    def matrix(self, all_ps, hel, mdl_MT, mdl_WT, GC_10, GC_11):
        """
        Returns amplitude squared summed/avg over colors for given phase space
        point and helicity in the current model.

        Parameters
        ----------
            all_ps: tf.tensor of shape [None, None, None]
                query values of ps points ( =dim (nevents, nexternal, 4))
            hel: tf.tensor of shape [4,]
                helicity values for the external particles
            mdl_MT: tf.tensor of shape [1]
                model parameter
            mdl_WT: tf.tensor of shape [1]
                model parameter
            GC_10: tf.tensor of shape [1]
                model parameter
            GC_11: tf.tensor of shape [1]
                model parameter
        
        Returns
        -------
            matrix: tf.tensor of shape [None]
                tensor of amplitudes squared
        
        """
        # TODO: check if replacing all the lists elements with single tensors is
        # good or not for the automation. For example:
        # jamp = [None] * color
        # jamp[0] = ...
        # has been replaced by
        # jamp0 = ...
        # Problem is that tf doesn't support tensor item assignment. Look into 
        # TensorArrays could be a way out ?
        ngraphs = 3
        nexternal = self.nexternal
        nwavefuncs = 5
        ncolor = 2
        ZERO = float_me(0.)
        #  
        # Color matrix
        #  
        denom = tf.constant([3,3], dtype=DTYPECOMPLEX)
        cf = tf.constant([[16,-2],
                          [-2,16]], dtype=DTYPECOMPLEX)
        # ----------
        # Begin code
        # ----------
        # all_ps[:,i] selects the particle and is a [nevt,4] tensor
        # wavefunctions output a [None,nevt] tensor based on spine
        # amplitudes are [nevt] tensors
        w0 = vxxxxx(all_ps[:,0],ZERO,hel[0],-1)
        w1 = vxxxxx(all_ps[:,1],ZERO,hel[1],-1)
        w2 = oxxxxx(all_ps[:,2],mdl_MT,hel[2],+1)
        w3 = ixxxxx(all_ps[:,3],mdl_MT,hel[3],-1)
        w4= VVV1P0_1(w0,w1,GC_10,ZERO,ZERO)
        # Amplitude(s) for diagram number 1
        amp0= FFV1_0(w3,w2,w4,GC_11)
        w4= FFV1_1(w2,w0,GC_11,mdl_MT,mdl_WT)
        # Amplitude(s) for diagram number 2
        amp1= FFV1_0(w3,w4,w1,GC_11)
        w4= FFV1_2(w3,w0,GC_11,mdl_MT,mdl_WT)
        # Amplitude(s) for diagram number 3
        amp2= FFV1_0(w4,w2,w1,GC_11)

        jamp0 = complex_tf(0,1)*amp0-amp1
        jamp1 = -complex_tf(0,1)*amp0-amp2
        jamp = tf.stack([jamp0,jamp1], axis=0)

        matrix = complex_tf(0,0)
        for i in tf.range(ncolor):
            ztemp = complex_tf(0,0)
            for j in tf.range(ncolor):
                ztemp = ztemp + cf[i,j]*jamp[j]
            matrix = matrix + ztemp * tf.math.conj(jamp[i])/denom[i]   
        return matrix


if __name__ == "__main__":
    matrix_elm_folder = "../../vegasflow_example/"
    model = None
    base_model = "models/sm"

    import sys
    import glob
    import copy
    import importlib.util
    import re
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
    model = import_ufo.import_model(f"{root_path}/{base_model}")
    model_params = get_model_param(model)

    # Clean the path
    sys.path = original_path

    import numpy as np
    COM_SQRTS = 7e3
    nevt = 1000
    ndim = 4*matrixflow.nexternal
    xrand = float_me(np.random.rand(nevt,ndim)*100)
    from parallel_rambo import parallel_rambo
    def phasespace_generator(xrand, nparticles):
        """ Takes as input an array of nevent x ndim random points and outputs
        an array of momenta (nevents x nparticles x 4)
        """
        return parallel_rambo(xrand, 4, COM_SQRTS)
    all_momenta, _ = phasespace_generator(xrand, 4)
    resflow = matrixflow.smatrix(all_momenta, *model_params)
    res = []
    for momenta in all_momenta.numpy():
        res.append(matrix.smatrix(momenta, model))
    res = np.stack(res) # [nevt,]

    print("Checking smatrix results...")
    np.testing.assert_allclose(res, resflow, rtol=1e-4)
    print("All good!")
