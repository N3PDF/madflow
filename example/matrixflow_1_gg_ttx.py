from vegasflow.configflow import float_me, DTYPE, DTYPEINT
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
    # not sure if to use class attributes like this or instance attributes
    nexternal = float_me(4)
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
        # self.jamp = []

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
        # self.amp2 = [0.] * ndiags
        # self.helEvals = []
        ans = complex_tf([0.], [0.])
        for hel in self.helicities:
            t = self.matrix(all_ps, hel, mdl_MT, mdl_WT, GC_10, GC_11)
            ans += t
            # self.helEvals.append([hel, t.real / denominator ])
        ans /= self.denominator
        return tf.math.real(ans)

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
        ngraphs = 3
        nexternal = self.nexternal
        nwavefuncs = 5
        ncolor = 2
        ZERO = float_me(0.)
        #  
        # Color matrix
        #  
        denom = tf.constant([3,3], dtype=DTYPEINT)
        cf = tf.constant([[16,-2],
                          [-2,16]], dtype=DTYPEINT)
        # ----------
        # Begin code
        # ----------
        amp = tf.zeros([ngraphs], dtype=DTYPECOMPLEX)
        w = tf.zeros([nwavefuncs], dtype=DTYPECOMPLEX)

        # all_ps[:,i] selects the particle and is a [nevt,4] tensor
        w0 = vxxxxx(all_ps[:,0],ZERO,hel[0],-1) # [nevt,6]
        w1 = vxxxxx(all_ps[:,1],ZERO,hel[1],-1)
        w2 = oxxxxx(all_ps[:,2],mdl_MT,hel[2],+1)
        w3 = ixxxxx(all_ps[:,3],mdl_MT,hel[3],-1)
        print("vxxxxx shape", w1.shape)
        print("oxxxxx shape", w2.shape)
        print("ixxxxx shape", w3.shape)
        exit()
        w4= VVV1P0_1(w[0],w[1],GC_10,ZERO,ZERO)
        # Amplitude(s) for diagram number 1
        amp[0]= FFV1_0(w[3],w[2],w[4],GC_11)
        w[4]= FFV1_1(w[2],w[0],GC_11,mdl_MT,mdl_WT)
        # Amplitude(s) for diagram number 2
        amp[1]= FFV1_0(w[3],w[4],w[1],GC_11)
        w[4]= FFV1_2(w[3],w[0],GC_11,mdl_MT,mdl_WT)
        # Amplitude(s) for diagram number 3
        amp[2]= FFV1_0(w[4],w[2],w[1],GC_11)

        jamp = [None] * ncolor

        jamp[0] = +complex(0,1)*amp[0]-amp[1]
        jamp[1] = -complex(0,1)*amp[0]-amp[2]

        self.amp2[0]+=abs(amp[0]*amp[0].conjugate())
        self.amp2[1]+=abs(amp[1]*amp[1].conjugate())
        self.amp2[2]+=abs(amp[2]*amp[2].conjugate())
        matrix = 0.
        for i in range(ncolor):
            ztemp = 0
            for j in range(ncolor):
                ztemp = ztemp + cf[i][j]*jamp[j]
            matrix = matrix + ztemp * jamp[i].conjugate()/denom[i]   
        self.jamp.append(jamp)

        return matrix


if __name__ == "__main__":
    # model = import_ufo.import_model("/home/marco/PhD/unimi/MG5_aMC/MG5_aMC_v2_8_2/models/sm") 
    momenta = [[100., 0., 0., 100.],
               [100., 0., 0.,-100.],
               [100., 100., 0., 0.],
               [100.,-100., 0., 0.]]

    mymatrix = MatrixFlow_1_gg_ttx()
    print('RESULT', mymatrix.smatrix(momenta, model))
    