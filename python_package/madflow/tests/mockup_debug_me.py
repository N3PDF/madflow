"""
    Mockup matrix element generated with commit e3b67c4ae2048909907841f58346b55c5b00c7a0
    from pyout for a regresion test
    The expected result for this integration is of 103.4 +- 0.1 with n=1e5
"""

import collections
from madflow.config import (
    int_me,
    float_me,
    DTYPE,
    DTYPEINT,
    run_eager,
    DTYPECOMPLEX,
    complex_tf,
    complex_me,
)
from madflow.wavefunctions_flow import oxxxxx, ixxxxx, vxxxxx, sxxxxx
import tensorflow as tf

ModelParamTuple = collections.namedtuple("Model", ["mdl_MT", "mdl_WT", "GC_10", "GC_11"])
mdl_MT = 173.0
mdl_WT = 1.4915000200271606
GC_10 = complex_me(-1.2177157847767195 - 0j)
GC_11 = complex_me(1.2177157847767195j)
model_params = (float_me(mdl_MT), float_me(mdl_WT), complex_me(GC_10), complex_me(GC_11))


#######################################################
# Concatenation of the output of pyout for g g > t t~ #
#######################################################


FFV1_0_signature = [
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX),
]


@tf.function(input_signature=FFV1_0_signature)
def FFV1_0(F1, F2, V3, COUP):
    cI = complex_tf(0, 1)
    COUP = complex_me(COUP)
    TMP0 = F1[2] * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI * (V3[4]))) + (
        F1[3] * (F2[4] * (V3[3] - cI * (V3[4])) + F2[5] * (V3[2] - V3[5]))
        + (
            F1[4] * (F2[2] * (V3[2] - V3[5]) - F2[3] * (V3[3] + cI * (V3[4])))
            + F1[5] * (F2[2] * (-V3[3] + cI * (V3[4])) + F2[3] * (V3[2] + V3[5]))
        )
    )
    vertex = COUP * -cI * TMP0
    return vertex


FFV1_1_signature = [
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
]


@tf.function(input_signature=FFV1_1_signature)
def FFV1_1(F2, V3, COUP, M1, W1):
    cI = complex_tf(0, 1)
    COUP = complex_me(COUP)
    M1 = complex_me(M1)
    W1 = complex_me(W1)
    F1 = [complex_tf(0, 0)] * 6
    F1[0] = F2[0] + V3[0]
    F1[1] = F2[1] + V3[1]
    P1 = complex_tf(
        tf.stack(
            [
                -tf.math.real(F1[0]),
                -tf.math.real(F1[1]),
                -tf.math.imag(F1[1]),
                -tf.math.imag(F1[0]),
            ],
            axis=0,
        ),
        0.0,
    )
    denom = COUP / (P1[0] ** 2 - P1[1] ** 2 - P1[2] ** 2 - P1[3] ** 2 - M1 * (M1 - cI * W1))
    F1[2] = (
        denom
        * cI
        * (
            F2[2]
            * (
                P1[0] * (-V3[2] + V3[5])
                + (
                    P1[1] * (V3[3] - cI * (V3[4]))
                    + (P1[2] * (cI * (V3[3]) + V3[4]) + P1[3] * (-V3[2] + V3[5]))
                )
            )
            + (
                F2[3]
                * (
                    P1[0] * (V3[3] + cI * (V3[4]))
                    + (
                        P1[1] * (-1.0 / 1.0) * (V3[2] + V3[5])
                        + (
                            P1[2] * (-1.0 / 1.0) * (cI * (V3[2] + V3[5]))
                            + P1[3] * (V3[3] + cI * (V3[4]))
                        )
                    )
                )
                + M1 * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI * (V3[4])))
            )
        )
    )
    F1[3] = (
        denom
        * (-cI)
        * (
            F2[2]
            * (
                P1[0] * (-V3[3] + cI * (V3[4]))
                + (
                    P1[1] * (V3[2] - V3[5])
                    + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) + P1[3] * (V3[3] - cI * (V3[4])))
                )
            )
            + (
                F2[3]
                * (
                    P1[0] * (V3[2] + V3[5])
                    + (
                        P1[1] * (-1.0 / 1.0) * (V3[3] + cI * (V3[4]))
                        + (P1[2] * (cI * (V3[3]) - V3[4]) - P1[3] * (V3[2] + V3[5]))
                    )
                )
                + M1 * (F2[4] * (-V3[3] + cI * (V3[4])) + F2[5] * (-V3[2] + V3[5]))
            )
        )
    )
    F1[4] = (
        denom
        * (-cI)
        * (
            F2[4]
            * (
                P1[0] * (V3[2] + V3[5])
                + (
                    P1[1] * (-V3[3] + cI * (V3[4]))
                    + (P1[2] * (-1.0 / 1.0) * (cI * (V3[3]) + V3[4]) - P1[3] * (V3[2] + V3[5]))
                )
            )
            + (
                F2[5]
                * (
                    P1[0] * (V3[3] + cI * (V3[4]))
                    + (
                        P1[1] * (-V3[2] + V3[5])
                        + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) - P1[3] * (V3[3] + cI * (V3[4])))
                    )
                )
                + M1 * (F2[2] * (-V3[2] + V3[5]) + F2[3] * (V3[3] + cI * (V3[4])))
            )
        )
    )
    F1[5] = (
        denom
        * cI
        * (
            F2[4]
            * (
                P1[0] * (-V3[3] + cI * (V3[4]))
                + (
                    P1[1] * (V3[2] + V3[5])
                    + (
                        P1[2] * (-1.0 / 1.0) * (cI * (V3[2] + V3[5]))
                        + P1[3] * (-V3[3] + cI * (V3[4]))
                    )
                )
            )
            + (
                F2[5]
                * (
                    P1[0] * (-V3[2] + V3[5])
                    + (
                        P1[1] * (V3[3] + cI * (V3[4]))
                        + (P1[2] * (-cI * (V3[3]) + V3[4]) + P1[3] * (-V3[2] + V3[5]))
                    )
                )
                + M1 * (F2[2] * (-V3[3] + cI * (V3[4])) + F2[3] * (V3[2] + V3[5]))
            )
        )
    )
    return tf.stack(F1, axis=0)


FFV1_2_signature = [
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
]


@tf.function(input_signature=FFV1_2_signature)
def FFV1_2(F1, V3, COUP, M2, W2):
    cI = complex_tf(0, 1)
    COUP = complex_me(COUP)
    M2 = complex_me(M2)
    W2 = complex_me(W2)
    F2 = [complex_tf(0, 0)] * 6
    F2[0] = F1[0] + V3[0]
    F2[1] = F1[1] + V3[1]
    P2 = complex_tf(
        tf.stack(
            [
                -tf.math.real(F2[0]),
                -tf.math.real(F2[1]),
                -tf.math.imag(F2[1]),
                -tf.math.imag(F2[0]),
            ],
            axis=0,
        ),
        0.0,
    )
    denom = COUP / (P2[0] ** 2 - P2[1] ** 2 - P2[2] ** 2 - P2[3] ** 2 - M2 * (M2 - cI * W2))
    F2[2] = (
        denom
        * cI
        * (
            F1[2]
            * (
                P2[0] * (V3[2] + V3[5])
                + (
                    P2[1] * (-1.0 / 1.0) * (V3[3] + cI * (V3[4]))
                    + (P2[2] * (cI * (V3[3]) - V3[4]) - P2[3] * (V3[2] + V3[5]))
                )
            )
            + (
                F1[3]
                * (
                    P2[0] * (V3[3] - cI * (V3[4]))
                    + (
                        P2[1] * (-V3[2] + V3[5])
                        + (P2[2] * (cI * (V3[2]) - cI * (V3[5])) + P2[3] * (-V3[3] + cI * (V3[4])))
                    )
                )
                + M2 * (F1[4] * (V3[2] - V3[5]) + F1[5] * (-V3[3] + cI * (V3[4])))
            )
        )
    )
    F2[3] = (
        denom
        * (-cI)
        * (
            F1[2]
            * (
                P2[0] * (-1.0 / 1.0) * (V3[3] + cI * (V3[4]))
                + (
                    P2[1] * (V3[2] + V3[5])
                    + (P2[2] * (cI * (V3[2] + V3[5])) - P2[3] * (V3[3] + cI * (V3[4])))
                )
            )
            + (
                F1[3]
                * (
                    P2[0] * (-V3[2] + V3[5])
                    + (
                        P2[1] * (V3[3] - cI * (V3[4]))
                        + (P2[2] * (cI * (V3[3]) + V3[4]) + P2[3] * (-V3[2] + V3[5]))
                    )
                )
                + M2 * (F1[4] * (V3[3] + cI * (V3[4])) - F1[5] * (V3[2] + V3[5]))
            )
        )
    )
    F2[4] = (
        denom
        * (-cI)
        * (
            F1[4]
            * (
                P2[0] * (-V3[2] + V3[5])
                + (
                    P2[1] * (V3[3] + cI * (V3[4]))
                    + (P2[2] * (-cI * (V3[3]) + V3[4]) + P2[3] * (-V3[2] + V3[5]))
                )
            )
            + (
                F1[5]
                * (
                    P2[0] * (V3[3] - cI * (V3[4]))
                    + (
                        P2[1] * (-1.0 / 1.0) * (V3[2] + V3[5])
                        + (P2[2] * (cI * (V3[2] + V3[5])) + P2[3] * (V3[3] - cI * (V3[4])))
                    )
                )
                + M2 * (F1[2] * (-1.0 / 1.0) * (V3[2] + V3[5]) + F1[3] * (-V3[3] + cI * (V3[4])))
            )
        )
    )
    F2[5] = (
        denom
        * cI
        * (
            F1[4]
            * (
                P2[0] * (-1.0 / 1.0) * (V3[3] + cI * (V3[4]))
                + (
                    P2[1] * (V3[2] - V3[5])
                    + (P2[2] * (cI * (V3[2]) - cI * (V3[5])) + P2[3] * (V3[3] + cI * (V3[4])))
                )
            )
            + (
                F1[5]
                * (
                    P2[0] * (V3[2] + V3[5])
                    + (
                        P2[1] * (-V3[3] + cI * (V3[4]))
                        + (P2[2] * (-1.0 / 1.0) * (cI * (V3[3]) + V3[4]) - P2[3] * (V3[2] + V3[5]))
                    )
                )
                + M2 * (F1[2] * (V3[3] + cI * (V3[4])) + F1[3] * (V3[2] - V3[5]))
            )
        )
    )
    return tf.stack(F2, axis=0)


VVV1P0_1_signature = [
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
]


@tf.function(input_signature=VVV1P0_1_signature)
def VVV1P0_1(V2, V3, COUP, M1, W1):
    cI = complex_tf(0, 1)
    COUP = complex_me(COUP)
    M1 = complex_me(M1)
    W1 = complex_me(W1)
    P2 = complex_tf(
        tf.stack(
            [tf.math.real(V2[0]), tf.math.real(V2[1]), tf.math.imag(V2[1]), tf.math.imag(V2[0])],
            axis=0,
        ),
        0.0,
    )
    P3 = complex_tf(
        tf.stack(
            [tf.math.real(V3[0]), tf.math.real(V3[1]), tf.math.imag(V3[1]), tf.math.imag(V3[0])],
            axis=0,
        ),
        0.0,
    )
    V1 = [complex_tf(0, 0)] * 6
    V1[0] = V2[0] + V3[0]
    V1[1] = V2[1] + V3[1]
    P1 = complex_tf(
        tf.stack(
            [
                -tf.math.real(V1[0]),
                -tf.math.real(V1[1]),
                -tf.math.imag(V1[1]),
                -tf.math.imag(V1[0]),
            ],
            axis=0,
        ),
        0.0,
    )
    TMP1 = V3[2] * P1[0] - V3[3] * P1[1] - V3[4] * P1[2] - V3[5] * P1[3]
    TMP2 = V3[2] * P2[0] - V3[3] * P2[1] - V3[4] * P2[2] - V3[5] * P2[3]
    TMP3 = P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]
    TMP4 = V2[2] * P3[0] - V2[3] * P3[1] - V2[4] * P3[2] - V2[5] * P3[3]
    TMP5 = V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]
    denom = COUP / (P1[0] ** 2 - P1[1] ** 2 - P1[2] ** 2 - P1[3] ** 2 - M1 * (M1 - cI * W1))
    V1[2] = denom * (
        TMP5 * (-cI * (P2[0]) + cI * (P3[0]))
        + (V2[2] * (-cI * (TMP1) + cI * (TMP2)) + V3[2] * (cI * (TMP3) - cI * (TMP4)))
    )
    V1[3] = denom * (
        TMP5 * (-cI * (P2[1]) + cI * (P3[1]))
        + (V2[3] * (-cI * (TMP1) + cI * (TMP2)) + V3[3] * (cI * (TMP3) - cI * (TMP4)))
    )
    V1[4] = denom * (
        TMP5 * (-cI * (P2[2]) + cI * (P3[2]))
        + (V2[4] * (-cI * (TMP1) + cI * (TMP2)) + V3[4] * (cI * (TMP3) - cI * (TMP4)))
    )
    V1[5] = denom * (
        TMP5 * (-cI * (P2[3]) + cI * (P3[3]))
        + (V2[5] * (-cI * (TMP1) + cI * (TMP2)) + V3[5] * (cI * (TMP3) - cI * (TMP4)))
    )
    return tf.stack(V1, axis=0)


smatrix_signature = [
    tf.TensorSpec(shape=[None, None, 4], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX),
]


matrix_signature = [
    tf.TensorSpec(shape=[None, None, 4], dtype=DTYPE),
    tf.TensorSpec(shape=[4], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX),
]


class Matrix_1_gg_ttx(object):
    nexternal = float_me(4)
    ndiags = float_me(3)
    ncomb = float_me(16)
    helicities = float_me(
        [
            [-1, -1, -1, 1],
            [-1, -1, -1, -1],
            [-1, -1, 1, 1],
            [-1, -1, 1, -1],
            [-1, 1, -1, 1],
            [-1, 1, -1, -1],
            [-1, 1, 1, 1],
            [-1, 1, 1, -1],
            [1, -1, -1, 1],
            [1, -1, -1, -1],
            [1, -1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, 1],
            [1, 1, -1, -1],
            [1, 1, 1, 1],
            [1, 1, 1, -1],
        ]
    )
    denominator = float_me(256)

    def __init__(self):
        """define the object"""
        self.clean()

    def clean(self):
        pass
        ##self.jamp = []

    @tf.function(input_signature=smatrix_signature)
    def smatrix(self, all_ps, mdl_MT, mdl_WT, GC_10, GC_11):
        #
        #  MadGraph5_aMC@NLO v. 2.9.2, 2021-02-14
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        #
        # MadGraph5_aMC@NLO StandAlone Version
        #
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #
        # Process: g g > t t~ WEIGHTED<=2 @1
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
            t = self.matrix(all_ps, hel, mdl_MT, mdl_WT, GC_10, GC_11)
            ans = ans + t

        return tf.math.real(ans) / self.denominator

    @tf.function(input_signature=matrix_signature)
    def matrix(self, all_ps, hel, mdl_MT, mdl_WT, GC_10, GC_11):
        #
        #  MadGraph5_aMC@NLO v. 2.9.2, 2021-02-14
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        #
        # Returns amplitude squared summed/avg over colors
        # for the point with external lines W(0:6,NEXTERNAL)
        #
        # Process: g g > t t~ WEIGHTED<=2 @1
        #
        #
        # Process parameters
        #
        ngraphs = 3
        nexternal = self.nexternal
        nwavefuncs = 5
        ncolor = 2
        ZERO = float_me(0.0)
        nevts = tf.shape(all_ps, out_type=DTYPEINT)[0]
        #
        # Color matrix
        #
        denom = tf.constant([3, 3], dtype=DTYPECOMPLEX)
        cf = tf.constant([[16, -2], [-2, 16]], dtype=DTYPECOMPLEX)
        #
        # Model parameters
        #
        # ----------
        # Begin code
        # ----------
        amp = [None] * ngraphs
        w = [None] * nwavefuncs
        w0 = vxxxxx(all_ps[:, 0], ZERO, hel[0], float_me(-1))
        w1 = vxxxxx(all_ps[:, 1], ZERO, hel[1], float_me(-1))
        w2 = oxxxxx(all_ps[:, 2], mdl_MT, hel[2], float_me(+1))
        w3 = ixxxxx(all_ps[:, 3], mdl_MT, hel[3], float_me(-1))
        w4 = VVV1P0_1(w0, w1, GC_10, ZERO, ZERO)
        # Amplitude(s) for diagram number 1
        amp0 = FFV1_0(w3, w2, w4, GC_11)
        w4 = FFV1_1(w2, w0, GC_11, mdl_MT, mdl_WT)
        # Amplitude(s) for diagram number 2
        amp1 = FFV1_0(w3, w4, w1, GC_11)
        w4 = FFV1_2(w3, w0, GC_11, mdl_MT, mdl_WT)
        # Amplitude(s) for diagram number 3
        amp2 = FFV1_0(w4, w2, w1, GC_11)

        jamp = tf.stack([complex_tf(0, 1) * amp0 - amp1, -complex(0, 1) * amp0 - amp2], axis=0)

        ##self.amp2[0]+=abs(amp[0]*amp[0].conjugate())
        # self.amp2[1]+=abs(amp[1]*amp[1].conjugate())
        # self.amp2[2]+=abs(amp[2]*amp[2].conjugate())
        matrix = tf.zeros(nevts, dtype=DTYPECOMPLEX)
        for i in range(ncolor):
            ztemp = tf.zeros(nevts, dtype=DTYPECOMPLEX)
            for j in range(ncolor):
                ztemp = ztemp + cf[i][j] * jamp[j]
            matrix = matrix + ztemp * tf.math.conj(jamp[i]) / denom[i]
        # self.jamp.append(jamp)

        return matrix
