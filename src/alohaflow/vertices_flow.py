"""
    Tensorised vertex functions
"""

from .config import complex_tf, complex_me, DTYPECOMPLEX, DTYPE
import tensorflow as tf

vertex_signature = [
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
]

ffv1_signature = [
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[None, None], dtype=DTYPECOMPLEX),
    tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX),
]


@tf.function(input_signature=ffv1_signature)
def FFV1_0(F1, F2, V3, COUP):
    """
    Fermion fermion gluon vertex

    Parameters
    ----------
        F1: tf.tensor
            first fermion wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        F2: tf.tensor
            second fermion wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        V3: tf.tensor
            gluon wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        COUP: tf.tensor
            coupling, ( =dim[]), DTYPE dtype

    Returns
    -------
        vertex: tf.tensor
            vertex contribution ( =dim[nevt,])

    """
    # print("ffv10")
    COUP = complex_me(COUP)
    im = complex_tf(0, 1)
    TMP0 = F1[2] * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + im * (V3[4]))) + (
        F1[3] * (F2[4] * (V3[3] - im * (V3[4])) + F2[5] * (V3[2] - V3[5]))
        + (
            F1[4] * (F2[2] * (V3[2] - V3[5]) - F2[3] * (V3[3] + im * (V3[4])))
            + F1[5] * (F2[2] * (-V3[3] + im * (V3[4])) + F2[3] * (V3[2] + V3[5]))
        )
    )
    vertex = COUP * -im * TMP0
    return vertex


@tf.function(input_signature=vertex_signature)
def FFV1_1(F2, V3, COUP, M1, W1):
    """
    Fermion fermion gluon vertex

    Parameters
    ----------
        F1: tf.tensor
            first fermion wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        F2: tf.tensor
            second fermion wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        V3: tf.tensor
            gluon wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        COUP: tf.tensor
            coupling, ( =dim[]), DTYPE dtype
        M1: tf.tensor
            coupling, ( =dim[]), DTYPE dtype
        W1: tf.tensor
            coupling, ( =dim[]), DTYPE dtype

    Returns
    -------
        vertex: tf.tensor
            vertex contribution ( =dim[6,nevt])
    """
    # print("ffv11")
    COUP = complex_me(COUP)
    M1 = complex_me(M1)
    W1 = complex_me(W1)
    im = complex_tf(0, 1)
    F10 = F2[0] + V3[0]
    F11 = F2[1] + V3[1]
    P1 = [-tf.math.real(F10), -tf.math.real(F11), -tf.math.imag(F11), -tf.math.imag(F10)]
    P1 = complex_tf(tf.stack(P1, axis=0), 0.0)
    denom = COUP / (P1[0] ** 2 - P1[1] ** 2 - P1[2] ** 2 - P1[3] ** 2 - M1 * (M1 - im * W1))

    # TODO: look here because writing ..(im..) raises bad operand type for unary +
    # this could be a problem for the automation process
    F12 = (
        denom
        * im
        * (
            F2[2]
            * (
                P1[0] * (-V3[2] + V3[5])
                + (
                    P1[1] * (V3[3] - im * (V3[4]))
                    + (P1[2] * (im * (V3[3]) + V3[4]) + P1[3] * (-V3[2] + V3[5]))
                )
            )
            + (
                F2[3]
                * (
                    P1[0] * (V3[3] + im * (V3[4]))
                    + (
                        P1[1] * (-1) * (V3[2] + V3[5])
                        + (P1[2] * (-1) * (im * (V3[2] + V3[5])) + P1[3] * (V3[3] + im * (V3[4])))
                    )
                )
                + M1 * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + im * (V3[4])))
            )
        )
    )
    F13 = (
        denom
        * (-im)
        * (
            F2[2]
            * (
                P1[0] * (-V3[3] + im * (V3[4]))
                + (
                    P1[1] * (V3[2] - V3[5])
                    + (P1[2] * (-im * (V3[2]) + im * (V3[5])) + P1[3] * (V3[3] - im * (V3[4])))
                )
            )
            + (
                F2[3]
                * (
                    P1[0] * (V3[2] + V3[5])
                    + (
                        P1[1] * (-1) * (V3[3] + im * (V3[4]))
                        + (P1[2] * (im * (V3[3]) - V3[4]) - P1[3] * (V3[2] + V3[5]))
                    )
                )
                + M1 * (F2[4] * (-V3[3] + im * (V3[4])) + F2[5] * (-V3[2] + V3[5]))
            )
        )
    )
    F14 = (
        denom
        * (-im)
        * (
            F2[4]
            * (
                P1[0] * (V3[2] + V3[5])
                + (
                    P1[1] * (-V3[3] + im * (V3[4]))
                    + (P1[2] * (-1) * (im * (V3[3]) + V3[4]) - P1[3] * (V3[2] + V3[5]))
                )
            )
            + (
                F2[5]
                * (
                    P1[0] * (V3[3] + im * (V3[4]))
                    + (
                        P1[1] * (-V3[2] + V3[5])
                        + (P1[2] * (-im * (V3[2]) + im * (V3[5])) - P1[3] * (V3[3] + im * (V3[4])))
                    )
                )
                + M1 * (F2[2] * (-V3[2] + V3[5]) + F2[3] * (V3[3] + im * (V3[4])))
            )
        )
    )
    F15 = (
        denom
        * im
        * (
            F2[4]
            * (
                P1[0] * (-V3[3] + im * (V3[4]))
                + (
                    P1[1] * (V3[2] + V3[5])
                    + (P1[2] * (-1) * (im * (V3[2] + V3[5])) + P1[3] * (-V3[3] + im * (V3[4])))
                )
            )
            + (
                F2[5]
                * (
                    P1[0] * (-V3[2] + V3[5])
                    + (
                        P1[1] * (V3[3] + im * (V3[4]))
                        + (P1[2] * (-im * (V3[3]) + V3[4]) + P1[3] * (-V3[2] + V3[5]))
                    )
                )
                + M1 * (F2[2] * (-V3[3] + im * (V3[4])) + F2[3] * (V3[2] + V3[5]))
            )
        )
    )
    return tf.stack([F10, F11, F12, F13, F14, F15], axis=0)


@tf.function(input_signature=vertex_signature)
def FFV1_2(F1, V3, COUP, M2, W2):
    """
    Fermion fermion gluon vertex

    Parameters
    ----------
        F1: tf.tensor
            first fermion wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        F2: tf.tensor
            second fermion wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        V3: tf.tensor
            gluon wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        COUP: tf.tensor
            coupling, ( =dim[]), DTYPE dtype
        M2: tf.tensor
            coupling, ( =dim[]), DTYPE dtype
        W2: tf.tensor
            coupling, ( =dim[]), DTYPE dtype

    Returns
    -------
        vertex: tf.tensor
            vertex contribution ( =dim[6,nevt])
    """
    # print("ffv12")
    COUP = complex_me(COUP)
    M2 = complex_me(M2)
    W2 = complex_me(W2)
    im = complex_tf(0, 1)
    F20 = F1[0] + V3[0]
    F21 = F1[1] + V3[1]
    P2 = [-tf.math.real(F20), -tf.math.real(F21), -tf.math.imag(F21), -tf.math.imag(F20)]
    P2 = complex_tf(tf.stack(P2, axis=0), 0.0)
    denom = COUP / (P2[0] ** 2 - P2[1] ** 2 - P2[2] ** 2 - P2[3] ** 2 - M2 * (M2 - im * W2))
    F22 = (
        denom
        * im
        * (
            F1[2]
            * (
                P2[0] * (V3[2] + V3[5])
                + (
                    P2[1] * (-1) * (V3[3] + im * (V3[4]))
                    + (P2[2] * (im * (V3[3]) - V3[4]) - P2[3] * (V3[2] + V3[5]))
                )
            )
            + (
                F1[3]
                * (
                    P2[0] * (V3[3] - im * (V3[4]))
                    + (
                        P2[1] * (-V3[2] + V3[5])
                        + (P2[2] * (im * (V3[2]) - im * (V3[5])) + P2[3] * (-V3[3] + im * (V3[4])))
                    )
                )
                + M2 * (F1[4] * (V3[2] - V3[5]) + F1[5] * (-V3[3] + im * (V3[4])))
            )
        )
    )
    F23 = (
        denom
        * (-im)
        * (
            F1[2]
            * (
                P2[0] * (-1) * (V3[3] + im * (V3[4]))
                + (
                    P2[1] * (V3[2] + V3[5])
                    + (P2[2] * (im * (V3[2] + V3[5])) - P2[3] * (V3[3] + im * (V3[4])))
                )
            )
            + (
                F1[3]
                * (
                    P2[0] * (-V3[2] + V3[5])
                    + (
                        P2[1] * (V3[3] - im * (V3[4]))
                        + (P2[2] * (im * (V3[3]) + V3[4]) + P2[3] * (-V3[2] + V3[5]))
                    )
                )
                + M2 * (F1[4] * (V3[3] + im * (V3[4])) - F1[5] * (V3[2] + V3[5]))
            )
        )
    )
    F24 = (
        denom
        * (-im)
        * (
            F1[4]
            * (
                P2[0] * (-V3[2] + V3[5])
                + (
                    P2[1] * (V3[3] + im * (V3[4]))
                    + (P2[2] * (-im * (V3[3]) + V3[4]) + P2[3] * (-V3[2] + V3[5]))
                )
            )
            + (
                F1[5]
                * (
                    P2[0] * (V3[3] - im * (V3[4]))
                    + (
                        P2[1] * (-1) * (V3[2] + V3[5])
                        + (P2[2] * (im * (V3[2] + V3[5])) + P2[3] * (V3[3] - im * (V3[4])))
                    )
                )
                + M2 * (F1[2] * (-1) * (V3[2] + V3[5]) + F1[3] * (-V3[3] + im * (V3[4])))
            )
        )
    )
    F25 = (
        denom
        * im
        * (
            F1[4]
            * (
                P2[0] * (-1) * (V3[3] + im * (V3[4]))
                + (
                    P2[1] * (V3[2] - V3[5])
                    + (P2[2] * (im * (V3[2]) - im * (V3[5])) + P2[3] * (V3[3] + im * (V3[4])))
                )
            )
            + (
                F1[5]
                * (
                    P2[0] * (V3[2] + V3[5])
                    + (
                        P2[1] * (-V3[3] + im * (V3[4]))
                        + (P2[2] * (-1) * (im * (V3[3]) + V3[4]) - P2[3] * (V3[2] + V3[5]))
                    )
                )
                + M2 * (F1[2] * (V3[3] + im * (V3[4])) + F1[3] * (V3[2] - V3[5]))
            )
        )
    )
    return tf.stack([F20, F21, F22, F23, F24, F25], axis=0)


@tf.function(input_signature=vertex_signature)
def VVV1P0_1(V2, V3, COUP, M1, W1):
    """
    Triple gluon vertex

    Parameters
    ----------
        V2: tf.tensor
            first gluon wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        V3: tf.tensor
            second gluon wavefunction ( =dim[6,nevt]), DTYPECOMPLEX dtype
        COUP: tf.tensor
            coupling, ( =dim[]), DTYPE dtype
        M1:tf.tensor
            coupling, ( =dim[]), DTYPE dtype
        W1:tf.tensor
            coupling, ( =dim[]), DTYPE dtype
    Returns
    -------
        vertex: tf.tensor
            vertex contribution ( =dim[6,nevt])
    """
    # print("vvv1p01")
    COUP = complex_me(COUP)
    M1 = complex_me(M1)
    W1 = complex_me(W1)
    im = complex_tf(0, 1)
    P2 = [tf.math.real(V2[0]), tf.math.real(V2[1]), tf.math.imag(V2[1]), tf.math.imag(V2[0])]
    P2 = complex_tf(tf.stack(P2, axis=0), 0.0)
    P3 = [tf.math.real(V3[0]), tf.math.real(V3[1]), tf.math.imag(V3[1]), tf.math.imag(V3[0])]
    P3 = complex_tf(tf.stack(P3, axis=0), 0.0)
    V10 = V2[0] + V3[0]
    V11 = V2[1] + V3[1]
    P1 = [-tf.math.real(V10), -tf.math.real(V11), -tf.math.imag(V11), -tf.math.imag(V10)]
    P1 = complex_tf(tf.stack(P1, axis=0), 0.0)
    TMP1 = V3[2] * P1[0] - V3[3] * P1[1] - V3[4] * P1[2] - V3[5] * P1[3]
    TMP2 = V3[2] * P2[0] - V3[3] * P2[1] - V3[4] * P2[2] - V3[5] * P2[3]
    TMP3 = P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]
    TMP4 = V2[2] * P3[0] - V2[3] * P3[1] - V2[4] * P3[2] - V2[5] * P3[3]
    TMP5 = V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]
    denom = COUP / (P1[0] ** 2 - P1[1] ** 2 - P1[2] ** 2 - P1[3] ** 2 - M1 * (M1 - im * W1))
    V12 = denom * (
        TMP5 * (-im * (P2[0]) + im * (P3[0]))
        + (V2[2] * (-im * (TMP1) + im * (TMP2)) + V3[2] * (im * (TMP3) - im * (TMP4)))
    )
    V13 = denom * (
        TMP5 * (-im * (P2[1]) + im * (P3[1]))
        + (V2[3] * (-im * (TMP1) + im * (TMP2)) + V3[3] * (im * (TMP3) - im * (TMP4)))
    )
    V14 = denom * (
        TMP5 * (-im * (P2[2]) + im * (P3[2]))
        + (V2[4] * (-im * (TMP1) + im * (TMP2)) + V3[4] * (im * (TMP3) - im * (TMP4)))
    )
    V15 = denom * (
        TMP5 * (-im * (P2[3]) + im * (P3[3]))
        + (V2[5] * (-im * (TMP1) + im * (TMP2)) + V3[5] * (im * (TMP3) - im * (TMP4)))
    )
    return tf.stack([V10, V11, V12, V13, V14, V15], axis=0)
