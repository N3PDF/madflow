"""
    Default settings for alohaFlow

    Since this program is to be installed together with VegasFlow
    it is allow to take the configuration from there
"""

from vegasflow.configflow import (
    run_eager,
    DTYPE,
    DTYPEINT,
    int_me,
    float_me,
    fzero,
    fone,
    ione,
    izero,
)

import tensorflow as tf

if DTYPE == tf.float64:
    DTYPECOMPLEX = tf.complex128
else:
    DTYPECOMPLEX = tf.complex64


def complex_tf(real, imag):
    """ Builds a tf.complex tensor from real and imaginary parts"""
    # print("complex tf")
    # python objects are stored with 32-bits, so cast first with float_me
    real = float_me(real)
    imag = float_me(imag)
    return tf.complex(real, imag)


def complex_me(cmp):
    """ Cast the input to complex type """
    # print("complex me")
    return tf.cast(cmp, dtype=DTYPECOMPLEX)
