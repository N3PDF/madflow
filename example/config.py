import tensorflow as tf
from vegasflow.configflow import float_me, DTYPE, DTYPEINT
if DTYPE == tf.float64:
    DTYPECOMPLEX = tf.complex128
else:
    DTYPECOMPLEX = tf.complex64


def complex_tf(real, imag):
    """ Builds a tf.complex tensor from real and imaginary parts"""
    # python objects are stored with 32-bits, so cast first with float_me
    real = float_me(real)
    imag = float_me(imag)
    return tf.complex(real, imag)

# TODO: check if this is really needed
def complex_me(cmp):
    """ Cast the input to complex type """
    return tf.complex(cmp.real, cmp.imag)