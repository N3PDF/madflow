import tensorflow as tf
from vegasflow.configflow import float_me, DTYPE, DTYPEINT
if DTYPE == tf.float64:
    DTYPECOMPLEX = tf.complex128
else:
    DTYPECOMPLEX = tf.complex64

# TODO: probably a None default value is not acceptable by tf.function
def complex_tf(real, imag=None):
    """ Builds a tf.complex tensor from real and imaginary parts"""
    # python objects are stored with 32-bits, so cast first with float_me
    real = float_me(real)
    if imag is None:
        imag = tf.zeros_like(real)
    else:
        imag = float_me(imag)
    return tf.complex(real, imag)


def complex_me(cmp):
    """ Cast the input to complex type """
    return tf.complex(cmp.real, cmp.imag)