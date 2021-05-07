"""
    Default settings for alohaFlow

    Since this program is to be installed together with VegasFlow
    it is allow to take the configuration from there
"""
import os
from pathlib import Path

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

import logging

module_name = __name__.split(".")[0]
logger = logging.getLogger(module_name)

DEFAULT_LOG_LEVEL = "2"
log_level_idx = os.environ.get("ALOHAFLOW_LOG_LEVEL", DEFAULT_LOG_LEVEL)
log_dict = {"0": logging.ERROR, "1": logging.WARNING, "2": logging.INFO, "3": logging.DEBUG}
bad_log_warning = None
if log_level_idx not in log_dict:
    bad_log_warning = log_level_idx
    log_level_idx = DEFAULT_LOG_LEVEL
log_level = log_dict[log_level_idx]

# Set level debug for development
logger.setLevel(log_level)
# Create a handler and format it
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_format = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# Now that the logging has been created, warn about the bad logging level
if bad_log_warning is not None:
    logger.warning(
        "Accepted log levels are: %s, received: %s", list(log_dict.keys()), bad_log_warning
    )
    logger.warning(f"Setting log level to its default value: {DEFAULT_LOG_LEVEL}")


def get_madgraph_path():
    madgraph_path = Path(os.environ.get("MADGRAPH_PATH", "../../../mg5amcnlo"))
    if not madgraph_path.exists():
        raise ValueError("Need a path for a madgraph installation")
    return madgraph_path
