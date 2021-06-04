"""
    Default settings for madflow

    This program is to be installed alongside VegasFlow and
    PDFFlow so we'll take the main configuration from there.
    The environment variables for log levels and float/int types
    are propagated to any other programs used by madflow
"""
import os
import logging
from pathlib import Path

# Read the madfflow environment variables
_log_level_idx = os.environ.get("MADFLOW_LOG_LEVEL")
_float_env = os.environ.get("MADFLOW_FLOAT", "64")
_int_env = os.environ.get("MADFLOW_INT", "32")

# Ensure that both vegasflow and pdfflow are consistent
# with the corresponding madflow choice
# For float/int the consistency is enforced, logs can be chosen
# differently for each program

if _log_level_idx is None:
    _log_level_idx = "2"
else:
    os.environ.setdefault("PDFFLOW_LOG_LEVEL", _log_level_idx)
    os.environ.setdefault("VEGASFLOW_LOG_LEVEL", _log_level_idx)

os.environ["VEGASFLOW_FLOAT"] = _float_env
os.environ["PDFFLOW_FLOAT"] = _float_env
os.environ["VEGASFLOW_INT"] = _int_env
os.environ["PDFFLOW_INT"] = _int_env

# Now import all functions and variables directly from one of the other programs
from pdfflow.configflow import (
    LOG_DICT,
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

# Configure logging
_log_level = LOG_DICT["2"]
logger = logging.getLogger(__name__.split(".")[0])
logger.setLevel(_log_level)

# Create and format the log handler
_console_handler = logging.StreamHandler()
_console_handler.setLevel(_log_level)
_console_format = logging.Formatter("[%(levelname)s] (%(name)s) %(message)s")
_console_handler.setFormatter(_console_format)
logger.addHandler(_console_handler)

import tensorflow as tf


def complex_tf(real, imag):
    """Builds a tf.complex tensor from real and imaginary parts"""
    # python objects are stored with 32-bits, so cast first with float_me
    real = float_me(real)
    imag = float_me(imag)
    return tf.complex(real, imag)


DTYPECOMPLEX = complex_tf(1.0, 1.0).dtype


def complex_me(cmp):
    """Cast the input to complex type"""
    return tf.cast(cmp, dtype=DTYPECOMPLEX)


def get_madgraph_path(madpath=None):
    """ Return the path to the madgrapt root """
    if madpath is None:
        madpath = os.environ.get("MADGRAPH_PATH", "../../../mg5amcnlo")
    madgraph_path = Path(madpath)
    if not madgraph_path.exists():
        raise ValueError(
            f"{madgraph_path} does not exist. "
            "Needs a valid path for Madgraph, can be given as env. variable MADGRAPH_PATH"
        )
    return madgraph_path


def get_madgraph_exe(madpath=None):
    """ Return the path to the madgraph executable """
    madpath = get_madgraph_path(madpath)
    mg5_exe = mg5_path / "bin/mg5_aMC"
    if not mg5_exe.exists():
        raise ValueError(f"Madgraph executablec ould not be found at {mg5_exe}")
    return mg5_exe
