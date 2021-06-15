"""
    Default settings for madflow

    This program is to be installed alongside VegasFlow and
    PDFFlow so we'll take the main configuration from there.
    The environment variables for log levels and float/int types
    are propagated to any other programs used by madflow
"""
import os
import logging
from distutils.spawn import find_executable
import subprocess as sp
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
# Log levels
LOG_DICT = {"0": logging.ERROR, "1": logging.WARNING, "2": logging.INFO, "3": logging.DEBUG}
_log_level = LOG_DICT[_log_level_idx]
logger = logging.getLogger(__name__.split(".")[0])
logger.setLevel(_log_level)

# Create and format the log handler
_console_handler = logging.StreamHandler()
_console_handler.setLevel(_log_level)
_console_format = logging.Formatter("[%(levelname)s] (<madflow>) %(message)s")
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
    """Return the path to the madgrapt root"""
    if madpath is None:
        madpath = os.environ.get("MADGRAPH_PATH", "mg5amcnlo")
    madgraph_path = Path(madpath)
    if not madgraph_path.exists():
        raise ValueError(
            f"""{madgraph_path} does not exist.
Are you sure Madgraph is installed? https://madflow.readthedocs.io/en/latest/installation.html#mg5-amc-nlo-plugin
MadFlow needs a valid path for Madgraph, can be given as env. variable MADGRAPH_PATH"""
        )
    # If the path exists, check whether the madgraph executable is there
    _ = get_madgraph_exe(madgraph_path)
    return madgraph_path


def get_madgraph_exe(madpath=None):
    """Return the path to the madgraph executable"""
    if madpath is None:
        madpath = get_madgraph_path(madpath)
    mg5_exe = madpath / "bin/mg5_aMC"
    if not mg5_exe.exists():
        raise ValueError(
            f"""Madgraph executable could not be found at {mg5_exe},
are you sure Madgraph is installed? https://madflow.readthedocs.io/en/latest/installation.html#mg5-amc-nlo-plugin"""
        )
    return mg5_exe


def _parse_amd_info(info):
    """Parse the information returned by
    rocm-smi to find out the amount of free memory in MB
    """
    for line in info.split("\n"):
        if line.startswith("GPU") and "Used" not in line:
            total_b = line.strip().rsplit(" ", 1)[-1]
    return int(total_b) / 1024 / 1024


def guess_events_limit(nparticles):
    """Given a number of particles, reads GPU memory to guess
    what should be the event limit.
    Use the smallest available GPU as the limit (but print warning in that case)
    """
    gpu_physical_devices = tf.config.list_physical_devices("GPU")
    memories = []
    for gpu in gpu_physical_devices:
        gpu_idx = gpu.name.rsplit(":", 1)[-1]
        # Nvidia and AMD GPU split
        if find_executable("nvidia-smi"):
            gpuinfo_command = (
                f"nvidia-smi --id={gpu_idx} --query-gpu=memory.total --format=csv,noheader,nounits"
            )
            parse = lambda x: int(x)
        elif find_executable("rocm-smi"):
            gpuinfo_command = f"rocm-smi -d {gpu_idx} --showmeminfo VRAM"
            parse = _parse_amd_info
        else:
            logger.error("No rocm-smi or nvidia-smi command found, GPU memory cannot be guessed")
            continue

        try:
            out = parse(
                sp.run(
                    gpuinfo_command, check=True, shell=True, capture_output=True, text=True
                ).stdout
            )
            memories.append(out)
        except sp.CalledProcessError:
            logger.error("Could not read the memory of GPU %d", gpu_idx)
        except ValueError:
            logger.error("Could not read the memory of GPU %d", gpu_idx)

    if not memories:
        return None

    if len(set(memories)) == 1:
        memory = memories[0]
    else:
        memory = min(memories)
        logger.warning(
            "Using the memory of GPU#%d: %d MiB to limit the events per device",
            memories.index(memory),
            memory,
        )

    # NOTE: this is based on heuristics in some of the available cards
    if memory < 13000:
        events_limit = int(1e5)
    else:
        events_limit = int(5e5)

    if nparticles > 5:
        events_limit //= 10
    return events_limit
