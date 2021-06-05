# Madflow

References: https://arxiv.org/abs/2105.10529

## Install `madflow`

#### From PyPI

To be done

#### From the repository

```bash
  git clone https://github.com/N3PDF/madflow.git
  cd madflow
  pip install .
```

### External tools

`madflow` relies in a number of external tools.
Some of them are just used for convenience and are optional, some are necessary for the proper functioning of the program.

#### MG5_aMC

A valid installation of MG5_aMC (2.8+) is necessary in order to generate matrix elements.
If you already have a valid installation, please add the following environment variable pointing to the right directory: `MADGRAPH_PATH`.
Below are the instructions for MG5_aMC 3.1.0, for a more recent release please visit the MG5_aMC@NLO [site](https://launchpad.net/mg5amcnlo).

```bash
wget https://launchpad.net/mg5amcnlo/3.0/3.1.x/+download/MG5_aMC_v3.1.0.tar.gz
tar xfz MG5_aMC_v3.1.0.tar.gz
export MADGRAPH_PATH=${PWD}/MG5_aMC_v3_1_0
```

#### LHAPDF

While `LHAPDF` is not strictly necessary to use the `madflow` library or run any of the scripts,
having access to the `lhapdf` python wrapper can be convenient in order to manage the different PDFsets.
Please install the latest version from the LHAPDF [site](https://lhapdf.hepforge.org/).

Otherwise, if your installed version of `pdfflow` is equal or greater than `1.2.1`,
you can manually install the [PDF sets](https://lhapdf.hepforge.org/pdfsets.html) in a suitable directory
and ensure that either the `PDFFLOW_DATA_PATH` or `LHAPDF_DATA_PATH` environment variables are pointing to it.

You can check your installed version of `pdfflow` with: `python -c 'import pdfflow ; print(pdfflow.__version__);'`

## Install plugin in MG5_aMC

In order to install the `madflow` plugin in MG5_aMC@NLO, it is necessary to link the `madgraph_plugin` folder inside the `PLUGIN` directory of MG5_aMC@NLO.
For instance, if the environment variable `$MADGRAPH_PATH` is pointing to the MG5_aMC root:

```bash
    ln -s ${PWD}/madgraph_plugin ${MADGRAPH_PATH}/PLUGIN/pyout
```

## Use `madflow`

For a more precise description of what `madflow` can do please visit the online documentation.

For convenience a script is provided which should have been installed alongside the library.
Using this script is possible to run any process at Leading Order, integrated with a `RAMBO`-like phasespace.

```bash
  madflow --help
```
```bash
    [-h] [-v] [-p PDF] [--no_pdf] [-c] [--madgraph_process MADGRAPH_PROCESS] [-m MASSIVE_PARTICLES] [-g] [--pt_cut PT_CUT] [--histograms]

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Print extra info
      -p PDF, --pdf PDF     PDF set
      --no_pdf              Don't use a PDF for the initial state
      -c, --enable_cuts     Enable the cuts
      --madgraph_process MADGRAPH_PROCESS
                            Set the madgraph process to be run
      -m MASSIVE_PARTICLES, --massive_particles MASSIVE_PARTICLES
                            Number of massive particles
      -g, --variable_g      Use variable g_s
      --pt_cut PT_CUT       Minimum pt for the outgoint particles
      --histograms          Generate LHE files/histograms
```
