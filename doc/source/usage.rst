.. _usage-label:

Usage
=====

For a more precise description of what `madflow` can do please visit the online documentation.

For convenience a script is provided which should have been installed alongside the library.
Using this script is possible to run any process at Leading Order, integrated with a `RAMBO`-like phasespace.

.. code-block:: bash

   madflow --help


.. code-block:: bash

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
