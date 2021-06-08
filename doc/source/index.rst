.. title::
    madflow's documentation!

=================================================================
MadFlow: parton-level monte carlo simulations directly to you GPU
=================================================================

.. 
   image:: https://zenodo.org/badge/DOI/10.5281/zenodo.XX.svg
   :target: https://doi.org/10.5281/zenodo.XX

.. image:: https://img.shields.io/badge/arXiv-physics.comp--ph%2F%20%20%20%202105.10529-%23B31B1B
   :target: https://arxiv.org/abs/2105.10529

Overview
--------

MadFlow is a framework for Monte Carlo simulation of particle physics processes designed to take full advantage of hardware accelerators.
Processes can be generated using `MadGraph5_aMC@NLO <https://launchpad.net/mg5amcnlo>`_ and are then output in vectorized (or tensorized)
form by the ``madflow``-provided plugin.

The vectorized output is compiled using the `TensorFlow <https://www.tensorflow.org/>`_ library (hence, tensorized)
and then integrated using the `VegasFlow <https://vegasflow.readthedocs.io/>`_ library.
The PDF interpolation is provided by `PDFFlow <https://pdfflow.readthedocs.io>`_.
All tools are capable of running hardware with different hardware acceleration capabilities, such as multi-threading CPU, single-GPU and multi-GPU setups.


Open Source
-----------
The ``madflow`` package is open source and available at https://github.com/N3PDF/madflow

Motivation and design
=====================

Madflow is developed by the Particle Physics group of the University of Milan.
Theoretical calculations in particle physics are incredibly time consuming operations,
sometimes taking months in big clusters all around the world.

These expensive calculations are driven by the high dimensional phase spaces and
the complexity of the integrands which can be composed of dozens or hundreds of diagrams.
Furthermore, most of these calculations are built upon very dated code and methodologies
not suitable for newest hardware.
These problems create a huge technical debt which is very difficult to overcome
by newcomers to the field.

With Madflow with aim to close this gap between theoretical and high performance computing
by providing a framework that it is maintainable, extensible and modern
while 

..
  How to cite ``madflow``?
  =========================

  When using ``madflow`` in your research, please cite the following publications:


  Bibtex:

  .. code-block:: latex

      @article{Carrazza:2020rdn,
          author = "Carrazza, Stefano and Cruz-Martinez, Juan M.",
          title = "{VegasFlow: accelerating Monte Carlo simulation across multiple hardware platforms}",
          eprint = "2002.12921",
          archivePrefix = "arXiv",
          primaryClass = "physics.comp-ph",
          reportNumber = "TIF-UNIMI-2020-8",
          doi = "10.1016/j.cpc.2020.107376",
          journal = "Comput. Phys. Commun.",
          volume = "254",
          pages = "107376",
          year = "2020"
      }


      @software{vegasflow_package,
          author       = {Juan Cruz-Martinez and
                          Stefano Carrazza},
          title        = {N3PDF/vegasflow: vegasflow v1.0},
          month        = feb,
          year         = 2020,
          publisher    = {Zenodo},
          version      = {v1.0},
          doi          = {10.5281/zenodo.3691926},
          url          = {https://doi.org/10.5281/zenodo.3691926}
      }

FAQ
===

Why the name ``MadFlow``?
---------------------------

It is a combination of the names `Madgraph` and `Tensorflow`.

- **Madgraph**: add here information about Madgraph and the proper references

- **TensorFlow**: the `tensorflow <https://www.tensorflow.org/>`_ is developed by Google and was made public in November of 2015. It is a perfect combination between performance and usability. With a focus on Deep Learning, TensorFlow provides an algebra library able to easily run operations in many different devices: CPUs, GPUs, TPUs with little input by the developer.



Indices and tables
==================

.. toctree::
    :maxdepth: 1
    :caption: Overview:
    :hidden: 

    Madflow<self>
    Installation<installation>
    usage

.. toctree::
    :maxdepth: 1
    :caption: Components:
    :hidden:

    phasespace
    lhewriter
    Package specs<apisrc/madflow>


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
