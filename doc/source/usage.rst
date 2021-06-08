.. _usage-label:

Usage
=====

Madflow automatic script
------------------------

With the installation of madflow, a script is provided to automatically generate and integrate
leading order cross sections.
After installation you can launch the script with the ``madflow`` command.

Leading Order integration
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``madflow`` script can integrate any leading order process using
the same syntax as MG5_aMC\@NLO:

.. code-block:: bash

  madflow --madgraph_process "p p > t t~ g" --pt_cut 60 --pdf NNPDF31_nnlo_as_0118 -m 2


The script will use MG5_aMC\@NLO and the ``madflow``-provided :ref:`plugin <plugin-label>`
to generate the relevant matrix element and a vectorized form of ``RAMBO`` to
integrate it.

Leading Order template
----------------------

The goal of the ``madflow`` script is not to be completely general but to serve as a quick way
of getting results and debugging.
Fore more complex and customized cross section calculations is recommended to build your own
integration script.

In order to simplify the process, a leading order template is provided by the ``madflow``
script.
To generate the relevant matrix element files and leading order script without performing the integration
you can use the ``--dry_run`` option.

.. code-block:: bash

  madflow --madgraph_process "p p > Z" --output pp2z --dry_run -m 1


The previous command will output to the ``pp2z`` all the necessary files to perform the integration
alongside a template for cross section calculations: ``leading_order.py``.
This template serves as a guide to start building your own fixed order calculation.
