.. _installation-label:

Installation
============

Main Package
------------

The package can be installed with pip:

.. code-block:: bash

  python3 -m pip install madflow


If you prefer a manual installation can do:

.. code-block:: bash

  git clone https://github.com/N3PDF/madflow.git
  cd madflow
  pip install .

if you are planning to extend or develop code use instead:

.. code-block:: bash

  pip install -e .


.. _plugin-label:

MG5_aMC\@NLO Plugin
--------------------

A valid installation of MG5_aMC\@NLO (2.8+) is necessary in order to generate matrix elements.

If you already have a valid installation, please add the following environment variable pointing to the right directory: ``MADGRAPH_PATH``.
Below are the instructions for MG5_aMC\@NLO 3.1.0, for a more recent release please visit the MG5_aMC\@NLO `site <https://launchpad.net/mg5amcnlo>`_.

.. code-block:: bash

  wget https://launchpad.net/mg5amcnlo/3.0/3.1.x/+download/MG5_aMC_v3.1.0.tar.gz
  tar xfz MG5_aMC_v3.1.0.tar.gz
  export MADGRAPH_PATH=${PWD}/MG5_aMC_v3_1_0


Once MG5_aMC\@NLO is installed, all that's left is to link the ``madflow`` plugin inside
the MG5_aMC\@NLO folder.


.. code-block:: bash

  madflow --autolink


If you prefer to link the plugin manually, it is necessary to link the
``madgraph_plugin`` folder inside the ``PLUGIN`` directory of MG5_aMC\@NLO.
For instance, if the environment variable ``$MADGRAPH_PATH`` is pointing to the MG5_aMC root:

.. code-block:: bash

    ln -s ${PWD}/madgraph_plugin ${MADGRAPH_PATH}/PLUGIN/pyout

.. 
  Package distributions
  ---------------------

  It is also possible to install the package from repositories such as `conda-forge <https://anaconda.org/conda-forge/madflow>`_ or the `Arch User Repository <https://aur.archlinux.org/packages/madflow/>`_. Note that in this cases MG5_aMC\@NLO is installed automatically.

  .. code-block:: bash

    conda install madflow -c conda-forge
    yay -S madflow

