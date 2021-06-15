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


Technical limitations
^^^^^^^^^^^^^^^^^^^^^

While, bugs permitting, ``madflow`` should be completely functional, its development
is still ongoing.
We tried to use conservative results for the default values to manage the memory
of hardware accelerators, however our own testing devices are limited.

The ``madflow`` script exposes the ``--events_limit`` flag which limits the maximum
amount of events that can be sent to the accelerator
Increasing the number of events run in the accelerator at once will decrease the latency
and accordingly will increase the performance of the calculation.
On the flip side, increasing the number of events will increase the amount of on-device
memory required to perform the calculation
(which the host computer should also have to prepare said calculation)
and unpleasant situations where the calculation goes Out of Memory (OOM) can occur.
If that's the case, please decrease the ``--events_limit`` flag.



.. _lotemplate-label:

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


Scope
^^^^^

With the leading order template we aim to kick-start more complex fixed order calculations.
The template is commented in a pedagogical step-by-step way and should suffice to run the simpler calculations
while making clear where changes are needed for more complex situations.

The main areas to address are:

- Phase-space and cuts:

In the template the phase space is a vectorized form of ``rambo`` which we have dubbed
``ramboflow``.
As the seasoned phenomenologist knows, the right choice of phase-space generator can mean
the difference between a convergent integration and crazy and unreasonable results.
Unfortunately building a fully general phase-space sampling algorithm is (as the very same
seasoned phenomenologist surely knows) a very much non-trivial subject.
For the time being ``ramboflow`` is the only phase-space generator provider by ``madflow``
and thus more complicated calculations will need to build their own.

The cuts on the phase-space, while trivial on their own, must be applied carefully
when building software targeting hardware accelerators.
The number-1 enemy of GPUs is branching, and cuts to the phase-space will mean
exactly that.
Therefore cuts (and equivalently any kind of multi-channeling algorithm)
should be applied in such a way that the number of events that are computed at each
single go is maximized.


- Matrix element evaluation:

While one of the advantages of ``madflow`` is to use the capabilities of MG5_aMC in order
to automatically generate matrix elements which can be evaluated in hardware accelerators
no hardware-specific optimization has been applied to the evaluation strategy
which remain that of `ALOHA <https://inspirehep.net/literature/922833>`_,
which is based in a raw evaluation of Feynman diagrams.

While this approach ensures universality, it also means the number of diagrams
can grow in a factorial manner, soon becoming intractable.
Processes with many particles in the final state will surely benefit of other
strategies.
In the future we aim to provide interfaces to other Matrix-elements provided
(also vectorized when possible) in order to address these short-comings.
