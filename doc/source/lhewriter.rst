.. _lhewriter-label:

Event output
============

A common workflow for particle physics Monte Carlo simulator is the generation
f events files (in the LHE format) to be analyzed later on.
Although the user may use the leading order :ref:`template <lotemplate-label>` to
write events in their own favourite way, ``madflow`` also provides an interface to
MG5_aMC\@NLO own event writer through the :py:class:`LheWriter <madflow.lhe_writer.LheWriter>` class.

In order to avoid hanging the calculation because of disc i/o, the events are dumped
asynchronously from the rest of the calculation.
As a consequence, ``LheWriter`` instances should be always used within a context manager
in order to avoid unpleasant situations in which the program exits before the full
length of events has been dumped to disk.

.. code-block:: python

  from madflow.lhe_writer import LheWriter
  from madflow.config import DTYPE
  from tensorflow import py_function

  with LheWriter("process_path", "process_name") as lhe_writer:
    
    def cross_section(xrand, weight=None, **kwargs):
      ...
      ps = phasespace(...)
      result = matrix.smatrix(ps, ...)
      py_function(func=lhe_writer.lhe_parser, inp=[ps, result*weight], Tout=DTYPe)
      return result


Note the usage of the TensorFlow function ``py_function``.
The usage of this function allows us to make calls to functions which haven't got
necessarily a GPU kernel.
Take in consideration, however, that the usage of such functions will trigger a copy
event from the hardware accelerator device to CPU.
