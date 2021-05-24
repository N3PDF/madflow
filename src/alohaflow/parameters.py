"""
    Utilities and functions to deal with the parameters of the model
"""
from .config import DTYPE, DTYPECOMPLEX, complex_me, float_me, run_eager
import numpy as np
import tensorflow as tf

GS_SIGNATURE = [tf.TensorSpec(shape=[None], dtype=DTYPECOMPLEX)]
ALPHAS_SIGNATURE = [tf.TensorSpec(shape=[None], dtype=DTYPE)]


@tf.function(input_signature=ALPHAS_SIGNATURE)
def _alphas_to_gs(alpha_s):
    return complex_me(2.0 * tf.math.sqrt(np.pi * alpha_s))


class Model:
    """This class is instantiated with knowledge about
    all couplings and parameters in the process of interest
    and provides an interface to compute them in a per-phase space
    basis

    Parameters
    ---------
        constants: tuple(DTYPE)
            tuple with all constants of the model
        functions: tuple(functions)
            tuple with all parameters of the model which depend on g_s
    """

    def __init__(self, constants, functions):
        self._tuple_constants = constants
        self._tuple_functions = functions
        self._constants = list(constants)
        self._to_evaluate = [tf.function(i, input_signature=GS_SIGNATURE) for i in functions]
        self._frozen = []

    @property
    def frozen(self):
        """Whether the model is frozen for a given value of alpha_s or not"""
        return bool(self._frozen)

    def freeze_alpha_s(self, alpha_s):
        """The model can be frozen to a specific value
        of alpha_s such that all phase space points are evaluated at that value
        Parameters
        ----------
            alpha_s: float
        """
        if self.frozen:
            raise ValueError("The model is already frozen")
        self._frozen = self._evaluate(float_me([alpha_s]))

    def unfreeze(self):
        """Remove the frozen status"""
        self._frozen = []

    @tf.function(input_signature=ALPHAS_SIGNATURE)
    def _evaluate(self, alpha_s):
        """Evaluate all couplings for the given values of alpha_s
        Parameters
        ----------
            alpha_s: tensor of shape (None,)
        """
        gs = _alphas_to_gs(alpha_s)
        results = [fun(gs) for fun in self._to_evaluate]
        if not results:
            return self._constants
        if not self._constants:
            return results
        return *self._constants, *results

    def get_masses(self):
        """Get the masses that entered the model as constants"""
        masses = []
        for key, val in self._tuple_constants._asdict().items():
            if key.startswith("mdl_"):
                masses.append(val)
        return masses

    def evaluate(self, alpha_s=None):
        """Evaluate alpha_s, if the model is frozen
        returns the frozen values"""
        if self.frozen:
            return self._frozen
        return self._evaluate(alpha_s)
