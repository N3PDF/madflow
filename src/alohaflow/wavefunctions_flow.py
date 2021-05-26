"""
    Wave functions
"""

from .config import DTYPECOMPLEX, complex_tf, complex_me, DTYPE, DTYPEINT, int_me, float_me

import tensorflow as tf
import tensorflow.math as tfmath

smom = tf.TensorSpec(shape=[None, 4], dtype=DTYPE)    # momenta signature
svec = tf.TensorSpec(shape=[None], dtype=DTYPE)       # vector signature
sscalar = tf.TensorSpec(shape=[], dtype=DTYPE)        # scalar signature
sscalar_int = tf.TensorSpec(shape=[], dtype=DTYPEINT) # scalar signature int

wave_signature = [smom] + [sscalar]*3

scalar_signature = [smom] + [sscalar]

sign_signature = [sscalar]*2

signvec_signature = [sscalar] + [svec]


@tf.function(input_signature=sign_signature)
def sign(x, y):
    """Fortran's sign transfer function"""
    # dropping the checks for the moment
    return x * tf.math.sign(y)


@tf.function(input_signature=signvec_signature)
def signvec(x, y):
    """Fortran's sign transfer function"""
    # dropping the checks for the moment
    return x * tf.math.sign(y)


@tf.function(input_signature=scalar_signature)
def sxxxxx(p, nss):
    """
    Defines a scalar wavefunction. Input momenta have shape (num events, 4).
    
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nss: tf.Tensor, of shape=()
    
    Returns
    -------
        tf.Tensor, of shape=(3,None)    
    """
    # Note: here p[:,i] selects the momentum dimension and is a [nevt,] tensor
    v0 = tf.expand_dims(complex_tf(p[:, 0] * nss, p[:, 3] * nss), 0)  # [nevt,] complex
    v1 = tf.expand_dims(complex_tf(p[:, 1] * nss, p[:, 2] * nss), 0)  # [nevt,] complex
    v = tf.expand_dims(complex_tf(1.0, 0.0), 0)
    return tf.concat([v0, v1, v], axis=0)


@tf.function(input_signature=wave_signature)
def ixxxxx(p, fmass, nhel, nsf):
    """
    Defines an inflow fermion wavefunction. Input momenta have shape
    (num events, 4).

    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        fmass: tf.Tensor, of shape=()
        nhel: tf.Tensor, of shape=()
        nsf: tf.Tensor, of shape=()
    
    Returns
    -------
        tf.Tensor, of shape=(6,None)    
    """
    # Note: here p[:,i] selects the momentum dimension and is a [nevt,] tensor
    v0 = tf.expand_dims(complex_tf(-p[:, 0] * nsf, -p[:, 3] * nsf), 0)  # [nevt,] complex
    v1 = tf.expand_dims(complex_tf(-p[:, 1] * nsf, -p[:, 2] * nsf), 0)  # [nevt,] complex
    nh = nhel * nsf  # either +1 or -1
    ip = (1 + nh) // 2
    im = (1 - nh) // 2

    massive = fmass != 0
    v = tf.cond(massive,
                lambda: _ix_massive(p, fmass, nsf, nh, ip, im),
                lambda: _ix_massless(p, nhel, nsf, nh)
               )
    return tf.concat([v0, v1, v], axis=0)


@tf.function(input_signature=wave_signature)
def oxxxxx(p, fmass, nhel, nsf):
    """ 
    Defines an outgoing fermion wavefunction. Input momenta have shape
    (num events, 4).

    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        fmass: tf.Tensor, of shape=()
        nhel: tf.Tensor, of shape=()
        nsf: tf.Tensor, of shape=()
    
    Returns
    -------
        tf.Tensor, of shape=(6,None)
    """
    v0 = tf.expand_dims(complex_tf(p[:, 0] * nsf, p[:, 3] * nsf), 0)  # [nevt,] complex
    v1 = tf.expand_dims(complex_tf(p[:, 1] * nsf, p[:, 2] * nsf), 0)  # [nevt,] complex
    nh = nhel * nsf  # either +1 or -1
    sqp0p3 = tfmath.sqrt(tfmath.maximum(p[:, 0] + p[:, 3], 0.0)) * nsf  # [nevt,]

    massive = fmass != 0
    v = tf.cond(massive,
                lambda: _ox_massive(p, fmass, nhel, nsf, nh),
                lambda: _ox_massless(p, nhel, nh, sqp0p3)
               )
    return tf.concat([v0, v1, v], axis=0)


@tf.function(input_signature=wave_signature)
def vxxxxx(p, vmass, nhel, nsv):
    """
    Defines a vector wavefunction. nhel=4 is for checking BRST. Inpu
    momenta have shape (num events, 4).

    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        fmass: tf.Tensor, of shape=()
        nhel: tf.Tensor, of shape=()
        nsv: tf.Tensor, of shape=()
    
    Returns
    -------
        tf.Tensor, of shape=(6,None)
    """
    nevts = tf.shape(p, out_type=DTYPEINT)[0]
    hel0 = 1.0 - tfmath.abs(nhel)

    sqh = float_me(tfmath.sqrt(0.5))
    nsvahl = nsv * tfmath.abs(nhel)
    pt2 = p[:, 1] ** 2 + p[:, 2] ** 2
    pp = tfmath.minimum(p[:, 0], tfmath.sqrt(pt2 + p[:, 3] ** 2))
    pt = tfmath.minimum(pp, tfmath.sqrt(pt2))

    v0 = tf.expand_dims(complex_tf(p[:, 0] * nsv, p[:, 3] * nsv), 0)  # [1,nevts] complex
    v1 = tf.expand_dims(complex_tf(p[:, 1] * nsv, p[:, 2] * nsv), 0)

    BRST = nhel == 4
    v = tf.cond(BRST,
                lambda: _vx_BRST_check(p, vmass, nevts),
                lambda: _vx_no_BRST_check(
                           p, vmass, nhel, nsv, hel0, sqh, nsvahl, pp, pt, nevts
                                     )
               )
    return tf.concat([v0, v1, v], axis=0)


#===============================================================================
# ixxxxx related functions
_ix_massive_signature = [smom] + [sscalar]*5
@tf.function(input_signature=_ix_massive_signature)
def _ix_massive(p, fmass, nsf, nh, ip, im):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        fmass: tf.Tensor, of shape=()
        nsf: tf.Tensor, of shape=()
        nh: tf.Tensor, of shape=()
        ip: tf.Tensor, of shape=()
        im: tf.Tensor, of shape=()

    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    pp = tfmath.minimum(
        p[:, 0], tfmath.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2 + p[:, 3] ** 2)
    )  # [nevt,]
    cond = tf.expand_dims(pp == 0, 0)
    return tf.where(cond,
                    _ix_massive_pp_zero(fmass, nsf, ip, im),
                    _ix_massive_pp_nonzero(p, fmass, nsf, nh, ip, im, pp))


_ix_massive_pp_zero_signature = [sscalar]*4
@tf.function(input_signature=_ix_massive_pp_zero_signature)
def _ix_massive_pp_zero(fmass, nsf, ip, im):
    """
    Parameters
    ----------
        fmass: tf.Tensor, of shape=()
        nsf: tf.Tensor, of shape=()
        ip: tf.Tensor, of shape=()
        im: tf.Tensor, of shape=()

    Returns
    -------
        tf.Tensor, of shape=(4,1) and dtype DTYPECOMPLEX
    """
    sqm = tfmath.sqrt(tfmath.abs(fmass))
    sqm = tf.stack(
        [sqm, sign(sqm, fmass)]
    )  # [fmass, fmass] ---> TODO: why calling sign on the result of a tfmath.sqrt ????
    v = [complex_tf(0,0)] * 4
    v[0] = complex_tf(ip * sqm[int_me(ip)], 0.0)
    v[1] = complex_tf(im * nsf * sqm[int_me(ip)], 0.0)
    v[2] = complex_tf(ip * nsf * sqm[int_me(im)], 0.0)
    v[3] = complex_tf(im * sqm[int_me(im)], 0.0)
    return tf.reshape(tf.stack(v), [4,1])


_ix_massive_pp_nonzero_signature = [smom] + [sscalar]*5 + [svec]
@tf.function(input_signature=_ix_massive_pp_nonzero_signature)
def _ix_massive_pp_nonzero(p, fmass, nsf, nh, ip, im, pp):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        fmass: tf.Tensor, of shape=()
        nsf: tf.Tensor, of shape=()
        nh: tf.Tensor, of shape=()
        ip: tf.Tensor, of shape=()
        im: tf.Tensor, of shape=()
        pp: tf.Tensor, of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    sf = tf.stack(
        [(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5], axis=0
    )  # [2,]
    omega = tf.stack(
        [tfmath.sqrt(p[:, 0] + pp), fmass / (tfmath.sqrt(p[:, 0] + pp))], axis=0
    )  # [2, nevt]
    sfomeg = tf.stack(
        [sf[0] * omega[int_me(ip)], sf[1] * omega[int_me(im)]], axis=0
    )  # [2,nevt]
    pp3 = tfmath.maximum(pp + p[:, 3], 0.0)  # [nevt,]
    chi1 = tf.where(
        pp3 == 0,
        complex_tf(-nh, 0),
        complex_tf(
            nh * p[:, 1] / tfmath.sqrt(2.0 * pp * pp3), p[:, 2] / tfmath.sqrt(2.0 * pp * pp3)
        ),
    )  # [nevt,] complex
    chi2 = complex_tf(tfmath.sqrt(pp3 * 0.5 / pp), 0.0)  # [nevt,] complex
    chi = tf.stack([chi2, chi1], axis=0)  # [2, nevt] complex
    v = [complex_tf(0,0)] * 4
    v[0] = complex_tf(sfomeg[0], 0.0) * chi[int_me(im)]  # [nevt,] complex
    v[1] = complex_tf(sfomeg[0], 0.0) * chi[int_me(ip)]
    v[2] = complex_tf(sfomeg[1], 0.0) * chi[int_me(im)]
    v[3] = complex_tf(sfomeg[1], 0.0) * chi[int_me(ip)]
    return tf.stack(v, axis=0)


_ix_massless_signature = [smom] + [sscalar]*3
@tf.function(input_signature=_ix_massless_signature)
def _ix_massless(p, nhel, nsf, nh):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nhel: tf.Tensorm of shape=()
        nsf: tf.Tensorm of shape=()
        nh: tf.Tensorm of shape=()

    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    sqp0p3 = tfmath.sqrt(tfmath.maximum(p[:, 0] + p[:, 3], 0.0)) * nsf  # [nevt,]
    chi1 = tf.where(sqp0p3 == 0,
                    _ix_massless_sqp0p3_zero(p, nhel),
                    _ix_massless_sqp0p3_nonzero(p, nh, sqp0p3)
                   )
    chi = tf.stack([complex_tf(sqp0p3, 0.0), chi1], axis=0)  # [2, nevt]
    return tf.cond(nh == 1,
                   lambda: _ix_massless_nh_one(chi),
                   lambda: _ix_massless_nh_not_one(chi)
                  )


_ix_massless_sqp0p3_zero_signature = [smom] + [sscalar]
@tf.function(input_signature=_ix_massless_sqp0p3_zero_signature)
def _ix_massless_sqp0p3_zero(p, nhel):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nhel: tf.Tensorm of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None) and dtype DTYPECOMPLEX
    """
    return complex_tf(-nhel * tfmath.sqrt(2.0 * p[:, 0]), 0.0)


_ix_massless_sqp0p3_nonzero_signature = [smom] + [sscalar] + [svec]
@tf.function(input_signature=_ix_massless_sqp0p3_nonzero_signature)
def _ix_massless_sqp0p3_nonzero(p, nh, sqp0p3):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nh: tf.Tensorm of shape=()
        sqp0p3: tf.Tensorm of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(None) and dtype DTYPECOMPLEX
    """
    return complex_tf(nh * p[:, 1] / sqp0p3, p[:, 2] / sqp0p3)  # [nevt,] complex


@tf.function(input_signature=[tf.TensorSpec(shape=[2,None], dtype=DTYPECOMPLEX)])
def _ix_massless_nh_one(chi):
    """
    Parameters
    ----------
        chi: tf.Tensor, of shape=(2,None) and dtype DTYPECOMPLEX

    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    v = [complex_tf(0,0)] * 4
    v[2] = chi[0]
    v[3] = chi[1]
    v[0] = tf.ones_like(v[2]) * complex_tf(0.0, 0.0)
    v[1] = tf.ones_like(v[2]) * complex_tf(0.0, 0.0)
    return tf.stack(v, axis=0)


@tf.function(input_signature=[tf.TensorSpec(shape=[2,None], dtype=DTYPECOMPLEX)])
def _ix_massless_nh_not_one(chi):
    """
    Parameters
    ----------
        chi: tf.Tensor, of shape=(2,None) and dtype DTYPECOMPLEX

    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    v = [complex_tf(0,0)] * 4
    v[0] = chi[1]
    v[1] = chi[0]
    v[2] = tf.ones_like(v[0]) * complex_tf(0.0, 0.0)
    v[3] = tf.ones_like(v[0]) * complex_tf(0.0, 0.0)
    return tf.stack(v, axis=0)

#===============================================================================
# oxxxxx related functions_
_ox_massive_signature = [smom] + [sscalar]*4
@tf.function(input_signature=_ox_massive_signature)
def _ox_massive(p, fmass, nhel, nsf, nh):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        fmass: tf.Tensor, of shape=()
        nhel: tf.Tensor, of shape=()
        nsf: tf.Tensor, of shape=()
        nh: tf.Tensor, of shape=()

    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    pp = tfmath.minimum(
        p[:, 0], tfmath.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2 + p[:, 3] ** 2)
    )  # [nevt,]
    cond = tf.expand_dims(pp == 0, 0)
    return tf.where(cond,
                    _ox_massive_pp_zero(fmass, nhel, nsf, nh),
                    _ox_massive_pp_nonzero(p, fmass, nsf, nh, pp)
                   )  # [4, nevt] complex


_ox_massive_pp_zero_signature = [sscalar]*4
@tf.function(input_signature=_ox_massive_pp_zero_signature)
def _ox_massive_pp_zero(fmass, nhel, nsf, nh):
    """
    Parameters
    ----------
        fmass: tf.Tensor, of shape=()
        nhel: tf.Tensor, of shape=()
        nsf: tf.Tensor, of shape=()
        nh: tf.Tensor, of shape=()

    Returns
    -------
        tf.Tensor, of shape=(4,1) and dtype DTYPECOMPLEX
    """
    sqm = tfmath.sqrt(tfmath.abs(fmass))
    sqm = tf.stack(
        [sqm, sign(sqm, fmass)]
    )  # [fmass, fmass] ---> why calling sign on the result of a tfmath.sqrt ????
    ip = -((1 - nh) // 2) * nhel
    im = (1 + nh) // 2 * nhel
    v = [complex_tf(0,0)] * 4
    v[0] = complex_tf(im * sqm[int_me(tfmath.abs(im))], 0.0)  # just a complex number
    v[1] = complex_tf(ip * nsf * sqm[int_me(tfmath.abs(im))], 0.0)
    v[2] = complex_tf(im * nsf * sqm[int_me(tfmath.abs(ip))], 0.0)
    v[3] = complex_tf(ip * sqm[int_me(tfmath.abs(ip))], 0.0)
    return tf.reshape(tf.stack(v), [4,1])  # [4,] complex


_ox_massive_pp_nonzero_signature = [smom] + [sscalar]*3 + [svec]
@tf.function(input_signature=_ox_massive_pp_nonzero_signature)
def _ox_massive_pp_nonzero(p, fmass, nsf, nh, pp):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        fmass: tf.Tensor, of shape=()
        nsf: tf.Tensor, of shape=()
        nh: tf.Tensor, of shape=()
        pp: tf.Tensor, of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    sf = tf.stack(
        [(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5], axis=0
    )  # [2,]
    omega = tf.stack(
        [tfmath.sqrt(p[:, 0] + pp), fmass / (tfmath.sqrt(p[:, 0] + pp))], axis=0
    )  # [2, nevt]
    ip = (1 + nh) // 2
    im = (1 - nh) // 2
    sfomeg = tf.stack(
        [sf[0] * omega[int_me(ip)], sf[1] * omega[int_me(im)]], axis=0
    )  # [2,nevt]
    pp3 = tfmath.maximum(pp + p[:, 3], 0.0)  # [nevt,]
    chi1 = tf.where(
        pp3 == 0,
        complex_tf(-nh, 0),
        complex_tf(
            nh * p[:, 1] / tfmath.sqrt(2.0 * pp * pp3), -p[:, 2] / tfmath.sqrt(2.0 * pp * pp3)
        ),
    )  # [nevt,] complex
    chi2 = complex_tf(tfmath.sqrt(pp3 * 0.5 / pp), 0.0)  # [nevt,] complex
    chi = tf.stack([chi2, chi1], axis=0)  # [2, nevt] complex
    v = [complex_tf(0,0)] * 4
    v[0] = complex_tf(sfomeg[1], 0.0) * chi[int_me(im)]  # [nevt,] complex
    v[1] = complex_tf(sfomeg[1], 0.0) * chi[int_me(ip)]
    v[2] = complex_tf(sfomeg[0], 0.0) * chi[int_me(im)]
    v[3] = complex_tf(sfomeg[0], 0.0) * chi[int_me(ip)]
    return tf.stack(v, axis=0)  # [4, nevt] complex


_ox_massless_signature = [smom] + [sscalar]*2 + [svec]
@tf.function(input_signature=_ox_massless_signature)
def _ox_massless(p, nhel, nh, sqp0p3):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nhel: tf.Tensorm of shape=()
        nh: tf.Tensorm of shape=()
        sqp0p3: tf.Tensor, of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    chi1 = tf.where(sqp0p3 == 0,
                    _ox_massless_sqp0p3_zero(p, nhel),
                    _ox_massless_sqp0p3_nonzero(p, nh, sqp0p3)
                   )
    chi = tf.stack([complex_tf(sqp0p3, 0.0), chi1], axis=0)  # [2, nevt]
    return tf.cond(nh == 1,
                   lambda: _ox_massless_nh_one(chi),
                   lambda: _ox_massless_nh_not_one(chi)
                  )


_ox_massless_sqp0p3_zero_signature = [smom] + [sscalar]
@tf.function(input_signature=_ox_massless_sqp0p3_zero_signature)
def _ox_massless_sqp0p3_zero(p, nhel):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nhel: tf.Tensorm of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None) and dtype DTYPECOMPLEX
    """
    return complex_tf(-nhel * tfmath.sqrt(2.0 * p[:, 0]), 0.0)  # [nevt,] complex
    # TODO: same as _ix_massless_sqp0p3_zero


_ox_massless_sqp0p3_nonzero_signature = [smom] + [sscalar] + [svec]
@tf.function(input_signature=_ox_massless_sqp0p3_nonzero_signature)
def _ox_massless_sqp0p3_nonzero(p, nh, sqp0p3):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nh: tf.Tensorm of shape=()
        sqp0p3: tf.Tensorm of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(None) and dtype DTYPECOMPLEX
    """
    return complex_tf(nh * p[:, 1] / sqp0p3, -p[:, 2] / sqp0p3)  # [nevt,] complex
    # TODO: same as _ix_massless_sqp0p3_nonzero


@tf.function(input_signature=[tf.TensorSpec(shape=[2,None], dtype=DTYPECOMPLEX)])
def _ox_massless_nh_one(chi):
    """
    Parameters
    ----------
        chi: tf.Tensor, of shape=(2,None) and dtype DTYPECOMPLEX

    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    v = [complex_tf(0,0)] * 4
    v[0] = chi[0]
    v[1] = chi[1]
    v[2] = tf.ones_like(v[0]) * complex_tf(0.0, 0.0)
    v[3] = tf.ones_like(v[0]) * complex_tf(0.0, 0.0)
    return tf.stack(v, axis=0)
    # TODO: same of _ix_massless_nh_not_one with chi.T


@tf.function(input_signature=[tf.TensorSpec(shape=[2,None], dtype=DTYPECOMPLEX)])
def _ox_massless_nh_not_one(chi):
    """
    Parameters
    ----------
        chi: tf.Tensor, of shape=(2,None) and dtype DTYPECOMPLEX

    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    v = [complex_tf(0,0)] * 4
    v[2] = chi[1]
    v[3] = chi[0]
    v[0] = tf.ones_like(v[2]) * complex_tf(0.0, 0.0)
    v[1] = tf.ones_like(v[2]) * complex_tf(0.0, 0.0)
    return tf.stack(v, axis=0)
    # TODO: same of _ix_massless_nh_one with chi.T

#===============================================================================
# vxxxxx related functions

_vx_BRST_check_signature = [smom] + [sscalar] + [sscalar_int]
@tf.function(input_signature=_vx_BRST_check_signature)
def _vx_BRST_check(p, vmass, nevts):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        vmass: tf.Tensor, of shape=()
        nevts: tf.Tensor, of shape=() and dtype DTYPEINT
    
    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    massless = vmass == 0
    return tf.cond(massless,
                   lambda: _vx_BRST_check_massless(p, nevts),
                   lambda: _vx_BRST_check_massive(p, vmass)
                  )


_vx_BRST_massless_signature = [smom] + [sscalar_int]
@tf.function(input_signature=_vx_BRST_massless_signature)
def _vx_BRST_check_massless(p, nevts):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nevts: tf.Tensor, of shape=() and dtype DTYPEINT
    
    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    vc = [complex_tf(0,0)] * 4
    vc[0] = tf.ones(nevts, dtype=DTYPE)
    vc[1] = p[:, 1] / p[:, 0]
    vc[2] = p[:, 2] / p[:, 0]
    vc[3] = p[:, 3] / p[:, 0]
    return complex_me(tf.stack(vc, axis=0))


_vx_BRST_massive_signature = [smom] + [sscalar]
@tf.function(input_signature=_vx_BRST_massive_signature)
def _vx_BRST_check_massive(p, vmass):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        vmass: tf.Tensor, of shape=()
    
    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    vc = [complex_tf(0,0)] * 4
    vc[0] = p[:, 0] / vmass
    vc[1] = p[:, 1] / vmass
    vc[2] = p[:, 2] / vmass
    vc[3] = p[:, 3] / vmass
    return complex_me(tf.stack(vc, axis=0))


_vx_BRST_signature = [smom] + [sscalar]*6 + [svec]*2 + [sscalar_int]
@tf.function(input_signature=_vx_BRST_signature)
def _vx_no_BRST_check(p, vmass, nhel, nsv, hel0, sqh, nsvahl, pp, pt, nevts):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        vmass: tf.Tensor, of shape=()
        nhel: tf.Tensor, of shape=()
        nsv: tf.Tensor, of shape=()
        hel0: tf.Tensor, of shape=()
        sqh: tf.Tensor, of shape=()
        nsvahl: tf.Tensor, of shape=()
        pp: tf.Tensor, of shape=(None)
        pt: tf.Tensor, of shape=(None)
        nevts: tf.Tensor, of shape=() and dtype DTYPEINT
    
    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """  
    massive = vmass != 0
    return tf.cond(massive,
                   lambda: _vx_no_BRST_check_massive(
                            p, vmass, nhel, hel0, sqh, nsvahl, pp, pt, nevts
                                                ),
                   lambda: _vx_no_BRST_check_massless(p, nhel, nsv, sqh, nevts))


_vx_BRST_massive_signature = [smom] + [sscalar]*5 + [svec]*2 + [sscalar_int]
@tf.function(input_signature=_vx_BRST_massive_signature)
def _vx_no_BRST_check_massive(p, vmass, nhel, hel0, sqh, nsvahl, pp, pt, nevts):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        vmass: tf.Tensor, of shape=()
        nhel: tf.Tensor, of shape=()
        hel0: tf.Tensor, of shape=()
        sqh: tf.Tensor, of shape=()
        nsvahl: tf.Tensor, of shape=()
        pp: tf.Tensor, of shape=(None)
        pt: tf.Tensor, of shape=(None)
        nevts: tf.Tensor, of shape=() and dtype DTYPEINT
    
    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """     
    cond = tf.expand_dims(pp == 0, 0)
    return tf.where(cond,
                    _vx_no_BRST_check_massive_pp_zero(nhel, sqh, nsvahl, nevts),
                    _vx_no_BRST_check_massive_pp_nonzero(
                        p, vmass, nhel, hel0, sqh, nsvahl, pp, pt, nevts
                                                    )
                   )


_vx_BRST_massive_pp_zero_signature = [sscalar]*3 + [sscalar_int]
@tf.function(input_signature=_vx_BRST_massive_pp_zero_signature)
def _vx_no_BRST_check_massive_pp_zero(nhel, sqh, nsvahl, nevts):
    """
    Parameters
    ----------
        nhel: tf.Tensor, of shape=()
        sqh: tf.Tensor, of shape=()
        nsvahl: tf.Tensor, of shape=()
        nevts: tf.Tensor, of shape=() and dtype DTYPEINT
    
    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """ 
    hel0 = 1.0 - tfmath.abs(nhel)
    v = [complex_tf(0,0)] * 4
    v[0] = tf.ones(nevts, dtype=DTYPECOMPLEX)
    v[1] = tf.ones_like(v[0]) * complex_tf(-nhel * sqh, 0.0)
    v[2] = tf.ones_like(v[0]) * complex_tf(0.0, nsvahl * sqh)
    v[3] = tf.ones_like(v[0]) * complex_tf(hel0, 0.0)
    return tf.stack(v, axis=0)  # [4,nevts] complex


@tf.function(input_signature=_vx_BRST_massive_signature)
def _vx_no_BRST_check_massive_pp_nonzero(
                                p, vmass, nhel, hel0, sqh, nsvahl, pp, pt, nevts
                                       ):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        vmass: tf.Tensor, of shape=()
        nhel: tf.Tensor, of shape=()
        hel0: tf.Tensor, of shape=()
        sqh: tf.Tensor, of shape=()
        nsvahl: tf.Tensor, of shape=()
        pp: tf.Tensor, of shape=(None)
        pt: tf.Tensor, of shape=(None)
        nevts: tf.Tensor, of shape=() and dtype DTYPEINT
    
    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """ 
    emp = p[:, 0] / (vmass * pp)
    v2 = tf.expand_dims(complex_tf(hel0 * pp / vmass, 0.0), 0)
    v5 = tf.expand_dims(complex_tf(hel0 * p[:, 3] * emp + nhel * pt / pp * sqh, 0), 0)
    condition = tf.expand_dims(pt != 0, 0)
    v34 = tf.where(condition,
                   _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(
                                    p, nhel, hel0, sqh, nsvahl, pp, pt, emp
                                                              ),
                   _vx_no_BRST_check_massive_pp_nonzero_pt_zero(
                                    p, nhel, sqh, nsvahl, nevts
                                                           )
                   )
    return tf.concat([v2, v34, v5], axis=0)  # [4,nevts] complex


_vx_BRST_massive_pp_nonzero_pt_nonzero_signature = [smom] + [sscalar]*4 + [svec]*3
@tf.function(input_signature=_vx_BRST_massive_pp_nonzero_pt_nonzero_signature)
def _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(
                                        p, nhel, hel0, sqh, nsvahl, pp, pt, emp
                                                  ):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        hel0: tf.Tensor, of shape=()
        sqh: tf.Tensor, of shape=()
        nsvahl: tf.Tensor, of shape=()
        pp: tf.Tensor, of shape=(None)
        pt: tf.Tensor, of shape=(None)
        emp: tf.Tensor, of shape=(None)
    
    Returns
    -------
        tf.Tensor, of shape=(2,None) and dtype DTYPECOMPLEX
    """ 
    v = [complex_tf(0,0)] * 2
    pzpt = p[:, 3] / (pp * pt) * sqh * nhel
    v[0] = complex_tf(
        hel0 * p[:, 1] * emp - p[:, 1] * pzpt, -nsvahl * p[:, 2] / pt * sqh
    )
    v[1] = complex_tf(
        hel0 * p[:, 2] * emp - p[:, 2] * pzpt, nsvahl * p[:, 1] / pt * sqh
    )
    return tf.stack(v, axis=0)


_vx_BRST_massive_pp_zero_signature = [smom] + [sscalar]*3 + [sscalar_int]
@tf.function(input_signature=_vx_BRST_massive_pp_zero_signature)
def _vx_no_BRST_check_massive_pp_nonzero_pt_zero(p, nhel, sqh, nsvahl, nevts):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nhel: tf.Tensor, of shape=()
        sqh: tf.Tensor, of shape=()
        nsvahl: tf.Tensor, of shape=()
        nevts: tf.Tensor, of shape=() and dtype DTYPEINT
    
    Returns
    -------
        tf.Tensor, of shape=(2,None) and dtype DTYPECOMPLEX
    """ 
    v = [complex_tf(0,0)] * 2
    v[0] = tf.ones(nevts, dtype=DTYPECOMPLEX) * complex_tf(-nhel * sqh, 0.0)
    v[1] = complex_tf(
        0.0, nsvahl * signvec(sqh, p[:, 3])
    )  # <------ this enters the sign operation with y as a real vector
    return tf.stack(v, axis=0)


_vx_BRST_massless_signature = [smom] + [sscalar]*3 + [sscalar_int]
@tf.function(input_signature=_vx_BRST_massless_signature)
def _vx_no_BRST_check_massless(p, nhel, nsv, sqh, nevts):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nhel: tf.Tensor, of shape=()
        nsv: tf.Tensor, of shape=()
        sqh: tf.Tensor, of shape=()
        nevts: tf.Tensor, of shape=() and dtype DTYPEINT
    
    Returns
    -------
        tf.Tensor, of shape=(4,None) and dtype DTYPECOMPLEX
    """
    pp = p[:, 0]
    pt = tfmath.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2)
    v2 = tf.ones([1, nevts], dtype=DTYPECOMPLEX) * complex_tf(0.0, 0.0)
    v5 = tf.expand_dims(complex_tf(nhel * pt / pp * sqh, 0.0), 0)
    cond = tf.expand_dims(pt != 0, 0)
    v34 = tf.where(cond,
                   _vx_no_BRST_check_massless_pt_nonzero(p, nhel, nsv, sqh, pp, pt),
                   _vx_no_BRST_check_massless_pt_zero(p, nhel, nsv, sqh, nevts))
    return tf.concat([v2, v34, v5], axis=0)


_vx_BRST_massless_pt_nonzero_signature = [smom] + [sscalar]*3 + [svec]*2
@tf.function(input_signature=_vx_BRST_massless_pt_nonzero_signature)
def _vx_no_BRST_check_massless_pt_nonzero(p, nhel, nsv, sqh, pp, pt):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nhel: tf.Tensor, of shape=()
        nsv: tf.Tensor, of shape=()
        sqh: tf.Tensor, of shape=()
        pp: tf.Tensor, of shape=(None)
        pt: tf.Tensor, of shape=(None)
    
    Returns
    -------
        tf.Tensor, of shape=(2,None) and dtype DTYPECOMPLEX
    """
    pzpt = p[:, 3] / (pp * pt) * sqh * nhel
    v = [complex_tf(0,0)] * 2
    v[0] = complex_tf(-p[:, 1] * pzpt, -nsv * p[:, 2] / pt * sqh)
    v[1] = complex_tf(-p[:, 2] * pzpt, nsv * p[:, 1] / pt * sqh)
    return tf.stack(v, axis=0)


_vx_BRST_massless_pt_zero_signature = [smom] + [sscalar]*3 + [sscalar_int]
@tf.function(input_signature=_vx_BRST_massless_pt_zero_signature)
def _vx_no_BRST_check_massless_pt_zero(p, nhel, nsv, sqh, nevts):
    """
    Parameters
    ----------
        p: tf.Tensor, of shape=(None,4)
        nhel: tf.Tensor, of shape=()
        nsv: tf.Tensor, of shape=()
        sqh: tf.Tensor, of shape=()
        nevts: tf.Tensor, of shape=() and dtype DTYPEINT
    
    Returns
    -------
        tf.Tensor, of shape=(2,None) and dtype DTYPECOMPLEX
    """
    v = [complex_tf(0,0)] * 2
    v[0] = tf.ones(nevts, dtype=DTYPECOMPLEX) * complex_tf(-nhel * sqh, 0.0)
    v[1] = complex_tf(
        0.0, nsv * signvec(sqh, p[:, 3])
    )  # <------ this enters the sign operation with y as a real vector
    return tf.stack(v, axis=0)