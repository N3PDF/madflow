"""
    Wave functions
"""

from .config import DTYPECOMPLEX, complex_tf, complex_me, DTYPE, DTYPEINT, int_me, float_me

import tensorflow as tf
import tensorflow.math as tfmath

SQH = float_me(tfmath.sqrt(0.5))
CZERO = complex_tf(0.0, 0.0)

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
    return x * tf.math.sign(y)


@tf.function(input_signature=signvec_signature)
def signvec(x, y):
    """Fortran's sign transfer function"""
    return x * tf.math.sign(y)


@tf.function(input_signature=scalar_signature)
def sxxxxx(p, nss):
    """
    Defines a scalar wavefunction. Input momenta have shape (num events, 4).

    Parameters
    ----------
        p: tf.Tensor, scalar boson four-momenta of shape=(None,4)
        nss: tf.Tensor, final|initial state of shape=(), values=(+1|-1)

    Returns
    -------
        phi: tf.Tensor, scalar wavefunction of shape=(None,3)
    """
    v0 = tf.expand_dims(complex_tf(p[:, 0] * nss, p[:, 3] * nss), 1)
    v1 = tf.expand_dims(complex_tf(p[:, 1] * nss, p[:, 2] * nss), 1)
    v = tf.expand_dims(complex_tf(1.0, 0.0), 1)
    return tf.concat([v0, v1, v], axis=1)


@tf.function(input_signature=wave_signature)
def ixxxxx(p, fmass, nhel, nsf):
    """
    Defines an inflow fermion wavefunction. Input momenta have shape
    (num events, 4).

    Parameters
    ----------
        p: tf.Tensor, fermion four-momenta of shape=(None,4)
        fmass: tf.Tensor, fermion mass of shape=()
        nhel: tf.Tensor, fermion helicity of shape=()
        nsf: tf.Tensor, particle|anti-particle of shape=(), values=(+1|-1)

    Returns
    -------
        |fi>: tf.Tensor, fermion wavefunction of shape=(None,6)
    """
    v0 = tf.expand_dims(complex_tf(-p[:, 0] * nsf, -p[:, 3] * nsf), 1)
    v1 = tf.expand_dims(complex_tf(-p[:, 1] * nsf, -p[:, 2] * nsf), 1)
    nh = nhel * nsf

    massive = fmass != 0
    v = tf.cond(massive,
                lambda: _ix_massive(p, fmass, nsf, nh),
                lambda: _ix_massless(p, nhel, nsf, nh)
               )
    return tf.concat([v0, v1, v], axis=1)


@tf.function(input_signature=wave_signature)
def oxxxxx(p, fmass, nhel, nsf):
    """
    Defines an outgoing fermion wavefunction. Input momenta have shape
    (num events, 4).

    Parameters
    ----------
        p: tf.Tensor, fermion four-momenta of shape=(None,4)
        fmass: tf.Tensor, fermion mass of shape=()
        nhel: tf.Tensor, fermion helicity of shape=()
        nsf: tf.Tensor, particle|anti-particle of shape=(), values=(+1|-1)

    Returns
    -------
         <fo|: tf.Tensor, fermion wavefunction of shape=(None,6)
    """
    v0 = tf.expand_dims(complex_tf(p[:, 0] * nsf, p[:, 3] * nsf), 1)
    v1 = tf.expand_dims(complex_tf(p[:, 1] * nsf, p[:, 2] * nsf), 1)
    nh = nhel * nsf

    massive = fmass != 0
    v = tf.cond(massive,
                lambda: _ox_massive(p, fmass, nhel, nsf, nh),
                lambda: _ox_massless(p, nhel, nsf, nh)
               )
    return tf.concat([v0, v1, v], axis=1)


@tf.function(input_signature=wave_signature)
def vxxxxx(p, vmass, nhel, nsv):
    """
    Defines a vector wavefunction. nhel=4 is for checking BRST.
    Input momenta have shape (num events, 4).

    Parameters
    ----------
        p: tf.Tensor, vector boson four-momenta of shape=(None,4)
        vmass: tf.Tensor, boson mass of shape=()
        nhel: tf.Tensor, boson helicity of shape=(), 0 is forbidden if vmass=0.0
        nsv: tf.Tensor, final|initial state of shape=(), values=(+1|-1)

    Returns
    -------
        epsilon^{mu(v)}: tf.Tensor, vector wavefunction of shape=(None,6)
    """
    hel0 = 1.0 - tfmath.abs(nhel)

    nsvahl = nsv * tfmath.abs(nhel)
    pt2 = p[:, 1] ** 2 + p[:, 2] ** 2
    pp = tfmath.minimum(p[:, 0], tfmath.sqrt(pt2 + p[:, 3] ** 2))
    pt = tfmath.minimum(pp, tfmath.sqrt(pt2))

    v0 = tf.expand_dims(complex_tf(p[:, 0] * nsv, p[:, 3] * nsv), 1)
    v1 = tf.expand_dims(complex_tf(p[:, 1] * nsv, p[:, 2] * nsv), 1)

    BRST = nhel == 4
    v = tf.cond(BRST,
                lambda: _vx_BRST_check(p, vmass),
                lambda: _vx_no_BRST_check(
                           p, vmass, nhel, nsv, hel0, nsvahl, pp, pt
                                     )
               )
    return tf.concat([v0, v1, v], axis=1)


#===============================================================================
# ixxxxx related functions
_ix_massive_signature = [smom] + [sscalar]*3
@tf.function(input_signature=_ix_massive_signature)
def _ix_massive(p, fmass, nsf, nh):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        fmass: tf.Tensor, fermion mass of shape=()
        nsf: tf.Tensor, particle|anti-particle of shape=()
        nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    pp = tfmath.minimum(
        p[:, 0], tfmath.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2 + p[:, 3] ** 2)
    )
    ip = (1 + nh) // 2
    im = (1 - nh) // 2

    cond = tf.expand_dims(pp == 0, 1)
    return tf.where(cond,
                    # exploit im, ip exchange symmetry in _ox_massive_pp_zero
                    _ox_massive_pp_zero(fmass, nsf, im, ip),
                    _ix_massive_pp_nonzero(p, fmass, nsf, nh, ip, im, pp))


_ix_massive_pp_nonzero_signature = [smom] + [sscalar]*3 + [sscalar]*2 + [svec]
@tf.function(input_signature=_ix_massive_pp_nonzero_signature)
def _ix_massive_pp_nonzero(p, fmass, nsf, nh, ip, im, pp):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        fmass: tf.Tensor, fermion mass of shape=()
        nsf: tf.Tensor, particle|anti-particle of shape=()
        nh: tf.Tensor, helicity times particle|anti-particle of shape=()
        ip: tf.Tensor, positive nh projector of shape=()
        im: tf.Tensor, negative nh projector of shape=()
        pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    sf = tf.stack(
        [(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5], axis=0
    )
    omega = tf.stack(
        [tfmath.sqrt(p[:, 0] + pp), fmass / (tfmath.sqrt(p[:, 0] + pp))], axis=0
    )
    sfomeg = tf.stack(
        [sf[0] * omega[int_me(ip)], sf[1] * omega[int_me(im)]], axis=0
    )
    pp3 = tfmath.maximum(pp + p[:, 3], 0.0)
    chi1 = tf.where(
        pp3 == 0,
        complex_tf(-nh, 0),
        complex_tf(
            nh * p[:, 1] / tfmath.sqrt(2.0 * pp * pp3), p[:, 2] / tfmath.sqrt(2.0 * pp * pp3)
        ),
    )
    chi2 = complex_tf(tfmath.sqrt(pp3 * 0.5 / pp), 0.0)
    chi = tf.stack([chi2, chi1], axis=0)
    v = [complex_tf(0,0)] * 4
    v[0] = complex_tf(sfomeg[0], 0.0) * chi[int_me(im)]
    v[1] = complex_tf(sfomeg[0], 0.0) * chi[int_me(ip)]
    v[2] = complex_tf(sfomeg[1], 0.0) * chi[int_me(im)]
    v[3] = complex_tf(sfomeg[1], 0.0) * chi[int_me(ip)]
    return tf.stack(v, axis=1)


_ix_massless_signature = [smom] + [sscalar]*3
@tf.function(input_signature=_ix_massless_signature)
def _ix_massless(p, nhel, nsf, nh):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        nhel: tf.Tensor, fermion helicity of shape=()
        nsf: tf.Tensor, particle|anti-particle of shape=()
        nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    sqp0p3 = tfmath.sqrt(tfmath.maximum(p[:, 0] + p[:, 3], 0.0)) * nsf
    chi1 = tf.where(sqp0p3 == 0,
                    _ix_massless_sqp0p3_zero(p, nhel),
                    _ix_massless_sqp0p3_nonzero(p, nh, sqp0p3)
                   )
    chi = tf.stack([complex_tf(sqp0p3, 0.0), chi1], axis=1)
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
        p: tf.Tensor, four-momenta of shape=(None,4)
        nhel: tf.Tensor, fermion helicity of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None) and dtype DTYPECOMPLEX

    Note: this function is the same for input `ixxxxx` and output `oxxxxx`
    waveforms
    """
    return complex_tf(-nhel * tfmath.sqrt(2.0 * p[:, 0]), 0.0)


_ix_massless_sqp0p3_nonzero_signature = [smom] + [sscalar] + [svec]
@tf.function(input_signature=_ix_massless_sqp0p3_nonzero_signature)
def _ix_massless_sqp0p3_nonzero(p, nh, sqp0p3):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        nh: tf.Tensor, helicity times particle|anti-particle of shape=()
        sqp0p3: tf.Tensor, max(E+pz,0)*nsf of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(None) and dtype DTYPECOMPLEX
    """
    return complex_tf(nh * p[:, 1] / sqp0p3, p[:, 2] / sqp0p3)


@tf.function(input_signature=[tf.TensorSpec(shape=[None,2], dtype=DTYPECOMPLEX)])
def _ix_massless_nh_one(chi):
    """
    Parameters
    ----------
        chi: tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    v = [complex_tf(0,0)] * 4
    v[2] = chi[:,0]
    v[3] = chi[:,1]
    v[0] = tf.ones_like(v[2]) * CZERO
    v[1] = tf.ones_like(v[2]) * CZERO
    return tf.stack(v, axis=1)


@tf.function(input_signature=[tf.TensorSpec(shape=[None,2], dtype=DTYPECOMPLEX)])
def _ix_massless_nh_not_one(chi):
    """
    Parameters
    ----------
        chi: tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    v = [complex_tf(0,0)] * 4
    v[0] = chi[:,1]
    v[1] = chi[:,0]
    v[2] = tf.ones_like(v[0]) * CZERO
    v[3] = tf.ones_like(v[0]) * CZERO
    return tf.stack(v, axis=1)

#===============================================================================
# oxxxxx related functions_
_ox_massive_signature = [smom] + [sscalar]*4
@tf.function(input_signature=_ox_massive_signature)
def _ox_massive(p, fmass, nhel, nsf, nh):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        fmass: tf.Tensor, fermion mass of shape=()
        nhel: tf.Tensor, fermion helicity of shape=()
        nsf: tf.Tensor, particle|anti-particle of shape=()
        nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    pp = tfmath.minimum(
        p[:, 0], tfmath.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2 + p[:, 3] ** 2)
    )
    ip = -((1 - nh) // 2) * nhel
    im = (1 + nh) // 2 * nhel

    cond = tf.expand_dims(pp == 0, 1)
    return tf.where(cond,
                    _ox_massive_pp_zero(fmass, nhel, ip, im),
                    _ox_massive_pp_nonzero(p, fmass, nsf, nh, pp)
                   )


_ox_massive_pp_zero_signature = [sscalar]*4
@tf.function(input_signature=_ox_massive_pp_zero_signature)
def _ox_massive_pp_zero(fmass, nsf, ip, im):
    """
    Parameters
    ----------
        fmass: tf.Tensor, fermion mass of shape=()
        nsf: tf.Tensor, particle|anti-particle of shape=()
        ip: tf.Tensor, positive nh projector of shape=()
        im: tf.Tensor, negative nh projector of shape=()

    Returns
    -------
        tf.Tensor, of shape=(1,4) and dtype DTYPECOMPLEX
    """
    sqm = tfmath.sqrt(tfmath.abs(fmass))
    sqm = tf.stack(
        [sqm, sign(sqm, fmass)]
    )

    v = [complex_tf(0,0)] * 4
    v[0] = complex_tf(im * sqm[int_me(tfmath.abs(im))], 0.0)
    v[1] = complex_tf(ip * nsf * sqm[int_me(tfmath.abs(im))], 0.0)
    v[2] = complex_tf(im * nsf * sqm[int_me(tfmath.abs(ip))], 0.0)
    v[3] = complex_tf(ip * sqm[int_me(tfmath.abs(ip))], 0.0)
    return tf.reshape(tf.stack(v), [1,4])


_ox_massive_pp_nonzero_signature = [smom] + [sscalar]*3 + [svec]
@tf.function(input_signature=_ox_massive_pp_nonzero_signature)
def _ox_massive_pp_nonzero(p, fmass, nsf, nh, pp):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        fmass: tf.Tensor, fermion mass of shape=()
        nsf: tf.Tensor, particle|anti-particle of shape=()
        nh: tf.Tensor, helicity times particle|anti-particle of shape=()
        pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    sf = tf.stack(
        [(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5], axis=0
    )
    omega = tf.stack(
        [tfmath.sqrt(p[:, 0] + pp), fmass / (tfmath.sqrt(p[:, 0] + pp))], axis=0
    )
    ip = (1 + nh) // 2
    im = (1 - nh) // 2
    sfomeg = tf.stack(
        [sf[0] * omega[int_me(ip)], sf[1] * omega[int_me(im)]], axis=0
    )
    pp3 = tfmath.maximum(pp + p[:, 3], 0.0)
    chi1 = tf.where(
        pp3 == 0,
        complex_tf(-nh, 0),
        complex_tf(
            nh * p[:, 1] / tfmath.sqrt(2.0 * pp * pp3), -p[:, 2] / tfmath.sqrt(2.0 * pp * pp3)
        ),
    )
    chi2 = complex_tf(tfmath.sqrt(pp3 * 0.5 / pp), 0.0)
    chi = tf.stack([chi2, chi1], axis=0)
    v = [complex_tf(0,0)] * 4
    v[0] = complex_tf(sfomeg[1], 0.0) * chi[int_me(im)]
    v[1] = complex_tf(sfomeg[1], 0.0) * chi[int_me(ip)]
    v[2] = complex_tf(sfomeg[0], 0.0) * chi[int_me(im)]
    v[3] = complex_tf(sfomeg[0], 0.0) * chi[int_me(ip)]
    return tf.stack(v, axis=1)


_ox_massless_signature = [smom] + [sscalar]*3
@tf.function(input_signature=_ox_massless_signature)
def _ox_massless(p, nhel, nsf, nh):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        nhel: tf.Tensor, fermion helicity of shape=()
        nsf: tf.Tensor, particle|anti-particle of shape=()
        nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    sqp0p3 = tfmath.sqrt(tfmath.maximum(p[:, 0] + p[:, 3], 0.0)) * nsf
    mult = tf.expand_dims(float_me([1,1,-1,1]), 0)
    chi0 = tf.where(sqp0p3 == 0,
                    _ix_massless_sqp0p3_zero(p, nhel),
                    _ix_massless_sqp0p3_nonzero(p*mult, nh, sqp0p3)
                   )
    chi = tf.stack([chi0, complex_tf(sqp0p3, 0.0)], axis=1)
    # ongoing fermion has nh inverted wrt the ingoing fermion
    return tf.cond(nh == 1,
                   lambda: _ix_massless_nh_not_one(chi),
                   lambda: _ix_massless_nh_one(chi)
                  )

#===============================================================================
# vxxxxx related functions

_vx_BRST_check_signature = [smom] + [sscalar]
@tf.function(input_signature=_vx_BRST_check_signature)
def _vx_BRST_check(p, vmass):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        vmass: tf.Tensor, boson mass of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    massless = vmass == 0
    return tf.cond(massless,
                   lambda: _vx_BRST_check_massless(p),
                   lambda: _vx_BRST_check_massive(p, vmass)
                  )


_vx_BRST_massless_signature = [smom]
@tf.function(input_signature=_vx_BRST_massless_signature)
def _vx_BRST_check_massless(p):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    return complex_me(p / p[:,:1])


_vx_BRST_massive_signature = [smom] + [sscalar]
@tf.function(input_signature=_vx_BRST_massive_signature)
def _vx_BRST_check_massive(p, vmass):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        vmass: tf.Tensor, boson mass of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    return complex_me(p / vmass)


_vx_BRST_signature = [smom] + [sscalar]*5 + [svec]*2
@tf.function(input_signature=_vx_BRST_signature)
def _vx_no_BRST_check(p, vmass, nhel, nsv, hel0, nsvahl, pp, pt):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        vmass: tf.Tensor, boson mass of shape=()
        nhel: tf.Tensor, boson helicity of shape=()
        nsv: tf.Tensor, final|initial state of shape=()
        hel0: tf.Tensor, zero helicity of shape=()
        nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value
                of shape=()
        pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)
        pt: tf.Tensor, of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    massive = vmass != 0
    return tf.cond(massive,
                   lambda: _vx_no_BRST_check_massive(
                            p, vmass, nhel, hel0, nsvahl, pp, pt
                                                ),
                   lambda: _vx_no_BRST_check_massless(p, nhel, nsv))


_vx_BRST_massive_signature = [smom] + [sscalar]*4 + [svec]*2
@tf.function(input_signature=_vx_BRST_massive_signature)
def _vx_no_BRST_check_massive(p, vmass, nhel, hel0, nsvahl, pp, pt):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        vmass: tf.Tensor, boson mass of shape=()
        nhel: tf.Tensor, boson helicity of shape=()
        hel0: tf.Tensor, zero helicity of shape=()
        nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value
                of shape=()
        pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)
        pt: tf.Tensor, of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    nevts = tf.shape(p, out_type=DTYPEINT)[0]
    cond = tf.expand_dims(pp == 0, 1)
    return tf.where(cond,
                    _vx_no_BRST_check_massive_pp_zero(nhel, nsvahl, nevts),
                    _vx_no_BRST_check_massive_pp_nonzero(
                        p, vmass, nhel, hel0, nsvahl, pp, pt
                                                    )
                   )


_vx_BRST_massive_pp_zero_signature = [sscalar]*2 + [sscalar_int]
@tf.function(input_signature=_vx_BRST_massive_pp_zero_signature)
def _vx_no_BRST_check_massive_pp_zero(nhel, nsvahl, nevts):
    """
    Parameters
    ----------
        nhel: tf.Tensor, boson helicity of shape=()
        nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value
                of shape=()
        nevts: tf.Tensor, number of events of shape=() and dtype DTYPEINT

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    hel0 = 1.0 - tfmath.abs(nhel)
    v = [complex_tf(0,0)] * 4
    v[0] = tf.ones(nevts, dtype=DTYPECOMPLEX)
    v[1] = tf.ones_like(v[0]) * complex_tf(-nhel * SQH, 0.0)
    v[2] = tf.ones_like(v[0]) * complex_tf(0.0, nsvahl * SQH)
    v[3] = tf.ones_like(v[0]) * complex_tf(hel0, 0.0)
    return tf.stack(v, axis=1)


@tf.function(input_signature=_vx_BRST_massive_signature)
def _vx_no_BRST_check_massive_pp_nonzero(
                                p, vmass, nhel, hel0, nsvahl, pp, pt
                                       ):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        vmass: tf.Tensor, boson mass of shape=()
        nhel: tf.Tensor, boson helicity of shape=()
        hel0: tf.Tensor, zero helicity of shape=()
        nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value
                of shape=()
        pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)
        pt: tf.Tensor, of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    emp = p[:, 0] / (vmass * pp)
    v2 = tf.expand_dims(complex_tf(hel0 * pp / vmass, 0.0), 1)
    v5 = tf.expand_dims(complex_tf(hel0 * p[:, 3] * emp + nhel * pt / pp * SQH, 0), 1)
    condition = tf.expand_dims(pt != 0, 1)
    v34 = tf.where(condition,
                   _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(
                                            p, nhel, hel0, nsvahl, pp, pt, emp
                                                              ),
                   _vx_no_BRST_check_massive_pp_nonzero_pt_zero(p, nhel, nsvahl)
                   )
    return tf.concat([v2, v34, v5], axis=1)


_vx_BRST_massive_pp_nonzero_pt_nonzero_signature = [smom] + [sscalar]*3 + [svec]*3
@tf.function(input_signature=_vx_BRST_massive_pp_nonzero_pt_nonzero_signature)
def _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(
                                        p, nhel, hel0, nsvahl, pp, pt, emp
                                                  ):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        hel0: tf.Tensor, zero helicity of shape=()
        nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value
                of shape=()
        pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)
        pt: tf.Tensor, of shape=(None)
        emp: tf.Tensor, of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX
    """
    v = [complex_tf(0,0)] * 2
    pzpt = p[:, 3] / (pp * pt) * SQH * nhel
    v[0] = complex_tf(
        hel0 * p[:, 1] * emp - p[:, 1] * pzpt, -nsvahl * p[:, 2] / pt * SQH
    )
    v[1] = complex_tf(
        hel0 * p[:, 2] * emp - p[:, 2] * pzpt, nsvahl * p[:, 1] / pt * SQH
    )
    return tf.stack(v, axis=1)


_vx_BRST_massive_pp_zero_signature = [smom] + [sscalar]*2
@tf.function(input_signature=_vx_BRST_massive_pp_zero_signature)
def _vx_no_BRST_check_massive_pp_nonzero_pt_zero(p, nhel, nsvahl):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        nhel: tf.Tensor, boson helicity of shape=()
        nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value
                of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX
    """
    v = [complex_tf(0,0)] * 2
    v[0] = tf.ones_like(p[:,0], dtype=DTYPECOMPLEX) * complex_tf(-nhel * SQH, 0.0)
    v[1] = complex_tf( 0.0, nsvahl * signvec(SQH, p[:, 3]) )
    return tf.stack(v, axis=1)


_vx_BRST_massless_signature = [smom] + [sscalar]*2
@tf.function(input_signature=_vx_BRST_massless_signature)
def _vx_no_BRST_check_massless(p, nhel, nsv):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        nhel: tf.Tensor, boson helicity of shape=()
        nsv: tf.Tensor, final|initial state of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    pp = p[:, 0]
    pt = tfmath.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2)
    v2 = tf.expand_dims(tf.zeros_like(p[:,0], dtype=DTYPECOMPLEX), 1)
    v5 = tf.expand_dims(complex_tf(nhel * pt / pp * SQH, 0.0), 1)
    cond = tf.expand_dims(pt != 0, 1)
    v34 = tf.where(cond,
                   _vx_no_BRST_check_massless_pt_nonzero(p, nhel, nsv, pp, pt),
                   _vx_no_BRST_check_massless_pt_zero(p, nhel, nsv))
    return tf.concat([v2, v34, v5], axis=1)


_vx_BRST_massless_pt_nonzero_signature = [smom] + [sscalar]*2 + [svec]*2
@tf.function(input_signature=_vx_BRST_massless_pt_nonzero_signature)
def _vx_no_BRST_check_massless_pt_nonzero(p, nhel, nsv, pp, pt):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        nhel: tf.Tensor, boson helicity of shape=()
        nsv: tf.Tensor, final|initial state of shape=()
        SQH: tf.Tensor, sqrt(1/2) of shape=()
        pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)
        pt: tf.Tensor, of shape=(None)

    Returns
    -------
        tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX
    """
    pzpt = p[:, 3] / (pp * pt) * SQH * nhel
    v = [complex_tf(0,0)] * 2
    v[0] = complex_tf(-p[:, 1] * pzpt, -nsv * p[:, 2] / pt * SQH)
    v[1] = complex_tf(-p[:, 2] * pzpt, nsv * p[:, 1] / pt * SQH)
    return tf.stack(v, axis=1)


_vx_BRST_massless_pt_zero_signature = [smom] + [sscalar]*2
@tf.function(input_signature=_vx_BRST_massless_pt_zero_signature)
def _vx_no_BRST_check_massless_pt_zero(p, nhel, nsv):
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        nhel: tf.Tensor, boson helicity of shape=()
        nsv: tf.Tensor, final|initial state of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX
    """
    v = [complex_tf(0,0)] * 2
    v[0] = tf.ones_like(p[:,0], dtype=DTYPECOMPLEX) * complex_tf(-nhel * SQH, 0.0)
    v[1] = complex_tf(0.0, nsv * signvec(SQH, p[:, 3]))
    return tf.stack(v, axis=1)