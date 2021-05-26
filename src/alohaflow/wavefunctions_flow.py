"""
    Wave functions
"""

from .config import DTYPECOMPLEX, complex_tf, complex_me, DTYPE, DTYPEINT, int_me, float_me

import tensorflow as tf
import tensorflow.math as tfmath

wave_signature = [
    tf.TensorSpec(shape=[None, 4], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
]

scalar_signature = [
    tf.TensorSpec(shape=[None, 4], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
]

sign_signature = [
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
]

signvec_signature = [
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
]


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
        fi: tf.Tensor, of shape=(3,None)    
    """
    # Note: here p[:,i] selects the momentum dimension and is a [nevt,] tensor
    v0 = tf.expand_dims(complex_tf(p[:, 0] * nss, p[:, 3] * nss), 0)  # [nevt,] complex
    v1 = tf.expand_dims(complex_tf(p[:, 1] * nss, p[:, 2] * nss), 0)  # [nevt,] complex
    v = tf.expand_dims(complex_tf(1.0, 0.0), 0)
    fi = tf.concat([v0, v1, v], axis=0)
    return fi


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
        fi: tf.Tensor, of shape=(6,None)    
    """
    # Note: here p[:,i] selects the momentum dimension and is a [nevt,] tensor
    v0 = tf.expand_dims(complex_tf(-p[:, 0] * nsf, -p[:, 3] * nsf), 0)  # [nevt,] complex
    v1 = tf.expand_dims(complex_tf(-p[:, 1] * nsf, -p[:, 2] * nsf), 0)  # [nevt,] complex
    nh = nhel * nsf  # either +1 or -1
    ip = (1 + nh) // 2
    im = (1 - nh) // 2

    def is_massive():
        pp = tfmath.minimum(
            p[:, 0], tfmath.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2 + p[:, 3] ** 2)
        )  # [nevt,]

        def pp_zero():
            sqm = tfmath.sqrt(tfmath.abs(fmass))
            sqm = tf.stack(
                [sqm, sign(sqm, fmass)]
            )  # [fmass, fmass] ---> TODO: why calling sign on the result of a tfmath.sqrt ????
            v2 = complex_tf(ip * sqm[int_me(ip)], 0.0)  # just a complex number
            v3 = complex_tf(im * nsf * sqm[int_me(ip)], 0.0)
            v4 = complex_tf(ip * nsf * sqm[int_me(im)], 0.0)
            v5 = complex_tf(im * sqm[int_me(im)], 0.0)
            v = tf.stack([v2, v3, v4, v5])  # [4,] complex
            return tf.reshape(v, [4, 1])

        def pp_not_zero():
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
            v2 = complex_tf(sfomeg[0], 0.0) * chi[int_me(im)]  # [nevt,] complex
            v3 = complex_tf(sfomeg[0], 0.0) * chi[int_me(ip)]
            v4 = complex_tf(sfomeg[1], 0.0) * chi[int_me(im)]
            v5 = complex_tf(sfomeg[1], 0.0) * chi[int_me(ip)]
            return tf.stack([v2, v3, v4, v5], axis=0)  # [nevt, 4] complex

        cond = tf.expand_dims(pp == 0, 0)
        return tf.where(cond, pp_zero(), pp_not_zero())  # [nevt, 4] complex

    def is_not_massive():
        sqp0p3 = tfmath.sqrt(tfmath.maximum(p[:, 0] + p[:, 3], 0.0)) * nsf  # [nevt,]

        def sqp0p3_zero():
            return complex_tf(-nhel * tfmath.sqrt(2.0 * p[:, 0]), 0.0)  # [nevt,] complex

        def sqp0p3_not_zero():
            return complex_tf(nh * p[:, 1] / sqp0p3, p[:, 2] / sqp0p3)  # [nevt,] complex

        chi1 = tf.where(sqp0p3 == 0, sqp0p3_zero(), sqp0p3_not_zero())
        chi = tf.stack([complex_tf(sqp0p3, 0.0), chi1], axis=0)  # [2, nevt]

        def nh_one():
            v4 = chi[0]  # [nevt,] complex
            v5 = chi[1]  # [nevt,] complex
            v2 = tf.ones_like(v4) * complex_tf(0.0, 0.0)  # [nevt,] complex
            v3 = tf.ones_like(v4) * complex_tf(0.0, 0.0)  # [nevt,] complex
            return tf.stack([v2, v3, v4, v5], axis=0)

        def nh_not_one():
            v2 = chi[1]
            v3 = chi[0]
            v4 = tf.ones_like(v2) * complex_tf(0.0, 0.0)
            v5 = tf.ones_like(v2) * complex_tf(0.0, 0.0)
            return tf.stack([v2, v3, v4, v5], axis=0)

        return tf.where(nh == 1, nh_one(), nh_not_one())

    massive = fmass != 0
    v = tf.where(massive, is_massive(), is_not_massive())
    fi = tf.concat([v0, v1, v], axis=0)
    return fi


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
        fi: tf.Tensor, of shape=(6,None)
    """
    v0 = tf.expand_dims(complex_tf(p[:, 0] * nsf, p[:, 3] * nsf), 0)  # [nevt,] complex
    v1 = tf.expand_dims(complex_tf(p[:, 1] * nsf, p[:, 2] * nsf), 0)  # [nevt,] complex
    nh = nhel * nsf  # either +1 or -1
    sqp0p3 = tfmath.sqrt(tfmath.maximum(p[:, 0] + p[:, 3], 0.0)) * nsf  # [nevt,]

    def is_massive():
        pp = tfmath.minimum(
            p[:, 0], tfmath.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2 + p[:, 3] ** 2)
        )  # [nevt,]

        def pp_zero():
            sqm = tfmath.sqrt(tfmath.abs(fmass))
            sqm = tf.stack(
                [sqm, sign(sqm, fmass)]
            )  # [fmass, fmass] ---> why calling sign on the result of a tfmath.sqrt ????
            ip = -((1 - nh) // 2) * nhel
            im = (1 + nh) // 2 * nhel
            v2 = complex_tf(im * sqm[int_me(tfmath.abs(im))], 0.0)  # just a complex number
            v3 = complex_tf(ip * nsf * sqm[int_me(tfmath.abs(im))], 0.0)
            v4 = complex_tf(im * nsf * sqm[int_me(tfmath.abs(ip))], 0.0)
            v5 = complex_tf(ip * sqm[int_me(tfmath.abs(ip))], 0.0)
            v = tf.stack([v2, v3, v4, v5])  # [4,] complex
            return tf.reshape(v, [4, 1])

        def pp_not_zero():
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
            v2 = complex_tf(sfomeg[1], 0.0) * chi[int_me(im)]  # [nevt,] complex
            v3 = complex_tf(sfomeg[1], 0.0) * chi[int_me(ip)]
            v4 = complex_tf(sfomeg[0], 0.0) * chi[int_me(im)]
            v5 = complex_tf(sfomeg[0], 0.0) * chi[int_me(ip)]
            return tf.stack([v2, v3, v4, v5], axis=0)  # [4, nevt] complex

        cond = tf.expand_dims(pp == 0, 0)
        return tf.where(cond, pp_zero(), pp_not_zero())  # [4, nevt] complex

    def is_not_massive():
        def sqp0p3_zero():
            return complex_tf(-nhel * tfmath.sqrt(2.0 * p[:, 0]), 0.0)  # [nevt,] complex

        def sqp0p3_not_zero():
            return complex_tf(nh * p[:, 1] / sqp0p3, -p[:, 2] / sqp0p3)  # [nevt,] complex

        chi1 = tf.where(sqp0p3 == 0, sqp0p3_zero(), sqp0p3_not_zero())
        chi = tf.stack([complex_tf(sqp0p3, 0.0), chi1], axis=0)  # [2, nevt]

        def nh_one():
            v2 = chi[0]  # [nevt,] complex
            v3 = chi[1]  # [nevt,] complex
            v4 = tf.ones_like(v2) * complex_tf(0.0, 0.0)  # [nevt,] complex
            v5 = tf.ones_like(v2) * complex_tf(0.0, 0.0)  # [nevt,] complex
            return tf.stack([v2, v3, v4, v5], axis=0)

        def nh_not_one():
            v4 = chi[1]
            v5 = chi[0]
            v2 = tf.ones_like(v4) * complex_tf(0.0, 0.0)
            v3 = tf.ones_like(v4) * complex_tf(0.0, 0.0)
            return tf.stack([v2, v3, v4, v5], axis=0)

        return tf.where(nh == 1, nh_one(), nh_not_one())

    massive = fmass != 0
    v = tf.where(massive, is_massive(), is_not_massive())
    fo = tf.concat([v0, v1, v], axis=0)
    return fo


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

    def is_BRST():
        def is_massless():
            vc2 = tf.ones(nevts, dtype=DTYPE)
            vc3 = p[:, 1] / p[:, 0]
            vc4 = p[:, 2] / p[:, 0]
            vc5 = p[:, 3] / p[:, 0]
            return complex_me(tf.stack([vc2, vc3, vc4, vc5], axis=0))

        def is_not_massless():
            vc2 = p[:, 0] / vmass
            vc3 = p[:, 1] / vmass
            vc4 = p[:, 2] / vmass
            vc5 = p[:, 3] / vmass
            return complex_me(tf.stack([vc2, vc3, vc4, vc5], axis=0))

        massless = vmass == 0
        v = tf.where(massless, is_massless(), is_not_massless())
        return tf.concat([v0, v1, v], axis=0)  # [6,nevts] complex

    def is_not_BRST():
        def is_massive():
            def pp_zero():
                hel0 = 1.0 - tfmath.abs(nhel)
                v2 = tf.ones(nevts, dtype=DTYPECOMPLEX)
                v3 = tf.ones_like(v2) * complex_tf(-nhel * sqh, 0.0)
                v4 = tf.ones_like(v2) * complex_tf(0.0, nsvahl * sqh)
                v5 = tf.ones_like(v2) * complex_tf(hel0, 0.0)
                return tf.stack([v2, v3, v4, v5], axis=0)  # [4,nevts] complex

            def pp_not_zero():
                emp = p[:, 0] / (vmass * pp)
                v2 = tf.expand_dims(complex_tf(hel0 * pp / vmass, 0.0), 0)
                v5 = tf.expand_dims(complex_tf(hel0 * p[:, 3] * emp + nhel * pt / pp * sqh, 0), 0)

                def pt_not_zero():
                    pzpt = p[:, 3] / (pp * pt) * sqh * nhel
                    v3 = complex_tf(
                        hel0 * p[:, 1] * emp - p[:, 1] * pzpt, -nsvahl * p[:, 2] / pt * sqh
                    )
                    v4 = complex_tf(
                        hel0 * p[:, 2] * emp - p[:, 2] * pzpt, nsvahl * p[:, 1] / pt * sqh
                    )
                    return tf.stack([v3, v4], axis=0)

                def pt_zero():
                    v3 = tf.ones(nevts, dtype=DTYPECOMPLEX) * complex_tf(-nhel * sqh, 0.0)
                    v4 = complex_tf(
                        0.0, nsvahl * signvec(sqh, p[:, 3])
                    )  # <------ this enters the sign operation with y as a real vector
                    return tf.stack([v3, v4], axis=0)

                condition = tf.expand_dims(pt != 0, 0)
                v34 = tf.where(condition, pt_not_zero(), pt_zero())
                return tf.concat([v2, v34, v5], axis=0)  # [4,nevts] complex

            cond = tf.expand_dims(pp == 0, 0)
            return tf.where(cond, pp_zero(), pp_not_zero())

        def is_not_massive():
            pp = p[:, 0]
            pt = tfmath.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2)
            v2 = tf.ones([1, nevts], dtype=DTYPECOMPLEX) * complex_tf(0.0, 0.0)
            v5 = tf.expand_dims(complex_tf(nhel * pt / pp * sqh, 0.0), 0)

            def pt_not_zero():
                pzpt = p[:, 3] / (pp * pt) * sqh * nhel
                v3 = complex_tf(-p[:, 1] * pzpt, -nsv * p[:, 2] / pt * sqh)
                v4 = complex_tf(-p[:, 2] * pzpt, nsv * p[:, 1] / pt * sqh)
                return tf.stack([v3, v4], axis=0)

            def pt_zero():
                v3 = tf.ones(nevts, dtype=DTYPECOMPLEX) * complex_tf(-nhel * sqh, 0.0)
                v4 = complex_tf(
                    0.0, nsv * signvec(sqh, p[:, 3])
                )  # <------ this enters the sign operation with y as a real vector
                return tf.stack([v3, v4], axis=0)

            cond = tf.expand_dims(pt != 0, 0)
            v34 = tf.where(cond, pt_not_zero(), pt_zero())
            return tf.concat([v2, v34, v5], axis=0)

        massive = vmass != 0
        v = tf.where(massive, is_massive(), is_not_massive())
        return tf.concat([v0, v1, v], axis=0)

    BRST = nhel == 4
    return tf.where(BRST, is_BRST(), is_not_BRST())
