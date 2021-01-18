"""
    Extension of rambo to generate many N-particle phase space in one go.

    This is the first non-fortran version where the source code is not shouting at you at any point
"""
import os
from vegasflow import float_me, run_eager
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

PI = float_me(np.pi)

def gen_unconstrained_momenta(xrand):
    """
        Generates unconstrained 4-momenta
    """
    costh = 2.*xrand[:, 0] - 1.
    sinth = tf.sqrt(1. - costh**2)
    phi = 2*PI*xrand[:, 1]
    energy = -tf.math.log(tf.reduce_prod(xrand[:, 2:4], axis=1))

    qx = energy*sinth*tf.math.sin(phi)
    qy = energy*sinth*tf.math.cos(phi)
    qz = energy*costh
    return tf.stack([qx, qy, qz, energy], axis=1)

def mul(a,b):
    """ Multiply tensors a and b where rank(a) = 2, rank(b) = 1
    and the first dimension has a size equal to the only dimension of b
    """
    return a*tf.reshape(b, (-1,1))
    
def parallel_rambo(xrand, n, sqrts, masses=None):
    """********************************************************************
    *                       RAMBO                                         *
    *    RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)                  *
    *                                                                     *
    *    a democratic multi-particle phase space generator                *
    *    authors:  S.D. Ellis,  R. Kleiss,  W.J. Stirling                 *
    *    -- adjusted by Hans Kuijf, weights are logarithmic (20-08-90)    *
    *    this is py version 1.0 -  written by O. Mattelaer                *
    *    this is parallel version 1.0 -  written by J. Cruz-Martinez      *
    *                                                                     *
    *    xrand = array of (array of) random numbers                       *
    *    n  = number of particles                                         *
    *    sqrts = total centre-of-mass energy                              *
    *    masses = particle masses                                         *
    *  return                                                             *
    *    p  = array of particle momenta ( dim=(4,nexternal-NINCOMING) )   *
    *    wt = array of weight of the events                               *
    ***********************************************************************"""
    if masses is not None:
        raise ValueError("The developer is too lazy to add massive particle just now")

    # Generate the unconstrained momenta Q
    all_q = []
    for i in range(n):
        q = gen_unconstrained_momenta(xrand[:, i*4:(i+1)*4])
        all_q.append(q)
    all_q = tf.stack(all_q) # (nparticles, nevents, 4:{x,y,z,E})

    # Compute the parameters for the conformal transformation
    sum_q = tf.reduce_sum(all_q, axis=0) # (nevents, 4)
    sum_q2 = sum_q**2
    qmass = tf.sqrt(sum_q2[:,-1] - tf.reduce_sum(sum_q2[:,:-1], axis=1))
    x = (sqrts/qmass) #.reshape(-1,1)
    bquad = mul(-sum_q, 1.0/qmass)
    bvec = bquad[:,:-1]
    gamma = -bquad[:,-1]
    a = 1.0/(1.0+gamma)

    # Perform the conformal transformation q->p
    all_p = []
    for q in all_q:
        bq = tf.reduce_sum(q[:,:-1]*bvec, axis=1)
        tmp = bq*a + q[:,-1]
        pvec = mul( q[:,:-1] + mul(bvec,tmp), x)
        pnrg = tf.reshape( ( q[:,-1]*gamma + bq ) * x, (-1,1))
        # Note that the Madgraph ME requires the momenta to be (E, px, py, pz)
        p = tf.concat([pnrg, pvec], axis=1)
        all_p.append(p)
    all_p = tf.stack(all_p, axis=1) # (nevents, nparticles, 4)

    # Compute the weight of the phase space point
    # (for n > 2)
    # can be done a la numpy since it is a constant
    wt = np.log(np.pi/2)
    for i in range(2, n):
        wt += np.log(np.pi/2) - 2.*np.log(i-1) - np.log(i)
    wt += (2*n-4)*np.log(sqrts)

    return all_p, float_me(wt)


if __name__ == "__main__":
    import random
    n = 4
    seed = 4
    sqrts = 7e3
    random.seed(seed)
    ndim = n*4
    all_rand = []
    for _ in range(2):
        all_rand.append([random.uniform(0,1) for _ in range(ndim)])
    xrand = float_me(all_rand)
    p, wt = parallel_rambo(xrand, n, sqrts)

    # Add here the madgraph rambo module for comparison
    mad_rambo = "../../mg5amcnlo/madgraph/various/rambo.py"
    import importlib.util
    rambo_spec = importlib.util.spec_from_file_location("rambo", mad_rambo)
    rambo_modu = importlib.util.module_from_spec(rambo_spec)
    rambo_spec.loader.exec_module(rambo_modu)
    rambo = getattr(rambo_modu, "RAMBO")
    flist = getattr(rambo_modu, "FortranList")
    #####
    random.seed(seed)
    p1r, wt1 = rambo(n, sqrts, flist(4))
    p1 = np.array(p1r).T
    p2r, wt2 = rambo(n, sqrts, flist(4))
    p2 = np.array(p2r).T

    print("Checking first ps point...")
    np.testing.assert_allclose(p[0], p1, rtol=1e-4)
    print("Checking second ps point...")
    np.testing.assert_allclose(p[1], p2, rtol=1e-4)
    print("Checking weight...")
    assert wt1==wt2==wt
    print("All good!")
