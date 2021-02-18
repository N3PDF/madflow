import tensorflow as tf
from vegasflow.configflow import DTYPE, DTYPEINT, int_me, float_me
from config import DTYPECOMPLEX
from config import complex_tf, complex_me

p_sgn = [tf.TensorSpec(shape=[None,4], dtype=DTYPE)]
float_sgn = [tf.TensorSpec(shape=[], dtype=DTYPE)]
int_sgn = [tf.TensorSpec(shape=[], dtype=DTYPEINT)]
wave_signature = p_sgn + float_sgn*3

@tf.function(input_signature=float_sgn*4 + int_sgn*2)
def ioxxxxx_rest(fmass, nhel, nsf, io, ip, im):
    """
    Defines rest massive branch for ingoing/outgoing fermion wavefunction.
    Outputs moving fermion spinor last four components.
    
    Parameters
    ----------
        fmass: tf.tensor of shape []
            particle mass
        nsf: tf.tensor of shape []
            +1 for particle, -1 for antiparticle
        nhel: tf.tensor of shape []
            particle helicity
        nsf: tf.tensor of shape []
            +1 for particle, -1 for antiparticle
        io: tf.tensor of shape []
            +1 for ingoing, -1 for outgoing particle
        ip: torch.tensor of shape [] of dtype DTYPEINT
            positive helicity projector
        im: torch.tensor of shape [] of dtype DTYPEINT
            negative helicity projector

    Returns
    -------
        tf.tensor of shape [4, None] of type DTYPECOMPLEX
    """    
    ipm = tf.stack([ip,im], axis=0)
    m = int_me([[(1+io)/2, -nhel*(1-io)/2],
                [nhel*(1-io)/2, (1+io)/2]])
    ii = tf.linalg.matvec(m, ipm)
    iip = ii[0]
    iim = ii[1]
    sqm = tf.math.sqrt(tf.math.abs(fmass))
    sqm = tf.stack([sqm, sqm*tf.math.sign(fmass)]) # [fmass, fmass] ---> TODO: why calling sign on the result of a sqrt ????
    v = [0]*4
    v[0] = complex_tf(float_me(iip)*sqm[iip],0.)
    v[1] = complex_tf(float_me(iim)*nsf*sqm[iip],0.)
    v[2] = complex_tf(float_me(iip)*nsf*sqm[iim],0.)
    v[3] = complex_tf(float_me(iim)*sqm[iim],0.)
    v = tf.reshape(tf.stack(v), [4,1])
    if io == 1:
        return v
    else:
        return tf.reverse(v,[0])

@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ] + float_sgn*4 + int_sgn*2)
def ioxxxxx_moving(p, pp, fmass, nsf, nh, io, ip, im):
    """
    Defines a moving massive branch for ingoing/outgoing fermion wavefunction.
    Outputs moving fermion spinor last four components.
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        pp: tf.tensor of shape [None]
            minimum between particle energy and three-momentum modulus
        fmass: tf.tensor of shape []
            particle mass
        nsf: tf.tensor of shape []
            +1 for particle, -1 for antiparticle
        nh: tf.tensor of shape []
            nhel*nsf product
        io: tf.tensor of shape []
            +1 for ingoing, -1 for outgoing particle
        ip: torch.tensor of shape [] of dtype DTYPEINT
            positive helicity projector
        im: torch.tensor of shape [] of dtype DTYPEINT
            negative helicity projector

    Returns
    -------
        tf.tensor of shape [4, None] of type DTYPECOMPLEX
    """    
    sf = tf.stack([(1+nsf+(1-nsf)*io*nh)*0.5,(1+nsf-(1-nsf)*io*nh)*0.5], axis=0)
    omega = tf.stack([tf.math.sqrt(p[:,0]+pp),fmass/(tf.math.sqrt(p[:,0]+pp))], axis=0)
    sfomeg = tf.stack([sf[0]*omega[ip],sf[1]*omega[im]], axis=0)
    if io == -1:
        sfomeg = tf.reverse(sfomeg,[0])
    pp3 = tf.math.maximum(pp+p[:,3],0.)
    chi1 = tf.where(pp3==0,
                    complex_tf(-nh,0),
                    complex_tf(nh*p[:,1]/tf.math.sqrt(2.*pp*pp3),
                               io*p[:,2]/tf.math.sqrt(2.*pp*pp3)))
    chi2 = complex_tf(tf.math.sqrt(pp3*0.5/pp),0.)
    chi = tf.stack([chi2, chi1], axis=0)
    v = [0]*4
    v[0] = complex_tf(sfomeg[0], 0.)*chi[im]
    v[1] = complex_tf(sfomeg[0], 0.)*chi[ip]
    v[2] = complex_tf(sfomeg[1], 0.)*chi[im]
    v[3] = complex_tf(sfomeg[1], 0.)*chi[ip]
    return tf.stack(v, axis=0)


@tf.function(input_signature=p_sgn + float_sgn*5)
def ioxxxxx_massive(p, fmass, nhel, nsf, nh, io):
    """
    Defines massive branch for ingoing/outgoing fermion wavefunction.
    Outputs fermion spinor last four components.
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        fmass: tf.tensor of shape []
            particle mass
        nhel: tf.tensor of shape []
            particle helicity
        nsf: tf.tensor of shape []
            +1 for particle, -1 for antiparticle
        nh: tf.tensor of shape []
            nhel*nsf product
        io: tf.tensor of shape []
            +1 for ingoing, -1 for outgoing particle

    Returns
    -------
        tf.tensor of shape [4, None] of type DTYPECOMPLEX
    """
    pp = tf.math.minimum(p[:,0], tf.math.sqrt(p[:,1]**2 + p[:,2]**2 + p[:,3]**2 ))
    ip = int_me((1+nh)/2)
    im = int_me((1-nh)/2)
    rest_args = [fmass, nhel, nsf, io, ip, im]
    mv_args = [p, pp, fmass, nsf, nh, io, ip, im]
    rest = tf.expand_dims(pp==0, 0)
    return tf.where(rest, ioxxxxx_rest(*rest_args),
                    ioxxxxx_moving(*mv_args))


@tf.function(input_signature=p_sgn + float_sgn*4)
def ioxxxxx_massless(p, nhel, nsf, nh, io):
    """
    Defines massless branch for ingoing/outgoing fermion wavefunction.
    Outputs fermion spinor last four components.
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        nhel: tf.tensor of shape []
            particle helicity
        nsf: tf.tensor of shape []
            +1 for particle, -1 for antiparticle
        nh: tf.tensor of shape []
            nhel*nsf product
        io: tf.tensor of shape []
            +1 for ingoing, -1 for outgoing particle

    Returns
    -------
        tf.tensor of shape [4, None] of type DTYPECOMPLEX
    """
    sqp0p3 = tf.math.sqrt(tf.math.maximum(p[:,0]+p[:,3],0.))*nsf
    chi1 = tf.where(
            sqp0p3==0,
            complex_tf(-nhel*tf.math.sqrt(2.*p[:,0]),0.),
            complex_tf(nh*p[:,1]/sqp0p3,p[:,2]/sqp0p3)
           )
    chi = tf.stack([complex_tf(sqp0p3,0.),chi1], axis=0)
    v = [tf.zeros_like(chi[0])]*4
    if nh*io == 1:
        v[2] = chi[0]
        v[3] = chi[1]
    else:
        v[0] = chi[1]
        v[1] = chi[0]
    return tf.stack(v, axis=0)


@tf.function(input_signature=float_sgn*3 + int_sgn)
def vxxxxx_rest(nhel, nsvahl, sqh, nevts):
    """
    Defines rest massive branch of a vector boson wavefunction.
    Last four components.
    
    Parameters
    ----------
        nhel: tf.tensor of shape []
            particle helicity
        nsvahl: tf.tensor of shape []
            nsv*abs(nhel) product
        sqh: tf.tensor of shape []
            square root of two over two
        nevts: tf.tensor of shape [] of dtype DTYPEINT
            number of events

    Returns
    -------
        tf.tensor of shape [4, None] of type DTYPECOMPLEX
    """
    v = [0]*4
    hel0 = 1.-tf.math.abs(nhel)
    v[0] = tf.ones(nevts, dtype=DTYPECOMPLEX)
    v[1] = tf.ones_like(v[0])*complex_tf(-nhel*sqh,0.)
    v[2] = tf.ones_like(v[0])*complex_tf(0.,nsvahl*sqh)
    v[3] = tf.ones_like(v[0])*complex_tf(hel0,0.)
    return tf.stack(v, axis=0)


@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ]*3 + float_sgn*4)
def vxxxxx_mv_nonzero_pt(p, pp, pt, emp, nhel, nsvahl, hel0, sqh):
    """
    Defines moving massive branch of a vector boson wavefunction.
    Last four components.
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        pp: tf.tensor of shape [None]
            minimum between particle energy and three-momentum modulus
        pt: tf.tensor of shape [None]
            particle transverse momenta
        nhel: tf.tensor of shape []
            particle helicity
        nsvahl: tf.tensor of shape []
            nsv*abs(nhel) product
        hel0: tf.tensor of shape []
            vector boson longitudinal polarization
        emp: tf.tensor of shape []
            gamma * pp
        sqh: tf.tensor of shape []
            square root of two over two
        nevts: tf.tensor of shape [] of dtype DTYPEINT
            number of events

    Returns
    -------
        tf.tensor of shape [4, None] of type DTYPECOMPLEX
    """
    v = [0]*2
    pzpt = p[:,3]/(pp*pt)*sqh*nhel
    v[0] = complex_tf(hel0*p[:,1]*emp-p[:,1]*pzpt, \
        -nsvahl*p[:,2]/pt*sqh)
    v[1] = complex_tf(hel0*p[:,2]*emp-p[:,2]*pzpt, \
        nsvahl*p[:,1]/pt*sqh) 
    return tf.stack(v, axis=0)


@tf.function(input_signature=p_sgn + float_sgn*3 + int_sgn)
def vxxxxx_mv_zero_pt(p, nhel, nsvahl, sqh, nevts):
    """
    Defines moving zero pt massive branch of a vector boson wavefunction.
    Last four components.
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        nhel: tf.tensor of shape []
            particle helicity
        nsvahl: tf.tensor of shape []
            nsv*abs(nhel) product
        sqh: tf.tensor of shape []
            square root of two over two
        nevts: tf.tensor of shape [] of dtype DTYPEINT
            number of events

    Returns
    -------
        tf.tensor of shape [4, None] of type DTYPECOMPLEX
    """
    v = [0]*2
    v[0] = tf.ones(nevts, dtype=DTYPECOMPLEX)*complex_tf(-nhel*sqh,0.)
    v[1] = complex_tf(0.,nsvahl*sqh*tf.math.sign(p[:,3]))
    return tf.stack(v, axis=0)


@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ]*2 + float_sgn*5 + int_sgn)
def vxxxxx_moving(p, pp, pt, vmass, nhel, nsvahl, hel0, sqh, nevts):
    """
    Defines moving massive branch of a vector boson wavefunction.
    Last four components.
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        pp: tf.tensor of shape [None]
            minimum between particle energy and three-momentum modulus
        vmass: tf.tensor of shape []
            particle mass
        nhel: tf.tensor of shape []
            particle helicity
        nsv: tf.tensor of shape []
            +1 for particle, -1 for antiparticle
        hel0: tf.tensor of shape []
            vector boson longitudinal polarization
        nsvahl: tf.tensor of shape []
            nsv*abs(nhel) product
        sqh: tf.tensor of shape []
            square root of two over two
        nevts: tf.tensor of shape [] of dtype DTYPEINT
            number of events

    Returns
    -------
        tf.tensor of shape [4, None] of type DTYPECOMPLEX
    """
    v = [0]*3
    emp = p[:,0]/(vmass*pp)
    v[0] = tf.expand_dims(complex_tf(hel0*pp/vmass,0.), 0)
    v[2] = tf.expand_dims(complex_tf(hel0*p[:,3]*emp+nhel*pt/pp*sqh, 0), 0)
    mv_nonzero_pt_args = [p, pp, pt, emp, nhel, nsvahl, hel0, sqh]
    mv_zero_pt_args = [p, nhel, nsvahl, sqh, nevts]
    condition = tf.expand_dims(pt!=0, 0)
    v[1] = tf.where(condition, vxxxxx_mv_nonzero_pt(*mv_nonzero_pt_args),
                   vxxxxx_mv_zero_pt(*mv_zero_pt_args))
    return tf.concat(v, axis=0)


@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ]*2 + float_sgn*3)
def vxxxxx_nonzero_pt(p, pp, pt, nhel, nsv, sqh):
    """
    Defines non-zero pt massless branch of a vector boson wavefunction.
    Two central components.
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        pp: tf.tensor of shape [None]
            vector boson energies
        pt: tf.tensor of shape [None]
            particle transverse momenta
        nhel: tf.tensor of shape []
            particle helicity
        nsv: tf.tensor of shape []
            +1 for particle, -1 for antiparticle
        sqh: tf.tensor of shape []
            square root of two over two
        nevts: tf.tensor of shape [] of dtype DTYPEINT
            number of events

    Returns
    -------
        tf.tensor of shape [2, None] of type DTYPECOMPLEX
    """
    v = [0]*2
    pzpt = p[:,3]/(pp*pt)*sqh*nhel
    v[0] = complex_tf(-p[:,1]*pzpt,-nsv*p[:,2]/pt*sqh)
    v[1] = complex_tf(-p[:,2]*pzpt,nsv*p[:,1]/pt*sqh)
    return tf.stack(v, axis=0)


@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ] + float_sgn*3 + int_sgn)
def vxxxxx_zero_pt(p, pp, nhel, nsv, sqh, nevts):
    """
    Defines zero pt massless branch of a vector boson wavefunction.
    Two central components.
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        pp: tf.tensor of shape [None]
            vector boson energies
        nhel: tf.tensor of shape []
            particle helicity
        nsv: tf.tensor of shape []
            +1 for particle, -1 for antiparticle
        sqh: tf.tensor of shape []
            square root of two over two
        nevts: tf.tensor of shape [] of dtype DTYPEINT
            number of events

    Returns
    -------
        tf.tensor of shape [2, None] of type DTYPECOMPLEX
    """
    v = [0]*2
    v[0] = tf.ones(nevts, dtype=DTYPECOMPLEX)*complex_tf(-nhel*sqh,0.)
    v[1] = complex_tf(0.,nsv*sqh*tf.math.sign(p[:,3]))
    return tf.stack(v, axis=0)


@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ]*2 + float_sgn*5 + int_sgn)
def vxxxxx_massive(p, pp, pt, vmass, nhel, hel0, nsvahl, sqh, nevts):
    """
    Defines massive branch of a vector boson wavefunction.
    Last four components.
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        pp: tf.tensor of shape [None]
            minimum between particle energy and three-momentum modulus
        vmass: tf.tensor of shape []
            particle mass
        nhel: tf.tensor of shape []
            particle helicity
        nsv: tf.tensor of shape []
            +1 for particle, -1 for antiparticle
        hel0: tf.tensor of shape []
            vector boson longitudinal polarization
        nsvahl: tf.tensor of shape []
            nsv*abs(nhel) product
        sqh: tf.tensor of shape []
            square root of two over two
        nevts: tf.tensor of shape [] of dtype DTYPEINT
            number of events

    Returns
    -------
        tf.tensor of shape [4, None] of type DTYPECOMPLEX
    """
    rest_args = [nhel, nsvahl, sqh, nevts]
    mv_args = [p, pp, pt, vmass, nhel, nsvahl, hel0, sqh, nevts]
    cond = tf.expand_dims(pp==0, 0)
    return tf.where(cond, vxxxxx_rest(*rest_args),
                    vxxxxx_moving(*mv_args))


@tf.function(input_signature=p_sgn + float_sgn*3 + int_sgn)
def vxxxxx_massless(p, nhel, nsv, sqh, nevts):
    """
    Defines massless branch of a vector boson wavefunction.
    Last four components.
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        nhel: tf.tensor of shape []
            particle helicity
        nsv: tf.tensor of shape []
            +1 for particle, -1 for antiparticle
        sqh: tf.tensor of shape []
            square root of two over two
        nevts: tf.tensor of shape [] of dtype DTYPEINT
            number of events

    Returns
    -------
        tf.tensor of shape [4, None] of type DTYPECOMPLEX
    """
    v = [0]*3
    pp = p[:,0]
    pt = tf.math.sqrt(p[:,1]**2 + p[:,2]**2)
    v[0] = tf.ones([1,nevts], dtype=DTYPECOMPLEX)*complex_tf(0.,0.)
    v[2] = tf.expand_dims(complex_tf(nhel*pt/pp*sqh, 0.), 0)
    nonzero_pt_args = [p, pp, pt, nhel, nsv, sqh]
    zero_pt_args = [p, pp, nhel, nsv, sqh, nevts]
    cond = tf.expand_dims(pt!=0, 0)
    v[1] = tf.where(cond, vxxxxx_nonzero_pt(*nonzero_pt_args),
                    vxxxxx_zero_pt(*zero_pt_args))
    return tf.concat(v, axis=0)


@tf.function(input_signature=wave_signature)
def ixxxxx(p, fmass, nhel, nsf):
    """
    Defines an ingoing fermion wavefunction
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        fmass: tf.tensor of shape []
            particle mass
        nhel: tf.tensor of shape []
            particle helicity
        nsf: tf.tensor of shape []
            +1 for particle, -1 for antiparticle

    Returns
    -------
        tf.tensor of shape [6, None] of type DTYPECOMPLEX
    """
    # print("ixxxxx")
    nevts = tf.shape(p, out_type=DTYPEINT)[0]
    v = [0.]*3
    v[0] = tf.expand_dims(complex_tf(-p[:,0]*nsf,-p[:,3]*nsf), 0)
    v[1] = tf.expand_dims(complex_tf(-p[:,1]*nsf,-p[:,2]*nsf), 0)
    nh = nhel*nsf
    massive_args = [p, fmass, nhel, nsf, nh, float_me(1)]
    massless_args = [p, nhel, nsf, nh, float_me(1)]
    massive = fmass != 0
    v[2] = tf.cond(massive,
                   lambda: ioxxxxx_massive(*massive_args),
                   lambda: ioxxxxx_massless(*massless_args))
    return tf.concat(v, axis=0)


@tf.function(input_signature=wave_signature)
def oxxxxx(p, fmass, nhel, nsf):
    """
    Defines an outgoing fermion wavefunction
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        fmass: tf.tensor of shape []
            particle mass
        nhel: tf.tensor of shape []
            particle helicity
        nsf: tf.tensor of shape []
            +1 for particle, -1 for antiparticle

    Returns
    -------
        tf.tensor of shape [6, None] of type DTYPECOMPLEX
    """
    # print("oxxxxx")
    nevts = tf.shape(p, out_type=DTYPEINT)[0]
    v = [0.]*3
    v[0] = tf.expand_dims(complex_tf(p[:,0]*nsf,p[:,3]*nsf), 0)
    v[1] = tf.expand_dims(complex_tf(p[:,1]*nsf,p[:,2]*nsf), 0)
    nh = nhel*nsf
    massive_args = [p, fmass, nhel, nsf, nh, float_me(-1)]
    massless_args = [p, nhel, nsf, nh, float_me(-1)]
    massive = fmass != 0
    v[2] = tf.cond(massive,
                   lambda: ioxxxxx_massive(*massive_args),
                   lambda: ioxxxxx_massless(*massless_args))
    return tf.concat(v, axis=0)


@tf.function(input_signature=wave_signature)
def vxxxxx(p, vmass, nhel, nsv):
    """
    Defines a vector boson wavefunction
    
    Parameters
    ----------
        p: tf.tensor of shape [None, 4]
            tensor of momenta
        vmass: tf.tensor of shape []
            particle mass
        nhel: tf.tensor of shape []
            particle helicity
        nsv: tf.tensor of shape []
            +1 for particle, -1 for antiparticle

    Returns
    -------
        tf.tensor of shape [6, None] of type DTYPECOMPLEX
    """
    # print("vxxxxx")
    nevts = tf.shape(p, out_type=DTYPEINT)[0]
    hel0 = 1.-tf.math.abs(nhel)
    sqh = float_me(tf.math.sqrt(0.5))
    nsvahl = nsv*tf.math.abs(nhel)
    pt2 = p[:,1]**2 + p[:,2]**2 
    pp = tf.math.minimum(p[:,0],tf.math.sqrt(pt2 + p[:,3]**2))
    pt = tf.math.minimum(pp,tf.math.sqrt(pt2))
    v = [0]*3
    v[0] = tf.expand_dims(complex_tf(p[:,0]*nsv,p[:,3]*nsv), 0)
    v[1] = tf.expand_dims(complex_tf(p[:,1]*nsv,p[:,2]*nsv), 0)
    if nhel == 4:
        massless = vmass == 0
        v[2] = tf.cond(massless,
                       lambda: complex_me( tf.transpose(p/p[:,0]) ),
                       lambda: complex_me( tf.transpose(p/vmass) ))        
    else:
        massive_args = [p, pp, pt, vmass, nhel, hel0, nsvahl, sqh, nevts]
        massless_args = [p, nhel, nsv, sqh, nevts]            
        massive = vmass != 0
        v[2] =  tf.cond(massive,
                        lambda: vxxxxx_massive(*massive_args),
                        lambda: vxxxxx_massless(*massless_args))
    return tf.concat(v, axis=0)
