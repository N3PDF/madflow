import tensorflow as tf
from tensorflow.math import sqrt, abs, minimum, maximum, sign
from vegasflow.configflow import DTYPE, DTYPEINT, int_me, float_me
from config import DTYPECOMPLEX
from config import complex_tf, complex_me

p_sgn = [tf.TensorSpec(shape=[None,4], dtype=DTYPE)]
float_sgn = [tf.TensorSpec(shape=[], dtype=DTYPE)]
int_sgn = [tf.TensorSpec(shape=[], dtype=DTYPEINT)]
wave_signature = p_sgn + float_sgn*3

@tf.function(input_signature=float_sgn*4 + int_sgn*2)
def ioxxxxx_rest(fmass, nhel, nsf, io, ip, im):
    ipm = tf.stack([ip,im], axis=0)
    m = int_me([[(1+io)/2, -nhel*(1-io)/2],
                [nhel*(1-io)/2, (1+io)/2]])
    ii = tf.linalg.matvec(m, ipm)
    iip = ii[0]
    iim = ii[1]
    sqm = sqrt(abs(fmass))
    sqm = tf.stack([sqm, sqm*sign(fmass)]) # [fmass, fmass] ---> TODO: why calling sign on the result of a sqrt ????
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
    sf = tf.stack([(1+nsf+(1-nsf)*io*nh)*0.5,(1+nsf-(1-nsf)*io*nh)*0.5], axis=0)
    omega = tf.stack([sqrt(p[:,0]+pp),fmass/(sqrt(p[:,0]+pp))], axis=0)
    sfomeg = tf.stack([sf[0]*omega[ip],sf[1]*omega[im]], axis=0)
    pp3 = maximum(pp+p[:,3],0.)
    chi1 = tf.where(pp3==0, complex_tf(-nh,0), complex_tf(nh*p[:,1]/sqrt(2.*pp*pp3), p[:,2]/sqrt(2.*pp*pp3))) # [nevt,] complex
    chi2 = complex_tf(sqrt(pp3*0.5/pp),0.)
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
    Parameters
    ----------
        io: tf.tensor
            dtype DTYPE, +1 for ingoing waveform, -1 for outgoing waveform
    """
    pp = minimum(p[:,0], sqrt(p[:,1]**2 + p[:,2]**2 + p[:,3]**2 ))
    ip = int_me((1+nh)/2)
    im = int_me((1-nh)/2)
    rest_args = [fmass, nhel, nsf, io, ip, im]
    mv_args = [p, pp, fmass, nsf, nh, io, ip, im]
    rest = tf.expand_dims(pp==0, 0)
    return tf.where(rest, ioxxxxx_rest(*rest_args),
                    ioxxxxx_moving(*mv_args))


@tf.function(input_signature=p_sgn + float_sgn*4)
def ioxxxxx_massless(p, nsf, nh, nhel, io):
    sqp0p3 = sqrt(maximum(p[:,0]+p[:,3],0.))*nsf
    chi1 = tf.where(
            sqp0p3==0,
            complex_tf(-nhel*sqrt(2.*p[:,0]),0.),
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


@tf.function(input_signature=wave_signature)
def ixxxxx(p,fmass,nhel,nsf):
    """Defines an ingoing fermion """
    # print("ixxxxx")
    nevts = tf.shape(p, out_type=DTYPEINT)[0]
    v = [0.]*3
    v[0] = tf.expand_dims(complex_tf(-p[:,0]*nsf,-p[:,3]*nsf), 0)
    v[1] = tf.expand_dims(complex_tf(-p[:,1]*nsf,-p[:,2]*nsf), 0)
    nh = nhel*nsf # either +1 or -1
    true_args = [p, fmass, nhel, nsf, nh, float_me(1)]
    false_args = [p, nsf, nh, nhel, float_me(1)]
    massive = fmass != 0
    v[2] = tf.cond(massive,
                   lambda: ioxxxxx_massive(*true_args),
                   lambda: ioxxxxx_massless(*false_args))
    return tf.concat(v, axis=0)


@tf.function(input_signature=wave_signature)
def oxxxxx(p,fmass,nhel,nsf):
    """ Defines an outgoing fermion """
    # print("oxxxxx")
    nevts = tf.shape(p, out_type=DTYPEINT)[0]
    v = [0.]*3
    v[0] = tf.expand_dims(complex_tf(p[:,0]*nsf,p[:,3]*nsf), 0)
    v[1] = tf.expand_dims(complex_tf(p[:,1]*nsf,p[:,2]*nsf), 0)
    nh = nhel*nsf # either +1 or -1
    true_args = [p, fmass, nhel, nsf, nh, float_me(-1)]
    false_args = [p, nsf, nh, nhel, float_me(-1)]
    massive = fmass != 0
    v[2] = tf.cond(massive,
                   lambda: ioxxxxx_massive(*true_args),
                   lambda: ioxxxxx_massless(*false_args))
    return tf.concat(v, axis=0)


@tf.function(input_signature=p_sgn + int_sgn)
def vxxxxx_brst_massless(p, nevts):
    vc = [0]*4
    vc[0] = tf.ones(nevts, dtype=DTYPE)
    vc[1]= p[:,1]/p[:,0]
    vc[2]= p[:,2]/p[:,0]
    vc[3]= p[:,3]/p[:,0]
    return complex_me(tf.stack(vc, axis=0))


@tf.function(input_signature=p_sgn + float_sgn + int_sgn)
def vxxxxx_brst_massive(p, vmass, nevts):
    vc = [0]*4
    vc[0] = p[:,0]/vmass
    vc[1] = p[:,1]/vmass
    vc[2] = p[:,2]/vmass
    vc[3] = p[:,3]/vmass
    return complex_me(tf.stack(vc, axis=0))

@tf.function(input_signature=float_sgn*3 + int_sgn)
def vxxxxx_rest(nhel, sqh, nsvahl, nevts):
    v = [0]*4
    hel0 = 1.-abs(nhel)
    v[0] = tf.ones(nevts, dtype=DTYPECOMPLEX)
    v[1] = tf.ones_like(v[0])*complex_tf(-nhel*sqh,0.)
    v[2] = tf.ones_like(v[0])*complex_tf(0.,nsvahl*sqh)
    v[3] = tf.ones_like(v[0])*complex_tf(hel0,0.)
    return tf.stack(v, axis=0) # [4,nevts] complex


@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ]*3 + float_sgn*4)
def vxxxxx_mv_nonzero_pt(p, pp, pt, emp, hel0, nsvahl, nhel, sqh):
    v = [0]*2
    pzpt = p[:,3]/(pp*pt)*sqh*nhel
    v[0] = complex_tf(hel0*p[:,1]*emp-p[:,1]*pzpt, \
        -nsvahl*p[:,2]/pt*sqh)
    v[1] = complex_tf(hel0*p[:,2]*emp-p[:,2]*pzpt, \
        nsvahl*p[:,1]/pt*sqh) 
    return tf.stack(v, axis=0)


@tf.function(input_signature=p_sgn + float_sgn*3 + int_sgn)
def vxxxxx_mv_zero_pt(p, nsvahl, nhel, sqh, nevts):
    v = [0]*2
    v[0] = tf.ones(nevts, dtype=DTYPECOMPLEX)*complex_tf(-nhel*sqh,0.)
    v[1] = complex_tf(0.,nsvahl*sqh*sign(p[:,3]))
    return tf.stack(v, axis=0)


@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ]*2 + float_sgn*5 + int_sgn)
def vxxxxx_moving(p, pp, pt, vmass, hel0, nsvahl, nhel, sqh, nevts):
    v = [0]*3
    emp = p[:,0]/(vmass*pp)
    v[0] = tf.expand_dims(complex_tf(hel0*pp/vmass,0.), 0)
    v[2] = tf.expand_dims(complex_tf(hel0*p[:,3]*emp+nhel*pt/pp*sqh, 0), 0)
    mv_nonzero_pt_args = [p, pp, pt, emp, hel0, nsvahl, nhel, sqh]
    mv_zero_pt_args = [p, nsvahl, nhel, sqh, nevts]
    condition = tf.expand_dims(pt!=0, 0)
    v[1] = tf.where(condition, vxxxxx_mv_nonzero_pt(*mv_nonzero_pt_args),
                   vxxxxx_mv_zero_pt(*mv_zero_pt_args))
    return tf.concat(v, axis=0)


@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ]*2 + float_sgn*3)
def vxxxxx_nonzero_pt(p, pp, pt, nhel, sqh, nsv):
    v = [0]*2
    pzpt = p[:,3]/(pp*pt)*sqh*nhel
    v[0] = complex_tf(-p[:,1]*pzpt,-nsv*p[:,2]/pt*sqh)
    v[1] = complex_tf(-p[:,2]*pzpt,nsv*p[:,1]/pt*sqh)
    return tf.stack(v, axis=0)


@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ] + float_sgn*3 + int_sgn)
def vxxxxx_zero_pt(p, pp, nhel, sqh, nsv, nevts):
    v = [0]*2
    v[0] = tf.ones(nevts, dtype=DTYPECOMPLEX)*complex_tf(-nhel*sqh,0.)
    v[1] = complex_tf(0.,nsv*sqh*sign(p[:,3]))
    return tf.stack(v, axis=0)


@tf.function(input_signature=p_sgn + [
    tf.TensorSpec(shape=[None], dtype=DTYPE)
    ]*2 + float_sgn*5 + int_sgn)
def vxxxxx_massive(p, pp, pt, vmass, hel0, nsvahl, nhel, sqh, nevts):
    rest_args = [nhel, sqh, nsvahl, nevts]
    mv_args = [p, pp, pt, vmass, hel0, nsvahl, nhel, sqh, nevts]
    cond = tf.expand_dims(pp==0, 0)
    return tf.where(cond, vxxxxx_rest(*rest_args),
                    vxxxxx_moving(*mv_args))


@tf.function(input_signature=p_sgn + float_sgn*3 + int_sgn)
def vxxxxx_massless(p, nhel, nsv, sqh, nevts):
    v = [0]*3
    pp = p[:,0]
    pt = sqrt(p[:,1]**2 + p[:,2]**2)
    v[0] = tf.ones([1,nevts], dtype=DTYPECOMPLEX)*complex_tf(0.,0.)
    v[2] = tf.expand_dims(complex_tf(nhel*pt/pp*sqh, 0.), 0)
    nonzero_pt_args = [p, pp, pt, nhel, sqh, nsv]
    zero_pt_args = [p, pp, nhel, sqh, nsv, nevts]
    cond = tf.expand_dims(pt!=0, 0)
    v[1] = tf.where(cond, vxxxxx_nonzero_pt(*nonzero_pt_args),
                    vxxxxx_zero_pt(*zero_pt_args))
    return tf.concat(v, axis=0)


@tf.function(input_signature=wave_signature)
def vxxxxx(p, vmass, nhel, nsv):
    """ Defines a vector wavefunction """
    # print("vxxxxx")
    nevts = tf.shape(p, out_type=DTYPEINT)[0]
    hel0 = 1.-abs(nhel)
    sqh = float_me(sqrt(0.5))
    nsvahl = nsv*abs(nhel)
    pt2 = p[:,1]**2 + p[:,2]**2 
    pp = minimum(p[:,0],sqrt(pt2 + p[:,3]**2))
    pt = minimum(pp,sqrt(pt2))
    v = [0]*3
    v[0] = tf.expand_dims(complex_tf(p[:,0]*nsv,p[:,3]*nsv), 0)
    v[1] = tf.expand_dims(complex_tf(p[:,1]*nsv,p[:,2]*nsv), 0)
    if nhel == 4:
        """ BRST checking """
        brst_massless_args = [p, nevts]
        brst_massive_args = [p, vmass, nevts]
        massless = vmass == 0
        v[2] = tf.cond(massless,
                       lambda: vxxxxx_brst_massless(*brst_massless_args),
                       lambda: vxxxxx_brst_massive(*brst_massive_args))        
    else:
        massive_args = [p, pp, pt, vmass, hel0, nsvahl, nhel, sqh, nevts]
        massless_args = [p, nhel, nsv, sqh, nevts]            
        massive = vmass != 0
        v[2] =  tf.cond(massive,
                        lambda: vxxxxx_massive(*massive_args),
                        lambda: vxxxxx_massless(*massless_args))
    return tf.concat(v, axis=0)
