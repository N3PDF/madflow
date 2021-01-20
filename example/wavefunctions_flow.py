import tensorflow as tf
from vegasflow.configflow import DTYPE, DTYPEINT, int_me, float_me
from config import DTYPECOMPLEX
from config import complex_tf, complex_me

keys = tf.constant([0,1,2,3,4,5], dtype=DTYPEINT)
vals = tf.constant([1,3,6,6,18,18], dtype=DTYPEINT)
init = tf.lookup.KeyValueTensorInitializer(keys, vals)
spin_to_size = tf.lookup.StaticHashTable(init, default_value=-1)

@tf.function
def WaveFunctionFlow(npoints, spin):
    print("Retracing")
    size = spin_to_size.lookup(spin)
    shape = tf.stack([npoints, size], axis=0)
    return tf.zeros(shape, dtype=DTYPE)


def sign(x,y):
    """Fortran's sign transfer function"""
    tmp = tf.constant(0. ,dtype=tf.float64)
    if y.dtype == DTYPECOMPLEX:
        if tf.math.abs(tf.math.imag(y)) < 1e-6 * tf.math.abs(tf.math.real(y)):
            tmp = tf.math.real(y)
        else:
            pass
            # TODO: raise here some error (not sure how it can be compatible with tf.function)
    else:
        tmp = tf.cast(y, dtype=DTYPE)
    if (tmp < 0.):
        return -tf.math.abs(x) 
    else:
        return tf.math.abs(x) 

def ixxxxx(p,fmass,nhel,nsf):
    """Defines an inflow fermion."""
    # TODO: make masks to filter the ps points correctly
    # only pp==0, pp3==0 and sqpp0p3 conditions must be changed 
    # an approach like filtering the points and updating a final vector should
    # be the best possible ! 
    # We must take into account the fact that the mask could possibly have zero entries
    # look at how pdfflow is implemented
    # Note: here p[:,i] selects the momentum dimension and is a [nevt,] tensor
    nevts = tf.shape(p, out_type=DTYPEINT)[0]    
    # fi = WaveFunctionFlow(nevts, 2) # output of shape [nevt, 6]    
    v0 = tf.expand_dims(complex_tf(-p[:,0]*nsf,-p[:,3]*nsf), 1) # [nevt,] complex
    v1 = tf.expand_dims(complex_tf(-p[:,1]*nsf,-p[:,2]*nsf), 1) # [nevt,] complex
    nh = nhel*nsf # either +1 or -1
    ip = (1+nh)//2
    im = (1-nh)//2
    if tf.constant(fmass != 0.):
        pp = tf.math.minimum(p[:,0], tf.math.sqrt(p[:,1]**2 + p[:,2]**2 + p[:,3]**2 )) # [nevt,]
        def true_fn():
            sqm = tf.math.sqrt(tf.math.abs(fmass))
            sqm = tf.stack([sqm, sign(sqm,fmass)]) # [fmass, fmass] ---> why calling sign on the result of a sqrt ????
            v2 = complex_tf(ip*sqm[int_me(ip)],0.) # just a complex number
            v3 = complex_tf(im*nsf*sqm[int_me(ip)],0.)
            v4 = complex_tf(ip*nsf*sqm[int_me(im)],0.)
            v5 = complex_tf(im*sqm[int_me(im)],0.)
            v = tf.stack([v2,v3,v4,v5]) # [4,] complex
            return tf.reshape(v, [1,4])
        def false_fn():
            sf = tf.concat([(1+nsf+(1-nsf)*nh)*0.5,(1+nsf-(1-nsf)*nh)*0.5], axis=0) # [2,]
            omega = tf.stack([tf.math.sqrt(p[:,0]+pp),fmass/(tf.math.sqrt(p[:,0]+pp))], axis=0) # [2, nevt]
            sfomeg = tf.stack([sf[0]*omega[int_me(ip)],sf[1]*omega[int_me(im)]], axis=0) # [2,nevt]
            pp3 = tf.math.maximum(pp+p[:,3],0.) # [nevt,]
            chi1 = tf.where(pp3==0, complex_tf(-nh,0), complex_tf(nh*p[:,1], p[:,2]/tf.math.sqrt(2.*pp*pp3))) # [nevt,] complex
            chi2 = tf.complex(tf.math.sqrt(pp3*0.5/pp),float_me(0.)) # [nevt,] complex 
            chi = tf.stack([chi2, chi1], axis=0) # [2, nevt] complex
            v2 = complex_tf(sfomeg[0], 0.)*chi[int_me(im)] # [nevt,] complex
            v3 = complex_tf(sfomeg[0], 0.)*chi[int_me(ip)]
            v4 = complex_tf(sfomeg[1], 0.)*chi[int_me(im)]
            v5 = complex_tf(sfomeg[1], 0.)*chi[int_me(ip)]
            return tf.stack([v2,v3,v4,v5], axis=1) # [nevt, 4] complex
        cond = tf.expand_dims(pp==0, 1)
        v = tf.where(cond, true_fn(), false_fn()) # [nevt, 4] complex
    else: 
        sqp0p3 = sqrt(max(p[:,0]+p[:,3],0.))*nsf # [nevt,]
        def true_fn():
            return complex_tf(-nhel*tf.math.sqrt(2.*p[:,0]),0.) # [nevt,] complex
        def false_fn():
            return complex_fn(nh*p[:,1]/sqp0p3,p[:,2]/sqp0p3) # [nevt,] complex
        chi1 = tf.where(sqp0p3, true_fn(), false_fn())
        chi = tf.concat([complex_fn(sqp0p3,0.),chi1], axis=0) # [2, nevt]
        def true_fn():
            v4 = chi[0] # [nevt,] complex
            v5 = chi[1] # [nevt,] complex
            v2 = tf.ones_like(v4)*complex_fn(0.,0.) # [nevt,] complex
            v3 = tf.ones_like(v4)*complex_fn(0.,0.) # [nevt,] complex
            return tf.stack([v2,v3,v4,v5], axis=1)
        def false_fn():
            v2 = chi[1]
            v3 = chi[0]
            v4 = tf.ones_like(v2)*complex_fn(0.,0.)
            v5 = tf.ones_like(v2)*complex_fn(0.,0.)
            return tf.stack([v2,v3,v4,v5], axis=1)
        v = tf.where(nh==1, true_fn(), false_fn())
    fi = tf.concat([v0,v1,v], axis=1)
    return fi


def oxxxxx(all_p,all_fmass,all_nhel,all_nsf):
    """ initialize an outgoing fermion"""
    # TODO: make masks to filter the ps points correctly
    # only pp==0, pp3==0 and sqpp0p3 conditions must be changed
    nevts = tf.shape(all_p, out_type=DTYPEINT)[0]
    fo = WaveFunction(2)
    fo[0] = complex(p[0]*nsf,p[3]*nsf)
    fo[1] = complex(p[1]*nsf,p[2]*nsf)
    nh = nhel*nsf
    if (fmass != 0.):
        pp = min(p[0],sqrt(p[1]**2 + p[2]**2 + p[3]**2 ))
        if (pp == 0.): 
            sqm = [sqrt(abs(fmass))]
            sqm.append( sign(sqm[0], fmass)) 
            ip = int(-((1-nh)//2) * nhel)
            im = int((1+nh)//2 * nhel)
            
            fo[2] = im*sqm[abs(im)]
            fo[3] = ip*nsf*sqm[abs(im)]
            fo[4] = im*nsf*sqm[abs(ip)]
            fo[5] = ip*sqm[abs(ip)]

        else:
            sf = [(1+nsf+(1-nsf)*nh)*0.5,(1+nsf-(1-nsf)*nh)*0.5]
            omega = [sqrt(p[0]+pp),fmass/(sqrt(p[0]+pp))]
            ip = (1+nh)//2
            im = (1-nh)//2
            sfomeg = [sf[0]*omega[ip],sf[1]*omega[im]]
            pp3 = max(pp+p[3],0.)
            if (pp3 == 0.):
                chi1 = complex(-nh,0.) 
            else:
                chi1 = complex(nh*p[1]/sqrt(2.*pp*pp3),\
                -p[2]/sqrt(2.*pp*pp3))
            chi = [complex(sqrt(pp3*0.5/pp)),chi1]

            fo[2] = sfomeg[1]*chi[im]
            fo[3] = sfomeg[1]*chi[ip]
            fo[4] = sfomeg[0]*chi[im]
            fo[5] = sfomeg[0]*chi[ip] 
            
    else: 
        sqp0p3 = sqrt(max(p[0]+p[3],0.))*nsf
        if (sqp0p3 == 0.):
            chi1 = complex(-nhel*sqrt(2.*p[0]),0.)
        else:
            chi1 = complex(nh*p[1]/sqp0p3,-p[2]/sqp0p3)
        chi = [complex(sqp0p3,0.),chi1]
        if (nh == 1):
            fo[2] = chi[0]
            fo[3] = chi[1]
            fo[4] = complex(0.,0.)
            fo[5] = complex(0.,0.) 
        else:
            fo[2] = complex(0.,0.)
            fo[3] = complex(0.,0.)
            fo[4] = chi[1]
            fo[5] = chi[0] 
    
    return fo


def vxxxxx(all_p,all_fmass,all_nhel,all_nsf):
    """ initialize a vector wavefunction. nhel=4 is for checking BRST"""
    # TODO: change the following conditions: 
    # pp==0, pt!=0 count
    nevts = tf.shape(all_p, out_type=DTYPEINT)[0]
    vc = WaveFunction(3)
    
    sqh = sqrt(0.5)
    nsvahl = nsv*abs(nhel)
    pt2 = p[1]**2 + p[2]**2 
    pp = min(p[0],sqrt(pt2 + p[3]**2))
    pt = min(pp,sqrt(pt2))

    vc[0] = complex(p[0]*nsv,p[3]*nsv)
    vc[1] = complex(p[1]*nsv,p[2]*nsv)

    if (nhel == 4):
        if (vmass == 0.):
            vc[2] = 1.
            vc[3]=p[1]/p[0]
            vc[4]=p[2]/p[0]
            vc[5]=p[3]/p[0]
        else:
            vc[2] = p[0]/vmass
            vc[3] = p[1]/vmass
            vc[4] = p[2]/vmass
            vc[5] = p[3]/vmass
        
        return vc 

    if (vmass != 0.):
        hel0 = 1.-abs(nhel) 

        if (pp == 0.):
            vc[2] = complex(0.,0.)
            vc[3] = complex(-nhel*sqh,0.)
            vc[4] = complex(0.,nsvahl*sqh) 
            vc[5] = complex(hel0,0.)

        else:
            emp = p[0]/(vmass*pp)
            vc[2] = complex(hel0*pp/vmass,0.)
            vc[5] = complex(hel0*p[3]*emp+nhel*pt/pp*sqh)
            if (pt != 0.):
                pzpt = p[3]/(pp*pt)*sqh*nhel
                vc[3] = complex(hel0*p[1]*emp-p[1]*pzpt, \
                    -nsvahl*p[2]/pt*sqh)
                vc[4] = complex(hel0*p[2]*emp-p[2]*pzpt, \
                    nsvahl*p[1]/pt*sqh) 
            else:
                vc[3] = complex(-nhel*sqh,0.)
                vc[4] = complex(0.,nsvahl*sign(sqh,p[3]))
    else: 
        pp = p[0]
        pt = sqrt(p[1]**2 + p[2]**2)
        vc[2] = complex(0.,0.)
        vc[5] = complex(nhel*pt/pp*sqh)
        if (pt != 0.):
            pzpt = p[3]/(pp*pt)*sqh*nhel
            vc[3] = complex(-p[1]*pzpt,-nsv*p[2]/pt*sqh)
            vc[4] = complex(-p[2]*pzpt,nsv*p[1]/pt*sqh)
        else:
            vc[3] = complex(-nhel*sqh,0.)
            vc[4] = complex(0.,nsv*sign(sqh,p[3]))
    
    return vc