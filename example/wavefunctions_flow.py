import tensorflow as tf
from tensorflow.math import sqrt, abs, minimum, maximum
from vegasflow.configflow import DTYPE, DTYPEINT, int_me, float_me
from config import DTYPECOMPLEX
from config import complex_tf, complex_me

wave_signature=[
    tf.TensorSpec(shape=[None,4], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE)
]

scalar_signature=[
    tf.TensorSpec(shape=[None,4], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE)
]

sign_signature=[
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[], dtype=DTYPE),
]

signvec_signature=[
    tf.TensorSpec(shape=[], dtype=DTYPE),
    tf.TensorSpec(shape=[None], dtype=DTYPE),
]


@tf.function(input_signature=sign_signature)
def sign(x,y):
    """Fortran's sign transfer function"""
    # print("sign")
    # dropping the checks for the moment
    return x*tf.math.sign(y)


@tf.function(input_signature=signvec_signature)
def signvec(x,y):
    """Fortran's sign transfer function"""
    # print("signvec")
    # dropping the checks for the moment
    return x*tf.math.sign(y)



@tf.function(input_signature=scalar_signature)
def sxxxxx(p,nss):
    """Defines a scalar."""
    # Note: here p[:,i] selects the momentum dimension and is a [nevt,] tensor
    nevts = tf.shape(p, out_type=DTYPEINT)[0]    
    v0 = tf.expand_dims(complex_tf(p[:,0]*nss,p[:,3]*nss), 0) # [nevt,] complex
    v1 = tf.expand_dims(complex_tf(p[:,1]*nss,p[:,2]*nss), 0) # [nevt,] complex
    v = tf.expand_dims(complex_tf(1.,0.), 0)
    fi = tf.concat([v0,v1,v], axis=0)
    return fi

@tf.function(input_signature=wave_signature)
def ixxxxx(p,fmass,nhel,nsf):
    """Defines an inflow fermion."""
    # print("ixxxxx")
    # Note: here p[:,i] selects the momentum dimension and is a [nevt,] tensor
    nevts = tf.shape(p, out_type=DTYPEINT)[0]    
    v0 = tf.expand_dims(complex_tf(-p[:,0]*nsf,-p[:,3]*nsf), 0) # [nevt,] complex
    v1 = tf.expand_dims(complex_tf(-p[:,1]*nsf,-p[:,2]*nsf), 0) # [nevt,] complex
    nh = nhel*nsf # either +1 or -1
    ip = (1+nh)//2
    im = (1-nh)//2
    def true_branch():
        pp = minimum(p[:,0], sqrt(p[:,1]**2 + p[:,2]**2 + p[:,3]**2 )) # [nevt,]
        def true_fn():
            sqm = sqrt(abs(fmass))
            sqm = tf.stack([sqm, sign(sqm,fmass)]) # [fmass, fmass] ---> TODO: why calling sign on the result of a sqrt ????
            v2 = complex_tf(ip*sqm[int_me(ip)],0.) # just a complex number
            v3 = complex_tf(im*nsf*sqm[int_me(ip)],0.)
            v4 = complex_tf(ip*nsf*sqm[int_me(im)],0.)
            v5 = complex_tf(im*sqm[int_me(im)],0.)
            v = tf.stack([v2,v3,v4,v5]) # [4,] complex
            return tf.reshape(v, [4,1])
        def false_fn():
            sf = tf.stack([(1+nsf+(1-nsf)*nh)*0.5,(1+nsf-(1-nsf)*nh)*0.5], axis=0) # [2,]
            omega = tf.stack([sqrt(p[:,0]+pp),fmass/(sqrt(p[:,0]+pp))], axis=0) # [2, nevt]
            sfomeg = tf.stack([sf[0]*omega[int_me(ip)],sf[1]*omega[int_me(im)]], axis=0) # [2,nevt]
            pp3 = maximum(pp+p[:,3],0.) # [nevt,]
            chi1 = tf.where(pp3==0, complex_tf(-nh,0), complex_tf(nh*p[:,1]/sqrt(2.*pp*pp3), p[:,2]/sqrt(2.*pp*pp3))) # [nevt,] complex
            chi2 = complex_tf(sqrt(pp3*0.5/pp),0.) # [nevt,] complex 
            chi = tf.stack([chi2, chi1], axis=0) # [2, nevt] complex
            v2 = complex_tf(sfomeg[0], 0.)*chi[int_me(im)] # [nevt,] complex
            v3 = complex_tf(sfomeg[0], 0.)*chi[int_me(ip)]
            v4 = complex_tf(sfomeg[1], 0.)*chi[int_me(im)]
            v5 = complex_tf(sfomeg[1], 0.)*chi[int_me(ip)]
            return tf.stack([v2,v3,v4,v5], axis=0) # [nevt, 4] complex
        cond = tf.expand_dims(pp==0, 0)
        return tf.where(cond, true_fn(), false_fn()) # [nevt, 4] complex
    def false_branch():
        sqp0p3 = sqrt(maximum(p[:,0]+p[:,3],0.))*nsf # [nevt,]
        def true_fn():
            return complex_tf(-nhel*sqrt(2.*p[:,0]),0.) # [nevt,] complex
        def false_fn():
            return complex_tf(nh*p[:,1]/sqp0p3,p[:,2]/sqp0p3) # [nevt,] complex
        chi1 = tf.where(sqp0p3==0, true_fn(), false_fn())
        chi = tf.stack([complex_tf(sqp0p3,0.),chi1], axis=0) # [2, nevt]
        def true_fn():
            v4 = chi[0] # [nevt,] complex
            v5 = chi[1] # [nevt,] complex
            v2 = tf.ones_like(v4)*complex_tf(0.,0.) # [nevt,] complex
            v3 = tf.ones_like(v4)*complex_tf(0.,0.) # [nevt,] complex
            return tf.stack([v2,v3,v4,v5], axis=0)
        def false_fn():
            v2 = chi[1]
            v3 = chi[0]
            v4 = tf.ones_like(v2)*complex_tf(0.,0.)
            v5 = tf.ones_like(v2)*complex_tf(0.,0.)
            return tf.stack([v2,v3,v4,v5], axis=0)
        return tf.where(nh==1, true_fn(), false_fn())
    massive = fmass != 0
    v = tf.where(massive, true_branch(), false_branch())
    fi = tf.concat([v0,v1,v], axis=0)
    return fi


@tf.function(input_signature=wave_signature)
def oxxxxx(p,fmass,nhel,nsf):
    """ initialize an outgoing fermion"""
    # print("oxxxxx")
    nevts = tf.shape(p, out_type=DTYPEINT)[0]
    v0 = tf.expand_dims(complex_tf(p[:,0]*nsf,p[:,3]*nsf), 0) # [nevt,] complex
    v1 = tf.expand_dims(complex_tf(p[:,1]*nsf,p[:,2]*nsf), 0) # [nevt,] complex
    nh = nhel*nsf # either +1 or -1
    sqp0p3 = sqrt(maximum(p[:,0]+p[:,3],0.))*nsf # [nevt,]
    def true_branch():
        pp = minimum(p[:,0], sqrt(p[:,1]**2 + p[:,2]**2 + p[:,3]**2 )) # [nevt,]
        def true_fn():
            sqm = sqrt(abs(fmass))
            sqm = tf.stack([sqm, sign(sqm,fmass)]) # [fmass, fmass] ---> why calling sign on the result of a sqrt ????
            ip = -((1-nh)//2) * nhel
            im = (1+nh)//2 * nhel
            v2 = complex_tf(im*sqm[int_me(abs(im))],0.) # just a complex number
            v3 = complex_tf(ip*nsf*sqm[int_me(abs(im))],0.)
            v4 = complex_tf(im*nsf*sqm[int_me(abs(ip))],0.)
            v5 = complex_tf(ip*sqm[int_me(abs(ip))],0.)
            v = tf.stack([v2,v3,v4,v5]) # [4,] complex
            return tf.reshape(v, [4,1])
        def false_fn():
            sf = tf.stack([(1+nsf+(1-nsf)*nh)*0.5,(1+nsf-(1-nsf)*nh)*0.5], axis=0) # [2,]
            omega = tf.stack([sqrt(p[:,0]+pp),fmass/(sqrt(p[:,0]+pp))], axis=0) # [2, nevt]
            ip = (1+nh)//2
            im = (1-nh)//2
            sfomeg = tf.stack([sf[0]*omega[int_me(ip)],sf[1]*omega[int_me(im)]], axis=0) # [2,nevt]
            pp3 = maximum(pp+p[:,3],0.) # [nevt,]
            chi1 = tf.where(pp3==0, complex_tf(-nh,0), complex_tf(nh*p[:,1]/sqrt(2.*pp*pp3), -p[:,2]/sqrt(2.*pp*pp3))) # [nevt,] complex
            chi2 = complex_tf(sqrt(pp3*0.5/pp),0.) # [nevt,] complex 
            chi = tf.stack([chi2, chi1], axis=0) # [2, nevt] complex
            v2 = complex_tf(sfomeg[1], 0.)*chi[int_me(im)] # [nevt,] complex
            v3 = complex_tf(sfomeg[1], 0.)*chi[int_me(ip)]
            v4 = complex_tf(sfomeg[0], 0.)*chi[int_me(im)]
            v5 = complex_tf(sfomeg[0], 0.)*chi[int_me(ip)]
            return tf.stack([v2,v3,v4,v5], axis=0) # [4, nevt] complex
        cond = tf.expand_dims(pp==0, 0)
        return tf.where(cond, true_fn(), false_fn()) # [4, nevt] complex
    def false_branch():
        def true_fn():
            return complex_tf(-nhel*sqrt(2.*p[:,0]),0.) # [nevt,] complex
        def false_fn():
            return complex_tf(nh*p[:,1]/sqp0p3, -p[:,2]/sqp0p3) # [nevt,] complex
        chi1 = tf.where(sqp0p3==0, true_fn(), false_fn())
        chi = tf.stack([complex_tf(sqp0p3,0.),chi1], axis=0) # [2, nevt]
        def true_fn():
            v2 = chi[0] # [nevt,] complex
            v3 = chi[1] # [nevt,] complex
            v4 = tf.ones_like(v2)*complex_tf(0.,0.) # [nevt,] complex
            v5 = tf.ones_like(v2)*complex_tf(0.,0.) # [nevt,] complex
            return tf.stack([v2,v3,v4,v5], axis=0)
        def false_fn():
            v4 = chi[1]
            v5 = chi[0]
            v2 = tf.ones_like(v4)*complex_tf(0.,0.)
            v3 = tf.ones_like(v4)*complex_tf(0.,0.)
            return tf.stack([v2,v3,v4,v5], axis=0)
        return tf.where(nh==1, true_fn(), false_fn())
    massive = fmass != 0
    v = tf.where(massive, true_branch(), false_branch())
    fo = tf.concat([v0,v1,v], axis=0)
    return fo


@tf.function(input_signature=wave_signature)
def vxxxxx(p,vmass,nhel,nsv):
    """ initialize a vector wavefunction. nhel=4 is for checking BRST"""
    # print("vxxxxx")
    nevts = tf.shape(p, out_type=DTYPEINT)[0]
    hel0 = 1.-abs(nhel)


    sqh = float_me(sqrt(0.5))
    nsvahl = nsv*abs(nhel)
    pt2 = p[:,1]**2 + p[:,2]**2 
    pp = minimum(p[:,0],sqrt(pt2 + p[:,3]**2))
    pt = minimum(pp,sqrt(pt2))

    v0 = tf.expand_dims(complex_tf(p[:,0]*nsv,p[:,3]*nsv), 0) # [1,nevts] complex
    v1 = tf.expand_dims(complex_tf(p[:,1]*nsv,p[:,2]*nsv), 0)

    def true_branch():
        def true_f():
            vc2 = tf.ones(nevts, dtype=DTYPE)
            vc3= p[:,1]/p[:,0]
            vc4= p[:,2]/p[:,0]
            vc5= p[:,3]/p[:,0]
            return complex_me(tf.stack([vc2,vc3,vc4,vc5], axis=0))
        def false_f():
            vc2 = p[:,0]/vmass
            vc3 = p[:,1]/vmass
            vc4 = p[:,2]/vmass
            vc5 = p[:,3]/vmass
            return complex_me(tf.stack([vc2,vc3,vc4,vc5], axis=0))
        massless = vmass == 0
        v = tf.where(massless, true_f(), false_f())        
        return tf.concat([v0,v1,v], axis=0) # [6,nevts] complex
    def false_branch():
        def true_fn():
            def true_fn():
                hel0 = 1.-abs(nhel)
                v2 = tf.ones(nevts, dtype=DTYPECOMPLEX)
                v3 = tf.ones_like(v2)*complex_tf(-nhel*sqh,0.)
                v4 = tf.ones_like(v2)*complex_tf(0.,nsvahl*sqh)
                v5 = tf.ones_like(v2)*complex_tf(hel0,0.)
                return tf.stack([v2,v3,v4,v5], axis=0) # [4,nevts] complex
            def false_fn():
                emp = p[:,0]/(vmass*pp)
                v2 = tf.expand_dims(complex_tf(hel0*pp/vmass,0.), 0)
                v5 = tf.expand_dims(complex_tf(hel0*p[:,3]*emp+nhel*pt/pp*sqh, 0), 0)
                def true_f():
                    pzpt = p[:,3]/(pp*pt)*sqh*nhel
                    v3 = complex_tf(hel0*p[:,1]*emp-p[:,1]*pzpt, \
                        -nsvahl*p[:,2]/pt*sqh)
                    v4 = complex_tf(hel0*p[:,2]*emp-p[:,2]*pzpt, \
                        nsvahl*p[:,1]/pt*sqh) 
                    return tf.stack([v3,v4], axis=0)
                def false_f():
                    v3 = tf.ones(nevts, dtype=DTYPECOMPLEX)*complex_tf(-nhel*sqh,0.)
                    v4 = complex_tf(0.,nsvahl*signvec(sqh,p[:,3])) # <------ this enters the sign operation with y as a real vector
                    return tf.stack([v3,v4], axis=0)
                condition = tf.expand_dims(pt!=0, 0)
                v34 = tf.where(condition, true_f(), false_f())            
                return tf.concat([v2,v34,v5], axis=0) # [4,nevts] complex
            cond = tf.expand_dims(pp==0, 0)
            return tf.where(cond, true_fn(), false_fn())
        def false_fn():
            pp = p[:,0]
            pt = sqrt(p[:,1]**2 + p[:,2]**2)
            v2 = tf.ones([1,nevts], dtype=DTYPECOMPLEX)*complex_tf(0.,0.)
            v5 = tf.expand_dims(complex_tf(nhel*pt/pp*sqh, 0.), 0)
            def true_fn():
                pzpt = p[:,3]/(pp*pt)*sqh*nhel
                v3 = complex_tf(-p[:,1]*pzpt,-nsv*p[:,2]/pt*sqh)
                v4 = complex_tf(-p[:,2]*pzpt,nsv*p[:,1]/pt*sqh)
                return tf.stack([v3,v4], axis=0)
            def false_fn():
                v3 = tf.ones(nevts, dtype=DTYPECOMPLEX)*complex_tf(-nhel*sqh,0.)
                v4 = complex_tf(0.,nsv*signvec(sqh,p[:,3])) # <------ this enters the sign operation with y as a real vector
                return tf.stack([v3,v4], axis=0)
            cond = tf.expand_dims(pt!=0, 0)
            v34 = tf.where(cond, true_fn(), false_fn())
            return tf.concat([v2,v34,v5], axis=0)
        massive = vmass != 0
        v =  tf.where(massive, true_fn(), false_fn())
        return tf.concat([v0,v1,v], axis=0)
    BRST = nhel == 4
    return tf.where(BRST, true_branch(), false_branch())
