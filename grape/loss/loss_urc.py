import numpy as np
from grape.calc import calc_U_tri, calc_U_tri_single, accumulate, calc_M, calc_dM


def loss_robust_urc(waveform,params,ops,d,ld):

    u_s = calc_U_tri_single(waveform,params,ops,d,ld)
    STEPS,DIM,_ = u_s.shape

    U = np.eye(DIM,dtype=np.complex128)
    U_f = np.zeros((STEPS,DIM,DIM),dtype=np.complex128) # consider renaming to uU

    for i in range(STEPS):
        U[:,:] = u_s[i,:,:]@U
        U_f[i,:,:] = U[:,:]
    
    M = calc_M(U_f,params.ks,params.vs)
    return M

def jacobian_robust_urc(waveform,params,ops,d,ld):

    u_s,du_s = calc_U_tri(waveform,params,ops,d,ld)
    _, dU_s, U_f, U_b_conj = accumulate(u_s,du_s)

    dM = calc_dM(dU_s, U_f, U_b_conj, params.ks, params.vs)
    return dM.flatten(order='F')



