import numpy as np
from grape.calc import calc_U_tri, calc_U_tri_single, accumulate, accumulate_single

def jacobian_fidelity(waveform,U_target,params,ops,d,ld):
    # assume U_target shape [DIM,DIM]
    u_s,du_s = calc_U_tri(waveform,params,ops,d,ld)
    STEPS,NPAR,DIM,_ = du_s.shape
    U, dU_s, _, _ = accumulate(u_s,du_s)
    
    overlap = np.einsum('ij,ij',U_target.conj(),U,optimize=False) # optimize==False is 10 times faster
    
    # overlap = U_target.conj()@U.flatten(order='F')
    
    # dU_s_c = dU_s.reshape(STEPS,NPAR,DIM**2).conj()
    gradient = -np.real(  overlap  *   np.einsum('ijkl,kl->ij', dU_s.conj(),U_target,optimize=True)     )/(params.SUBSPACE_DIM*np.abs(overlap))
    # gradient = -np.real(  overlap  *   dU_s_c@U_target.flatten()  )/(params.SUBSPACE_DIM*np.abs(overlap))
    return gradient.flatten(order='F') # flatten by column

def loss_fidelity(waveform,U_target,params,ops,d,ld):
    # assume U_target shape [DIM,DIM]
    u_s = calc_U_tri_single(waveform,params,ops,d,ld)
    U = accumulate_single(u_s)

    overlap = np.einsum('ij,ij',U_target.conj(),U,optimize=False) # optimize==False is 10 times faster
    
    loss = 1 - np.abs(overlap)/params.SUBSPACE_DIM
    return loss

