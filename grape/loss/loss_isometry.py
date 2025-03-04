import numpy as np
from grape.calc import calc_U_tri, calc_U_tri_single, accumulate, accumulate_single


def jacobian_isometry(waveform,iso_target,iso_init,params,ops,d,ld):
    """Calculate loss function and its gradient simultaneously

    Arguments
        waveform[3*STEPS]  : phases stored in reverse time order, x then y then mw in flattened vector
        iso_target[DIM,]   : target state
        iso_init[DIM,]     : initial state
        params dict        : dictionary of constants
        ops[9,:,:]         : operators of Hamiltonian expansion
        d[9,DIM]           : diagonal elements of ops
        ld[9,DIM-1]        : lower diagonal elements of ops

    Return
        loss float         : loss function evaluated at waveform
        gradient[3*STEPS,] : gradient of loss with respect to waveform
    """

    u_s,du_s = calc_U_tri(waveform,params,ops,d,ld)
    STEPS,NPAR,DIM,_ = du_s.shape
    U, dU_s, _, _ = accumulate(u_s,du_s)

    # U_fin is shape [DIM**2,]
    # dU is shape [STEPS,NPAR,DIM,DIM]
    # STEPS = dU.shape[0]
    
    # overlap = np.dot(iso_target.conj(),U[:,params.I33])
    overlap = iso_target.conj()@U@iso_init
    gradient = np.zeros(params.NPAR*STEPS,dtype=np.float64)
    
    gradient = -np.real( overlap.conj() * (dU_s@iso_init)@iso_target.conj()   )/np.abs(overlap)

    return gradient.flatten(order='F')

def loss_isometry(waveform,iso_target,iso_init,params,ops,d,ld):
    """Calculate loss function and its gradient simultaneously

    Arguments
        waveform[3*STEPS]  : phases stored in reverse time order, x then y then mw in flattened vector
        iso_target[DIM,]   : target state
        iso_init[DIM,]     : initial state
        params dict        : dictionary of constants
        ops[9,:,:]         : operators of Hamiltonian expansion
        d[9,DIM]           : diagonal elements of ops
        ld[9,DIM-1]        : lower diagonal elements of ops

    Return
        loss float         : loss function evaluated at waveform
    """
    # assume U_target shape [DIM**2,]
    u_s = calc_U_tri_single(waveform,params,ops,d,ld)
    U_fin = accumulate_single(u_s)
    
    # # U_s is shape [STEPS,DIM**2,]
    # U_fin = U_s[-1,:].reshape(params.DIM,params.DIM,order='F')
    
    overlap = iso_target.conj()@U_fin@iso_init

    # overlap = np.dot(iso_target.conj(),U_fin[:,params.I33])

    loss = 1 - np.abs(overlap)    

    return loss

