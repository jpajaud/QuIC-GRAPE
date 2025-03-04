import numpy as np
from scipy.linalg import expm
from grape.calc import calc_U_tri, calc_U_tri_single, accumulate, accumulate_single

def calc_dV(r,basis):
    H = np.tensordot(r,basis,axes=(0,0))
    DIM = H.shape[0]
    w,V = np.linalg.eigh(H)
    wexp = np.exp(-1j*w)
    U = V@np.diag(wexp)@V.T.conj()
    num  = np.subtract.outer(wexp,wexp)
    den  = np.subtract.outer(w,w) + np.eye(DIM)
    fact = num/den + np.diag(-1j*wexp)
    return U,np.einsum('hi,ji,ajk,kl,il,ml->ahm',V,V.conj(),basis,V,fact,V.conj(),optimize=True)

def loss_EVO(waveform,basis,U_target,params,ops,d,ld):
    i = params.NPAR*params.STEPS
    waveform_U = waveform[:i]
    r = waveform[i:]

    H = np.tensordot(r,basis,axes=(0,0))
    V = expm(-1j*H)

    U_target_c = U_target.conj()

    u_s = calc_U_tri_single(waveform_U,params,ops,d,ld)
    U = accumulate_single(u_s)

    overlap = np.sum((V@U@V.T.conj())*U_target_c)
    loss = 1 - np.abs(overlap)/params.SUBSPACE_DIM
    return loss

def jacobian_EVO(waveform,basis,U_target,params,ops,d,ld):
    i = params.NPAR*params.STEPS
    waveform_U = waveform[:i]
    r = waveform[i:]

    U_target_c = U_target.conj()

    V, dV = calc_dV(r,basis)
    
    u_s,du_s = calc_U_tri(waveform_U,params,ops,d,ld)
    U, dU_s, _, _ = accumulate(u_s,du_s)
    # STEPS,NPAR,DIM,_ = du_s.shape

    overlap = np.sum((V@U@V.T.conj())*U_target_c)
    doverlap_U = np.einsum('ij,srjk,lk,il->sr',V,dU_s,V.conj(),U_target_c)
    doverlap_r = np.einsum('aij,jk,lk,il->a',dV,U,V.conj(),U_target_c)+np.einsum('ij,jk,alk,il->a',V,U,dV.conj(),U_target_c)

    gradient_U = -np.real( overlap.conj()*doverlap_U )/(np.abs(overlap)*params.SUBSPACE_DIM)
    gradient_r = -np.real( overlap.conj()*doverlap_r )/(np.abs(overlap)*params.SUBSPACE_DIM)

    return np.concatenate((gradient_U.flatten(order='F'),gradient_r))
    
