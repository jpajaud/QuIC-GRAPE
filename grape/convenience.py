import numpy as np
from scipy.linalg import expm
from scipy.io import savemat
from grape.calc import calc_ops, calc_U_tri_single, accumulate_single, gen_GellMann_traceless

def calc_U(waveform,params):
    ops,d,ld = calc_ops(params)
    u_s = calc_U_tri_single(waveform,params,ops,d,ld)
    return accumulate_single(u_s)

def GellMann_transformation(r):
    D = int(np.sqrt(len(r)+1))
    basis = gen_GellMann_traceless(D)
    return expm(-1j*np.tensordot(r,basis,axes=(0,0)))

def fidelity(U1,U2,V1=None,V2=None):
    
    if V1 is not None:
        U11 = (V1@U1@V1.T.conj())
    else:
        U11 = U1

    if V2 is not None:
        U21 = (V2@U2@V2.T.conj())
    else:
        U21 = U2

    return np.abs(np.sum(U11*U21.conj()))/U1.shape[0]

def concat_save(filename,waveforms,params):
    
    rf_wave = np.array([[],[]],dtype=np.float64)
    mw_wave = np.array([[]])

    for waveform in waveforms:
        wave3d = waveform.reshape(params.NPAR,-1)
        rf_wave = np.hstack((rf_wave,wave3d[:2,:]))
        mw_wave = np.hstack((mw_wave,wave3d[2:,:]))

    points = rf_wave.shape[1]
    tot_time = points*params.DT
    rf_freq = params.rf_freq
    rf_amp_x = params.rf_amp_x

    savemat(filename,{'opt_params':{'rf_wave':rf_wave,'mw_wave':mw_wave,'points':points,'rf_freq':rf_freq,'rf_amp_x':rf_amp_x,'tot_time':tot_time}})