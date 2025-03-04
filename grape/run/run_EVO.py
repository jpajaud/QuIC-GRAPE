import os, time

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from grape.calc import calc_ops, expand_from_subspace, gen_GellMann_traceless, calc_U_tri_single, accumulate_single
from grape.loss import loss_EVO, jacobian_EVO
from grape import WaveformParams

def calc_exact_map(waveform_U,r,basis,U_target,params,ops,d,ld):
    # this seems to work well enough for Unitaries with degeneracies

    V_opt = expm(-1j*np.tensordot(r,basis,axes=(0,0)))

    u_s = calc_U_tri_single(waveform_U,params,ops,d,ld)
    U = accumulate_single(u_s)

    Δ = np.angle(np.diag((V_opt@U@V_opt.T.conj())@U_target.T.conj())).mean()
    wu,vu = np.linalg.eig(U)
    ww,vw = np.linalg.eig(U_target)

    # s = np.outer(wu,ww.conj())
    inds_u = np.argsort(np.angle(np.exp(-1j*Δ)*wu))
    inds_w = np.argsort(np.angle(ww))

    return vw[:,inds_w]@vu[:,inds_u].T.conj()


class EVOTarget:
    def __init__(self,stop_iter):
        self.waveform=None
        self.stop_iter = stop_iter
    def loss(self,waveform,basis,U_target,params,ops,d,ld):
        loss_val = loss_EVO(waveform,basis,U_target,params,ops,d,ld)
        print(loss_val)
        if loss_val<self.stop_iter:
            self.waveform = waveform
            raise StopIteration()
        return loss_val
    def jacobian(self,waveform,*args):
        return jacobian_EVO(waveform,*args)


def optimize_EVO(U_target,STEPS=150,DT=4e-6,stop_iter=1e-5,max_iter=1000,subspace_vec=None,subspace_type='isolated',savefile=None):

    if subspace_vec is not None:
        assert U_target.shape[0] == sum(subspace_vec)
        U_target = expand_from_subspace(U_target,subspace_vec=subspace_vec)

    params = WaveformParams.prepare_params(STEPS=STEPS,DT=DT,stop_iter=stop_iter,subspace_vec=subspace_vec,subspace_type=subspace_type)
    ops,d,ld = calc_ops(params)
    
    basis = gen_GellMann_traceless(params.DIM)
    r0 = np.random.randn(basis.shape[0])

    objective = EVOTarget(stop_iter)

    waveform0 = np.concatenate((np.random.uniform(0,2*np.pi,size=params.STEPS*params.NPAR),r0))
    elapsed = -1
    try:
        start = time.time()
        res = minimize(objective.loss,waveform0,jac=objective.jacobian,args=(basis,U_target,params,ops,d,ld),options={'maxiter':max_iter,'gtol':-1})
    except StopIteration: # add except keyboard interrupt
        print('StopIteration raised')
        waveform = objective.waveform
    else:
        print(res.message)
        waveform = res.x
    finally:
        elapsed = time.time() - start

    i = params.NPAR*params.STEPS
    waveform_U = waveform[:i]
    r = waveform[i:]
    
    V_map = calc_exact_map(waveform_U,r,basis,U_target,params,ops,d,ld)
    # now V_map@U_fin@V_map.T.conj() == U_target

    if savefile is not None:
        data = {'waveform':waveform_U,
                'r':r,
                'V_map':V_map,
                'U_target':U_target,
                'time_elapsed':elapsed}
        data.update(params.save_dict)
        np.savez(savefile,**data)

    return waveform,params,elapsed
