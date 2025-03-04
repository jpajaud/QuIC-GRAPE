import os, time

import numpy as np
from scipy.optimize import minimize
from grape.calc import calc_ops, expand_from_subspace
from grape.loss import loss_fidelity, jacobian_fidelity
from grape import WaveformParams


class FidelityTarget:
    def __init__(self,stop_iter):
        self.waveform=None
        self.stop_iter = stop_iter
    def loss(self,waveform,U_target,params,ops,d,ld):
        loss_val = loss_fidelity(waveform,U_target,params,ops,d,ld)
        print(loss_val)
        if loss_val<self.stop_iter:
            self.waveform = waveform
            raise StopIteration()
        return loss_val
    def jacobian(self,waveform,*args):
        return jacobian_fidelity(waveform,*args)


def optimize_conventional(U_target,STEPS=150,DT=4e-6,stop_iter=1e-5,max_iter=1000,subspace_vec=None,subspace_type='isolated',savefile=None):

    if subspace_vec is not None:
        assert U_target.shape[0] == sum(subspace_vec)
        U_target = expand_from_subspace(U_target,subspace_vec=subspace_vec)

    params = WaveformParams.prepare_params(STEPS=STEPS,DT=DT,stop_iter=stop_iter,subspace_vec=subspace_vec,subspace_type=subspace_type)
    ops,d,ld = calc_ops(params)
    
    objective = FidelityTarget(stop_iter)

    # fidelity_target = FidelityTarget(loss_fidelity,jacobian_fidelity,stop_iter,max_iter)
    waveform0 = np.random.uniform(0,2*np.pi,size=params.STEPS*params.NPAR)
    elapsed = -1
    try:
        start = time.time()
        res = minimize(objective.loss,waveform0,jac=objective.jacobian,args=(U_target,params,ops,d,ld),options={'maxiter':max_iter})#,callback=fidelity_target.callback)
    except StopIteration:
        print('StopIteration raised')
        waveform = objective.waveform
    else:
        print(res.message)
        waveform = res.x
    finally:
        elapsed = time.time() - start

    if savefile is not None:
        data = {'waveform':waveform,
                'U_target':U_target,
                'time_elapsed':elapsed}
        data.update(params.save_dict)
        np.savez(savefile,**data)

    return waveform,params,elapsed
