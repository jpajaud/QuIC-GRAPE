import os, time

import numpy as np
from scipy.optimize import minimize
from grape.calc import calc_ops, expand_from_subspace
from grape.loss import loss_fidelity, jacobian_fidelity
from grape import WaveformParams


class RobustTarget:
    def __init__(self,stop_iter):
        self.waveform=None
        self.stop_iter = stop_iter
    def loss(self,waveform,U_target,params,ops_list,d_list,ld_list):
        loss_val = 0
        for ops,d,ld in zip(ops_list,d_list,ld_list):
            # don't need separate params object because only, STEPS, DIM, NPAR and so forth are access by loss_fidelity
            loss_val += loss_fidelity(waveform,U_target,params,ops,d,ld)
        loss_val /= len(ops_list)
        print(loss_val)
        if loss_val<self.stop_iter:
            self.waveform = waveform
            raise StopIteration()
        return loss_val
    def jacobian(self,waveform,U_target,params,ops_list,d_list,ld_list):
        gradient = np.zeros(waveform.shape)
        for ops,d,ld in zip(ops_list,d_list,ld_list):
            # don't need separate params object because only, STEPS, DIM, NPAR and so forth are access by loss_fidelity
            gradient += jacobian_fidelity(waveform,U_target,params,ops,d,ld)
        return gradient/len(ops_list)


def optimize_robust(U_target,STEPS=150,DT=4e-6,stop_iter=1e-3,max_iter=1000,subspace_vec=None,subspace_type='isolated',error={'set_detuning':[-40,0,40]},savefile=None):

    if subspace_vec is not None:
        assert U_target.shape[0] == sum(subspace_vec)
        U_target = expand_from_subspace(U_target,subspace_vec=subspace_vec)

    params = WaveformParams.prepare_params(STEPS=STEPS,DT=DT,stop_iter=stop_iter,subspace_vec=subspace_vec,subspace_type=subspace_type)
    

    # make list of ops,d,ld for all errors
    ops_list = []
    d_list = []
    ld_list = []
    if len(error) == 0:
        ops,d,ld = calc_ops(params)
        ops_list.append(ops)
        d_list.append(d)
        ld_list.append(ld)
    for name, values in error.items():
        set_error = getattr(params,name)
        print('robust')
        for value in values:
            set_error(value)
            print(params.rf_det)
            ops,d,ld = calc_ops(params)
            ops_list.append(ops)
            d_list.append(d)
            ld_list.append(ld)
            params.reset_errors()
    
    
    objective = RobustTarget(stop_iter)

    waveform0 = np.random.uniform(0,2*np.pi,size=params.STEPS*params.NPAR)
    elapsed = -1
    try:
        start = time.time()
        res = minimize(objective.loss,waveform0,jac=objective.jacobian,args=(U_target,params,ops_list,d_list,ld_list),options={'gtol':0,'maxiter':max_iter})#,callback=fidelity_target.callback)
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
