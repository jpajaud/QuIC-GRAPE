import os, time

import numpy as np
from scipy.optimize import minimize
from grape.calc import calc_ops
from grape.loss import loss_isometry, jacobian_isometry
from grape import WaveformParams


class IsometryTarget:
    def __init__(self,stop_iter):
        self.waveform=None
        self.stop_iter = stop_iter
    def loss(self,waveform,iso_target,iso_init,params,ops,d,ld):
        loss_val = loss_isometry(waveform,iso_target,iso_init,params,ops,d,ld)
        print(loss_val)
        if loss_val<self.stop_iter:
            self.waveform = waveform
            raise StopIteration()
        return loss_val
    def jacobian(self,waveform,*args):
        return jacobian_isometry(waveform,*args)


def optimize_isometry(iso_target,iso_init=None,STEPS=25,DT=4e-6,stop_iter=1e-5,max_iter=1000,savefile=None):


    params = WaveformParams.prepare_params(STEPS=STEPS,DT=DT,stop_iter=stop_iter)
    ops,d,ld = calc_ops(params)
    
    objective = IsometryTarget(stop_iter)

    # fidelity_target = FidelityTarget(loss_fidelity,jacobian_fidelity,stop_iter,max_iter)
    
    if iso_init is None:
        iso_init = params.init_state.ravel()
    else:
        assert isinstance(iso_init,np.ndarray), "iso_init must be numpy.ndarray"
        
        iso_init = iso_init.ravel()
    iso_target = iso_target.ravel()

    waveform0 = np.random.uniform(0,2*np.pi,size=params.STEPS*params.NPAR)
    elapsed = -1
    try:
        start = time.time()
        res = minimize(objective.loss,waveform0,jac=objective.jacobian,args=(iso_target,iso_init,params,ops,d,ld),options={'maxiter':max_iter})#,callback=fidelity_target.callback)
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
                'iso_target':iso_target,
                'iso_init':iso_init,
                'time_elapsed':elapsed}
        data.update(params.save_dict)
        np.savez(savefile,**data)

    return waveform,params,elapsed
