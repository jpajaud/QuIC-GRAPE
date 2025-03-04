import os, time

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from grape.calc import calc_ops, expand_from_subspace
from grape import WaveformParams
from grape.loss import loss_robust_urc, jacobian_robust_urc, loss_fidelity, jacobian_fidelity
from grape.run import optimize_conventional

class RobustURCTarget:
    def __init__(self,stop_iter):
        self.waveform=None
        self.stop_iter = stop_iter
    def loss(self,waveform,*args):
        loss_val = loss_robust_urc(waveform,*args)
        if loss_val<self.stop_iter:
            self.waveform = waveform
            raise StopIteration()
        print(loss_val)
        return loss_val
    def jacobian(self,waveform,*args):
        return jacobian_robust_urc(waveform,*args)

def optimize_robust_urc(U_target,STEPS=150,DT=4e-6,stop_iter=1e-5,fidelity_bound=.001,M_stop=-1,max_iter=1000,subspace_vec=None,subspace_type='isolated',savefile=None):

    
    waveform_start,params,elapsed = optimize_conventional(U_target=U_target,STEPS=STEPS,DT=DT,stop_iter=stop_iter,max_iter=max_iter,subspace_vec=subspace_vec,subspace_type=subspace_type,savefile=None)    
    
    ops,d,ld = calc_ops(params)

    if subspace_vec is not None:
        assert U_target.shape[0] == sum(subspace_vec)
        U_target = expand_from_subspace(U_target,subspace_vec=subspace_vec)
    
    constraint_function = lambda x:loss_fidelity(x,U_target,params,ops,d,ld)
    constraint_jacobian = lambda x:jacobian_fidelity(x,U_target,params,ops,d,ld)
    constraint = NonlinearConstraint(constraint_function,0,fidelity_bound,jac=constraint_jacobian)
    
    objective = RobustURCTarget(M_stop)
    try:
        start = time.time()
        res = minimize(objective.loss,waveform_start,jac=objective.jacobian,constraints=constraint,args=(params,ops,d,ld),method='SLSQP',options={'maxiter':max_iter})
        
    except StopIteration:
        print('StopIteration raised')
        waveform = objective.waveform
    else:
        print(res.message)
        waveform = res.x
    finally:
        elapsed += time.time()-start

    if savefile is not None:
        data = {'waveform':waveform,
                'U_target':U_target,
                'time_elapsed':elapsed}
        data.update(params.save_dict)
        np.savez(savefile,**data)

    return waveform, params, elapsed
