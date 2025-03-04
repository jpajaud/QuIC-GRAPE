import os, time

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from grape.calc import calc_ops, expand_from_subspace
from grape import WaveformParams
from grape.loss import loss_fidelity, jacobian_fidelity
from grape.run import optimize_conventional

class RobustV2Target:
    def __init__(self,stop_iter=0):
        self.waveform=None
        self.stop_iter = stop_iter
    def loss(self,waveform,U_target,params,ops_list,d_list,ld_list):
        
        loss0 = loss_fidelity(waveform,U_target,params,ops_list[0],d_list[0],ld_list[0])
        lossΔ = 0
        N_ops = len(ops_list)-1
        for i in range(N_ops):
            lossΔ += loss_fidelity(waveform,U_target,params,ops_list[i+1],d_list[i+1],ld_list[i+1])
        loss_val = lossΔ - loss0

        print(loss_val,loss0)
        if loss_val<self.stop_iter:
            self.waveform = waveform
            raise StopIteration()
        return loss_val
    def jacobian(self,waveform,U_target,params,ops_list,d_list,ld_list):
        gradient0 = jacobian_fidelity(waveform,U_target,params,ops_list[0],d_list[0],ld_list[0])
        gradientΔ = np.zeros_like(gradient0)
        N_ops = len(ops_list)-1
        for i in range(N_ops):
            gradientΔ += loss_fidelity(waveform,U_target,params,ops_list[i+1],d_list[i+1],ld_list[i+1])
        return gradientΔ - gradient0


def optimize_robust_v2(U_target,STEPS=150,DT=4e-6,stop_iter=1e-5,fidelity_bound=.001,M_stop=-1,max_iter=1000,subspace_vec=None,subspace_type='isolated',error=('set_detuning',[-40,40]),savefile=None):

    
    waveform_start,params,elapsed = optimize_conventional(U_target=U_target,STEPS=STEPS,DT=DT,stop_iter=stop_iter,max_iter=max_iter,subspace_vec=subspace_vec,subspace_type=subspace_type,savefile=None)    
    
    ops,d,ld = calc_ops(params)

    if subspace_vec is not None:
        assert U_target.shape[0] == sum(subspace_vec)
        U_target = expand_from_subspace(U_target,subspace_vec=subspace_vec)
    
    ops_list = []
    d_list = []
    ld_list = []
    set_error = getattr(params,error[0])
    for err in ([0]+error[1]):
        set_error(err)
        ops,d,ld = calc_ops(params)
        params.reset_errors()
        ops_list.append(ops)
        d_list.append(d)
        ld_list.append(ld)
    
    constraint_function = lambda x:loss_fidelity(x,U_target,params,ops,d,ld)
    constraint_jacobian = lambda x:jacobian_fidelity(x,U_target,params,ops,d,ld)
    constraint = NonlinearConstraint(constraint_function,0,fidelity_bound,jac=constraint_jacobian)
    
    objective = RobustV2Target(M_stop)
    try:
        start = time.time()
        res = minimize(objective.loss,waveform_start,jac=objective.jacobian,constraints=constraint,args=(U_target,params,ops_list,d_list,ld_list),method='SLSQP',options={'maxiter':max_iter})
        
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
