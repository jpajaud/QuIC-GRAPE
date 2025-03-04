from copy import deepcopy
from typing import Union
import numpy as np
from grape.calc import calc_U_tri_single, calc_ops, accumulate_single
from grape import WaveformParams

def simulate(state0:np.ndarray,U_step:np.ndarray,steps:int)->np.ndarray:
    DIM = state0.shape[0]
    states = np.zeros((DIM,steps),dtype=np.complex128)
    states[:,0] = state0.ravel()
    for i in range(1,steps):
        states[:,i] = U_step@states[:,i-1]
    return states

def expectation(state,meas_ops):
    # state is shape (DIM,STEPS)
    # meas_ops is shape (NOPS,DIM,DIM)
    return np.einsum('is,nij,js->ns',state.conj(),meas_ops,state).real

def evolve(state:np.ndarray,meas_ops:np.ndarray,waveform:np.ndarray,params:WaveformParams,evolver_errors:Union[dict,None]=None,NRUNS=60,NSTEPS=100)->np.ndarray:
    # meas_ops is of shape (NOPS,DIM,DIM) first index allows stacking of multiple operators

    default_errors = deepcopy(params.default_errors)

    if evolver_errors is None:
        evolver_errors = default_errors
    
    if set(evolver_errors.keys())!=set(default_errors.keys()):
        # allow for evolver_errors to contain only some keys
        default_errors.update(evolver_errors)
        evolver_errors = default_errors


    NOPS,DIM,_ = meas_ops.shape

    fidelity = np.zeros((NRUNS,NSTEPS))

    # exact simulation
    params.reset_errors()
    ops,d,ld = calc_ops(params)
    u_s = calc_U_tri_single(waveform,params,ops,d,ld)
    U_exact = accumulate_single(u_s)
    states_exact = simulate(state,U_exact,NSTEPS)
    meas_exact = expectation(states_exact,meas_ops)
    states_exact_c = states_exact.conj()

    meas_exper = np.zeros((NRUNS,NOPS,NSTEPS))
    for i in range(NRUNS):
        
        new_waveform = params.mutate_self_and_waveform(waveform,evolver_errors)
        ops,d,ld = calc_ops(params)
        u_s = calc_U_tri_single(new_waveform,params,ops,d,ld)
        U_step = accumulate_single(u_s)
        states_exper = simulate(state,U_step,NSTEPS)

        fidelity[i,:] = np.abs(np.sum(states_exact_c*states_exper,axis=0))
        meas_exper[i,:,:] = expectation(states_exper,meas_ops)


    return fidelity.mean(axis=0), meas_exact, meas_exper.mean(axis=0)

def evolve_H(state:np.ndarray,H:np.ndarray,meas_ops:np.ndarray,waveform:np.ndarray,params:WaveformParams,evolver_errors:Union[dict,None]=None,NRUNS=60,NSTEPS=100)->np.ndarray:
    # meas_ops is of shape (NOPS,DIM,DIM) first index allows stacking of multiple operators

    default_errors = deepcopy(params.default_errors)

    if evolver_errors is None:
        evolver_errors = default_errors
    
    if set(evolver_errors.keys())!=set(default_errors.keys()):
        # allow for evolver_errors to contain only some keys
        default_errors.update(evolver_errors)
        evolver_errors = default_errors


    NOPS,DIM,_ = meas_ops.shape

    # exact simulation
    params.reset_errors()
    ops,d,ld = calc_ops(params)
    u_s = calc_U_tri_single(waveform,params,ops,d,ld)
    U_exact = accumulate_single(u_s)
    states_exact = simulate(state,U_exact,NSTEPS)
    meas_exact = expectation(states_exact,meas_ops)
    states_exact_c = states_exact.conj()
        
    # new_waveform = params.mutate_self_and_waveform(waveform,evolver_errors)
    # ops,d,ld = calc_ops(params)
    ops[0,:,:] += H/params.DT # the DT will get added back in in the next line
    d = np.diagonal(ops,axis1=1,axis2=2).real
    ld = np.diagonal(ops,-1,axis1=1,axis2=2)
    u_s = calc_U_tri_single(waveform,params,ops,d,ld)
    U_step = accumulate_single(u_s)
    states_exper = simulate(state,U_step,NSTEPS)

    fidelity = np.abs(np.sum(states_exact_c*states_exper,axis=0))
    meas_exper = expectation(states_exper,meas_ops)


    return fidelity, meas_exact, meas_exper