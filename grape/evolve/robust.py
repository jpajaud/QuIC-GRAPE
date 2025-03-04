import numpy as np
from grape.loss import loss_fidelity
from grape.calc import calc_ops

def test_robustness(waveform,U_target,params,error=None):
    
    if error is None:
        # error = params.default_error
        error = ('set_detuning',[-100,100])

    params.reset_errors()
    ops,d,ld = calc_ops(params)

    set_error = getattr(params,error[0])

    N_samp = 20
    Δs = np.linspace(error[1][0],error[1][1],N_samp)
    fidelity_loss = np.zeros(N_samp)
    for i in range(N_samp):
        set_error(Δs[i])
        ops,d,ld = calc_ops(params)
        params.reset_errors()
        fidelity_loss[i] = loss_fidelity(waveform,U_target,params,ops,d,ld)

    return Δs,fidelity_loss