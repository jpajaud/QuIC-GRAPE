import numpy as np
from grape.calc import clebsch_gordan, ang_mom, direct_sum

def calc_b(waveform,params):
    """Hamiltonian at step s is of the form
        H[s,:,:] = ∑_l ops[l,:,:]*b[l,s]

    Arguments
        waveform[NPAR*STEPS,]    : phases ordered as [x_end... x_start,y_end... y_start, mw_end... mw_start]
                                   reverse order is to simplify calculation of gradient
        params                   : dictionary of constants
    
    Return
        b[STEPS,NOPS]            : coefficients of Hamiltonian equation
        db[STEPS,NPAR,NOPS]      : derivative of coefficients with respect to all phases
        
    """

    # waveform is of shape (N_steps,)
    # order of timesteps is latest phase first
    phases = waveform.reshape(params.NPAR,-1) # 1 is a constant
    cpmw = np.cos(phases[0,:]).reshape(-1,1)
    spmw = np.sin(phases[0,:]).reshape(-1,1)


    # b is (STEPS,NOPS)
    b = np.hstack([np.ones((phases.shape[1],1)),
                   params.mw_amp*cpmw/2,
                  -params.mw_amp*spmw/2,
                  ])
    
    db = np.hstack([np.zeros((phases.shape[1],1)),
                   -params.mw_amp*spmw/2,
                   -params.mw_amp*cpmw/2,
                   ])
    
    # need to reshape because NPAR is 1
    return b,db[:,None,:]

def calc_ops(params):
    """Calculates operators of QuIC B Hamiltonian

    eigenstate ordering
    0     ...   8  9    ... 15
    |4,-4⟩...|4,4⟩, |3,3⟩...|3,-3⟩,
    this ordering makes all operators tridiagonal

    Arguments
        params : dictionary of constants

    Return
        ops[3,DIM,DIM] : tridiagonal operators in order H0,fx4,fy4,fz4,fx3,fy3,fz3,mw_sx,mw_sy
        d[3,DIM]       : diagonal elements of ops
        ld[3,DIM-1]    : lower diagonal elements of ops
    """

    ACZ = np.zeros((params.DIM,params.DIM),dtype=np.complex128)
    for m in range(2,-4,-1):
        # iterate over all m ≠ 3

        V3m  = np.zeros(params.DIM,dtype=np.complex128) # projector onto |3,m⟩   index is 12-m
        V3m[11-m] = 1 # this needs to be flipped
        V4m1 = np.zeros(params.DIM,dtype=np.complex128) # projector onto |4,m+1⟩ index is m+4
        V4m1[6+m] = 1
        ACZ += (np.diag(V3m-V4m1))*(clebsch_gordan(3,1,4,m,1,m+1)**2)/(3-m) #minus sign corrected 4 June 2012
    

    ACZ *= (params.mw_amp**2)/(8*params.rf_bias)

    # ACZ = np.diag(ACZ)

    _,_,jz_up = [direct_sum(j,params.DIM_DN) for j in ang_mom(spin=params.SPIN_UP,convention='Reversed')]   # reverse ordering to align with eigenstate ordering convention
    _,_,jz_dn = [direct_sum(params.DIM_UP,j) for j in ang_mom(spin=params.SPIN_DN,convention='Standard')]
    
    proj_up = direct_sum(np.eye(params.DIM_UP),params.DIM_DN,dtype=np.complex128)
    proj_dn = direct_sum(params.DIM_UP,np.eye(params.DIM_DN),dtype=np.complex128)

    mw_x = np.zeros((params.DIM,params.DIM),dtype=np.complex128)
    mw_y = np.zeros((params.DIM,params.DIM),dtype=np.complex128)

    mw_x[params.I44,params.I33] = 1
    mw_x[params.I33,params.I44] = 1
    mw_y[params.I44,params.I33] = -1j
    mw_y[params.I33,params.I44] = 1j

    H0 =  (((3/2)*params.rf_bias*(1+params.grel))-((25/2)*params.grel*(params.rf_bias**2)*(1/params.hf_freq))-((1/2)*(params.mw_det-7*params.rf_det))) * (proj_up-proj_dn)\
         +(params.rf_bias*(1+params.grel)*jz_dn)\
         +(params.grel*(params.rf_bias**2)*(1/params.hf_freq)*(jz_up@jz_up-jz_dn@jz_dn))\
         +((-1*params.rf_det*(jz_up+params.grel*jz_dn)))\
         +ACZ

    # ordering of operators
    # 0,H0 # bias field
    # 1,mw_sx
    # 2,mw_sy

    N_OPS = 3

    ops = np.zeros((N_OPS,params.DIM,params.DIM),dtype=np.complex128)
    ops[0,:,:] = H0
    ops[1,:,:] = mw_x
    ops[2,:,:] = mw_y

    d  = np.zeros((N_OPS,params.DIM),dtype=np.float64)
    ld = np.zeros((N_OPS,params.DIM-1),dtype=np.complex128)

    # separating diagonal and lower diagonal makes calculations faster
    for i in range(N_OPS):
        d[i,:] = np.diag(ops[i,:,:]).real
        ld[i,:] = np.diag(ops[i,:,:],-1)

    return ops, d, ld
