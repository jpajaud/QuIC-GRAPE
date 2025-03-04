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

    # waveform is of shape (3*N_steps)
    # order of timesteps is latest phase first
    phases = waveform.reshape(params.NPAR,-1) # 3 is a constant
    cpx = np.cos(phases[0,:]).reshape(-1,1)
    spx = np.sin(phases[0,:]).reshape(-1,1)
    cpy = np.cos(phases[1,:]).reshape(-1,1)
    spy = np.sin(phases[1,:]).reshape(-1,1)
    cpmw = np.cos(phases[2,:]).reshape(-1,1)
    spmw = np.sin(phases[2,:]).reshape(-1,1)
    cp2x = np.cos(2*phases[0,:]).reshape(-1,1)
    cp2y = np.cos(2*phases[1,:]).reshape(-1,1)
    sp2x = np.sin(2*phases[0,:]).reshape(-1,1)
    sp2y = np.sin(2*phases[1,:]).reshape(-1,1)
    cpdt = np.cos( phases[0,:] - phases[1,:] ).reshape(-1,1)
    spdt = np.sin( phases[0,:] - phases[1,:] ).reshape(-1,1)


    # b is (STEPS,NOPS)
    b = np.hstack([np.ones((phases.shape[1],1)),
                    (params.rwa_order*params.rf_amp_x*params.rf_det*spx + params.rwa_order*params.rf_amp_y*params.rf_det*cpy + 2*params.rf_freq*(params.rf_amp_x*cpx + params.rf_amp_y*spy))/(4*params.rf_freq),
                   (-params.rwa_order*params.rf_amp_x*params.rf_det*cpx + params.rwa_order*params.rf_amp_y*params.rf_det*spy + 2*params.rf_freq*(-params.rf_amp_x*spx + params.rf_amp_y*cpy))/(4*params.rf_freq),
                    -params.rwa_order*(params.rf_amp_x**2*(2*cp2x - 1) - 2*params.rf_amp_x*params.rf_amp_y*spdt + params.rf_amp_y**2*(2*cp2y - 1))/(16*params.rf_freq),
                     params.grel*(params.rwa_order*params.rf_amp_x*params.rf_det*cpx - params.rwa_order*params.rf_amp_y*params.rf_det*spy - params.rf_amp_x*(params.rwa_order*params.rf_bias*(params.grel + 1) - 2*params.rf_freq)*cpx - params.rf_amp_y*(params.rwa_order*params.rf_bias*(params.grel + 1) + 2*params.rf_freq)*spy)/(4*params.rf_freq),
                     params.grel*(params.rwa_order*params.rf_amp_x*params.rf_det*spx - params.rwa_order*params.rf_amp_y*params.rf_det*cpy + params.rf_amp_x*(params.rwa_order*params.rf_bias*(params.grel + 1) + 2*params.rf_freq)*spx - params.rf_amp_y*(params.rwa_order*params.rf_bias*(params.grel + 1) - 2*params.rf_freq)*cpy)/(4*params.rf_freq),
                     params.rwa_order*params.grel**2*(params.rf_amp_x**2*(2*cp2x - 1) + 2*params.rf_amp_x*params.rf_amp_y*spdt + params.rf_amp_y**2*(2*cp2y - 1))/(16*params.rf_freq),
                     params.mw_amp*cpmw/2,
                    -params.mw_amp*spmw/2,
                    ])
    
    z = np.zeros((phases.shape[1],1))

    db_dx = np.hstack([z,
                       params.rf_amp_x*(params.rwa_order*params.rf_det*cpx - 2*params.rf_freq*spx)/(4*params.rf_freq),
                       params.rf_amp_x*(params.rwa_order*params.rf_det*spx - 2*params.rf_freq*cpx)/(4*params.rf_freq),
                       params.rwa_order*params.rf_amp_x*(2*params.rf_amp_x*sp2x + params.rf_amp_y*cpdt)/(8*params.rf_freq),
                       params.grel*params.rf_amp_x*(params.rwa_order*params.rf_bias*(params.grel + 1) - params.rwa_order*params.rf_det - 2*params.rf_freq)*spx/(4*params.rf_freq),
                       params.grel*params.rf_amp_x*(params.rwa_order*params.rf_bias*(params.grel + 1) + params.rwa_order*params.rf_det + 2*params.rf_freq)*cpx/(4*params.rf_freq),
                       params.rwa_order*params.grel**2*params.rf_amp_x*(-2*params.rf_amp_x*sp2x + params.rf_amp_y*cpdt)/(8*params.rf_freq),
                       z,
                       z,
                       ])
    
    db_dy = np.hstack([z,
                       params.rf_amp_y*(-params.rwa_order*params.rf_det*spy + 2*params.rf_freq*cpy)/(4*params.rf_freq),
                       params.rf_amp_y*(params.rwa_order*params.rf_det*cpy - 2*params.rf_freq*spy)/(4*params.rf_freq),
                       params.rwa_order*params.rf_amp_y*(-params.rf_amp_x*cpdt + 2*params.rf_amp_y*sp2y)/(8*params.rf_freq),
                      -params.grel*params.rf_amp_y*(params.rwa_order*params.rf_bias*(params.grel + 1) + params.rwa_order*params.rf_det + 2*params.rf_freq)*cpy/(4*params.rf_freq),
                       params.grel*params.rf_amp_y*(params.rwa_order*params.rf_bias*(params.grel + 1) + params.rwa_order*params.rf_det - 2*params.rf_freq)*spy/(4*params.rf_freq),
                      -params.rwa_order*params.grel**2*params.rf_amp_y*(params.rf_amp_x*cpdt + 2*params.rf_amp_y*sp2y)/(8*params.rf_freq),
                       z,
                       z,
                      ])
    
    db_dmw = np.hstack([z,
                        z,
                        z,
                        z,
                        z,
                        z,
                        z,
                       -params.mw_amp*spmw/2,
                       -params.mw_amp*cpmw/2,
                       ])
    
    db = np.stack((db_dx,db_dy,db_dmw),axis=1)

    return b,db

def calc_ops(params):
    """Calculates operators of QuIC B Hamiltonian

    eigenstate ordering
    0     ...   8  9    ... 15
    |4,-4⟩...|4,4⟩, |3,3⟩...|3,-3⟩,
    this ordering makes all operators tridiagonal

    Arguments
        params : dictionary of constants

    Return
        ops[9,DIM,DIM] : tridiagonal operators in order H0,fx4,fy4,fz4,fx3,fy3,fz3,mw_sx,mw_sy
        d[9,DIM]       : diagonal elements of ops
        ld[9,DIM-1]    : lower diagonal elements of ops
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

    jx_up,jy_up,jz_up = [direct_sum(j,params.DIM_DN) for j in ang_mom(spin=params.SPIN_UP,convention='Reversed')]   # reverse ordering to align with eigenstate ordering convention
    jx_dn,jy_dn,jz_dn = [direct_sum(params.DIM_UP,j) for j in ang_mom(spin=params.SPIN_DN,convention='Standard')]
    
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
    # 1,fx4
    # 2,fy4
    # 3,fz4
    # 4,fx3
    # 5,fy3
    # 6,fz3
    # 7,mw_sx
    # 8,mw_sy

    N_OPS = 9

    ops = np.zeros((N_OPS,params.DIM,params.DIM),dtype=np.complex128)
    ops[0,:,:] = H0
    ops[1,:,:] = jx_up
    ops[2,:,:] = jy_up
    ops[3,:,:] = jz_up
    ops[4,:,:] = jx_dn
    ops[5,:,:] = jy_dn
    ops[6,:,:] = jz_dn
    ops[7,:,:] = mw_x
    ops[8,:,:] = mw_y

    d  = np.zeros((N_OPS,params.DIM),dtype=np.float64)
    ld = np.zeros((N_OPS,params.DIM-1),dtype=np.complex128)

    # separating diagonal and lower diagonal makes calculations faster
    for i in range(N_OPS):
        d[i,:] = np.diag(ops[i,:,:]).real
        ld[i,:] = np.diag(ops[i,:,:],-1)

    return ops, d, ld
