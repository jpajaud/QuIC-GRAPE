import numpy as np
from scipy.linalg import eigh_tridiagonal
from grape.calc import calc_b, calc_hessenberg

def step_processor(Hd,Hld,R_hess,db,ops,dt):
    """Calculate unitary quantities and derivative for a single phase step
    
    Arguments
        Hd[DIM,]            : diagonal elements of Hamiltonian
        Hld[DIM-1,]         : lower diagonal elements of Hamiltonian
        R_hess[DIM,]        : phases of transformation from Hessenberg form
        db[NPAR,NOPS]       : derivative of Hamiltonian expansion with respect to phases
        ops[NOPS,DIM,DIM]   : operators in Hamiltonian expansion
        dt int              : timestep of each phase step

    Return
        U[DIM,DIM]          : Unitary
        dU[NPAR,DIM,DIM]    : derivative of Unitary with respect to phases
        
    """

    w,v = eigh_tridiagonal(Hd,np.abs(Hld))
    v_true = np.diag(R_hess)@v # converting back from Hessenberg form
    v_true_c = v_true.T.conj().copy()

    DIM, = Hd.shape
    NPAR,NOPS = db.shape

    wexp = np.exp(-1j*w*dt)
    U = np.diag(wexp)
    num  = np.subtract.outer(wexp,wexp)
    den  = np.subtract.outer(w,w) + np.eye(DIM)
    fact = num/den + np.diag(-1j*dt*wexp)

    dU = np.zeros((NPAR,DIM,DIM),dtype=np.complex128)
    for l in range(NOPS):
        ops_l = (v_true_c@ops[l,:,:]@v_true)*fact
        for p in range(NPAR):
            dU[p,:,:] += db[p,l]*ops_l
            # dU_dx  += db_dx[l] *ops_l * fact

    U      = v_true@U     @v_true_c
    for p in range(NPAR):
        dU[p,:,:] = v_true@dU[p,:,:]@v_true_c        
    
    return U,dU

def step_processor_single(Hd,Hld,R_hess,db,ops,dt):
    """Calculate unitary quantities without derivative for a single phase step
    
    Arguments
        Hd[DIM,]            : diagonal elements of Hamiltonian
        Hld[DIM-1,]         : lower diagonal elements of Hamiltonian
        R_hess[DIM,]        : phases of transformation from Hessenberg form
        db[NPAR,NOPS]       : derivative of Hamiltonian expansion with respect to phases
        ops[NOPS,DIM,DIM]   : operators in Hamiltonian expansion
        dt int              : timestep of each phase step

    Return
        U[DIM,DIM]          : Unitary
        dU[NPAR,DIM,DIM]    : derivative of Unitary with respect to phases
        
    """

    w,v = eigh_tridiagonal(Hd,np.abs(Hld))
    v_true = np.diag(R_hess)@v # converting back from Hessenberg form
    
    wexp = np.exp(-1j*w*dt)
    U = np.diag(wexp)
    U = v_true@U@v_true.T.conj()
    return U


def calc_U_tri(waveform,params,ops,d,ld):
    """Calculates waveform step unitaries and relevant derivatives
    
    Arguments
        waveform[NPAR*STEPS]       : phases stored in reverse time order, x then y then mw in flattened vector
        params dict                : dictionary of constants
        ops[NOPS,:,:]              : operators of Hamiltonian expansion
        d[NOPS,DIM]                : diagonal elements of ops
        ld[NOPS,DIM-1]             : lower diagonal elements of ops

    Return
        u_s[DIM,DIM]             : unitary at each waveform step
        du_s[NPAR,STEPS,DIM,DIM]  : derivative of unitary at waveform step with respect to phase at that step
    """
    
    b,db = calc_b(waveform,params)
    
    Hd       = b@d   # diagonal        [STEPS,DIM]
    Hld      = b@ld  # lower diagonal  [STEPS,DIM-1]

    # calculate hessenberg diagonal elements
    R_hess = calc_hessenberg(Hld) # transformation from real to complex [STEPS,DIM]

    u_s    = np.zeros((params.STEPS,params.DIM,params.DIM),dtype=np.complex128)
    du_s   = np.zeros((params.STEPS,params.NPAR,params.DIM,params.DIM),dtype=np.complex128)
    for i in range(params.STEPS):
        u_s[i,:,:],du_s[i,:,:,:] = step_processor(Hd[i,:],Hld[i,:],R_hess[i,:],db[i,:,:],ops,params.DT)
    
    return u_s,du_s

def calc_U_tri_single(waveform,params,ops,d,ld):
    """Calculates waveform step unitaries
    
    Arguments
        waveform[NPAR*STEPS] : phases stored in reverse time order, x then y then mw in flattened vector
        params dict          : dictionary of constants
        ops[NOPS,:,:]        : operators of Hamiltonian expansion
        d[NOPS,DIM]          : diagonal elements of ops
        ld[NOPS,DIM-1]       : lower diagonal elements of ops

    Return
        u_s[DIM,DIM]         : unitary at each waveform step
    """
    
    b,db = calc_b(waveform,params)
    
    Hd       = b@d   # diagonal        [STEPS,DIM]
    Hld      = b@ld  # lower diagonal  [STEPS,DIM-1]

    # calculate hessenberg diagonal elements
    R_hess = calc_hessenberg(Hld) # transformation from real to complex [STEPS,DIM]

    u_s    = np.zeros((params.STEPS,params.DIM,params.DIM),dtype=np.complex128)
    for i in range(params.STEPS):
        u_s[i,:,:] = step_processor_single(Hd[i,:],Hld[i,:],R_hess[i,:],db[i,:,:],ops,params.DT)
    
    return u_s


def accumulate(u_s, du_s):
    """Calculates cumulative unitaries and derivative of final unitary with respect to each phase

    Arguments
        u_s[STEPS,DIM,DIM]        : unitary at each waveform step
        du_s[STEPS,NPAR,DIM,DIM]  : derivative of unitary at each waveform step respect to phase at that step
    
    Return
        U[DIM,DIM]                : final unitary
        dU_s[STEPS,NPAR,DIM,DIM]  : derivative of final unitary with respect to each phase
        U_f[STEPS,DIM,DIM]        : Cumulative unitary at each step in a waveform
                                    U_f[i] = U[i]@U[i-1]@...@U[1]@U[0]
        U_b_conj[STEPS+1,DIM,DIM] : Cumulative unitary defined in reverse
                                    U_b_conj[-1] = np.eye(DIM)
                                    U_b_conj[-i] = I@U[STEPS]@U[STEPS-1]@...@U[1-i]
    """
    # u_s is unitary at each step, shape (STEPS,DIM,DIM)
    # du_s is derivative of unitary at step s by phase n, shape (STEPS,NPAR,DIM,DIM)
    
    # STEPS,NPAR,DIM,_ = du_s.shape
    STEPS = du_s.shape[0]
    DIM = du_s.shape[2]

    U = np.eye(DIM,dtype=np.complex128)
    U_f = np.zeros((STEPS,DIM,DIM),dtype=np.complex128) # consider renaming to uU


    for i in range(STEPS):
        U[:,:] = u_s[i,:,:]@U
        U_f[i,:,:] = U[:,:]

    U = np.eye(DIM,dtype=np.complex128)
    U_b = np.zeros((STEPS+1,DIM,DIM),dtype=np.complex128) # consider renaming to Uu

    U_b[-1,:,:] = U[:,:]
    for i in range(STEPS-1,-1,-1):
        U = U@u_s[i,:,:]
        U_b[i,:,:] = U[:,:]

    # U is now U_final
    U_b_conj = np.transpose(U_b,axes=(0,2,1)).conj()
    dU_s = np.zeros_like(du_s) # shape is (STEPS,NPAR,DIM,DIM)

    # y@x pattern
    dU_s[0,:,:,:] = np.tensordot(du_s[0,:,:,:],U_b[1,:,:],axes=(1,1)).transpose((0,2,1))   # U_b[1,:,:]@du_s[0,:,:,:]
    for i in range(1,STEPS):
        # x@y  x=du_s,y=U_b_conj@U pattern followed by y@x x=res,y=U_b[i+1,:,:]
        dU_s[i,:,:,:] = np.tensordot(np.tensordot(du_s[i,:,:,:],U_b_conj[i,:,:]@U,axes=(2,0)),U_b[i+1,:,:],axes=(1,1)).transpose((0,2,1))

        # dU_s[i,j,:,:] = U_b[i+1,:,:]@du_s[i]@U_b_conj[i,:,:]@U
 
    # U        : final unitary                      (DIM,DIM)
    # dU_s     : derivative of final v phase/step   (STEPS,NPAR,DIM,DIM)
    # U_f      : forward propagation                (STEPS,DIM,DIM)
    # U_b_conj : conjugate of backward propagation  (STEPS+1,DIM,DIM)
    return U, dU_s, U_f, U_b_conj

def accumulate_single(u_s):
    """Calculates final unitary

    Arguments
        u_s[STEPS,DIM,DIM] : unitary at each waveform step
        
    Return
        U[DIM,DIM]         : final unitary
    """
    # u_s is unitary at each step, shape (STEPS,DIM,DIM)
    # du_s is derivative of unitary at step s by phase n, shape (STEPS,NPAR,DIM,DIM)
    
    # STEPS,NPAR,DIM,_ = du_s.shape
    STEPS,DIM,_ = u_s.shape

    U = np.eye(DIM,dtype=np.complex128)

    for i in range(STEPS):
        U[:,:] = u_s[i,:,:]@U

    return U


