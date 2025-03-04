import copy
import numpy as np
from typing import Union
from grape.calc import clebsch_gordan, ang_mom, direct_sum

PARAMS_KEY = 'params_data'

def make_mask_pair(subspace_vector:np.ndarray):
    # subspace_vector is 1 for elements that are in subspace
    # example  [1, 1, 0, 0], both first and second element are in subspace
    
    # mask = kc*np.outer(c,c) + kd*np.outer(d,d)
    # mask = [ [1, 1, 1, 1],  
    #          [1, 1, 1, 1],
    #          [1, 1, 0, 0],
    #          [1, 1, 0, 0]  ]

    # final result is np.sum(np.outer(x.conj(),x)*mask)
    # 
    # csum = np.dot(x,c); dsum = np.dot(x,d)
    # final result = kc*csum*csum.conj() + kd*dsum*dsum.conj()
    # or 
    # final result is np.sum(np.outer(x.conj(),y)+np.outer(y.conj(),x))*mask)
    #
    # cxsum = np.dot(x,c); cysum = np.dot(y,c)
    # dxsum = np.dot(x,d); dysum = np.dot(y,c)
    # final result = 2*np.real(kc*cxsum.conj()*cysum + kd*dxsum.conj()*dysum)

    s = sum(subspace_vector)
    D = len(subspace_vector)
    A = np.sqrt(s**2 + 4*s*(D-s))
    B = np.sqrt(1/s**2 - 4*(D-s)/(s*A*A))

    kc = (s+A)/2
    kd = (s-A)/2

    # constant for elements in subspace
    cin = np.sqrt((1/s+B)/2)
    din = np.sqrt((1/s-B)/2) 

    # constant for elements out of subspace
    cout = 1/(A*cin)
    dout = -1/(A*din)

    c = np.zeros(D,dtype=np.float64)
    c[subspace_vector==1] = cin
    c[subspace_vector==0] = cout

    d = np.zeros(D,dtype=np.float64)
    d[subspace_vector==1] = din
    d[subspace_vector==0] = dout

    ks = np.array([kc,kd])
    vs = np.vstack((c,d))
    return ks,vs

# class Environment/ System/ Architecture/ Hamiltonian/ Context/ Computer
class WaveformParams:
    __slots__      : tuple = ('rf_det','rf_bias','rf_amp_x','rf_amp_y','mw_amp','mw_det','phase_error')
    SPIN_UP        : float
    DIM_UP         : int
    SPIN_DN        : float
    DIM_DN         : int
    DIM            : int
    SUBSPACE_DIM   : int
    subspace_vec   : np.ndarray
    subspace_type  : str
    DT             : float
    STEPS          : int
    stop_iter      : float
    I44            : int
    I33            : int
    rf_det_error   : float
    mw_amp_error   : float
    grel           : float
    hf_freq        : float
    mw_amp         : float
    mw_amp_nom     : float
    mw_det         : float
    rf_det         : float
    rf_bias        : float
    rwa_order      : int
    rf_amp_x       : float
    rf_amp_x_nom   : float
    rf_amp_y       : float
    rf_amp_y_nom   : float
    rf_freq        : float
    init_state     : np.ndarray
    phase_error    : float
    default_errors : dict
    NOPS           : int
    NPAR           : int
    ks             : np.ndarray
    vs             : np.ndarray
    save_dict      : dict
    params_key     : str

    def __new__(cls,*,SPIN_UP,DIM_UP,SPIN_DN,DIM_DN,DIM,DT,STEPS,stop_iter,I44,I33,rf_det_error,mw_amp_error,grel,hf_freq,mw_amp,mw_det,rf_det,rf_bias,rwa_order,rf_amp_x,rf_amp_y,rf_freq,save_dict,params_key,subspace_vec=None,subspace_type='isolated'):
        cls.SPIN_UP       = SPIN_UP
        cls.DIM_UP        = DIM_UP
        cls.SPIN_DN       = SPIN_DN
        cls.DIM_DN        = DIM_DN
        cls.DIM           = DIM
        cls.NOPS          = 9
        cls.NPAR          = 3
        cls.SUBSPACE_DIM  = (DIM if subspace_vec is None else int(sum(subspace_vec)))
        if subspace_vec is None:
            ks = np.array([1.])
            vs = np.ones((1,DIM),dtype=np.float64)
        else:
            if subspace_type == 'isolated':
                ks = np.array([1.])
                vs = subspace_vec.astype(np.float64).reshape(1,-1)
            elif subspace_type == 'connected':
                ks,vs = make_mask_pair(subspace_vec)
        cls.subspace_vec  = subspace_vec
        cls.subspace_type = subspace_type
        cls.ks            = ks
        cls.vs            = vs
        cls.DT            = DT
        cls.STEPS         = STEPS
        cls.stop_iter     = stop_iter
        cls.I44           = I44
        cls.I33           = I33
        cls.rf_det_error  = rf_det_error
        cls.mw_amp_error  = mw_amp_error
        cls.grel          = grel
        cls.hf_freq       = hf_freq
        cls.mw_amp_nom    = mw_amp
        cls.rwa_order     = rwa_order
        cls.rf_amp_x_nom  = rf_amp_x
        cls.rf_amp_y_nom  = rf_amp_y
        cls.rf_freq       = rf_freq
        cls.save_dict     = save_dict
        cls.params_key    = params_key
        cls.init_state    = np.array([1. if i==I33 else 0. for i in range(DIM)],dtype=np.complex128).reshape(-1,1)

        # for evolver
        cls.default_errors = {'rf_det_err':100,
                              'rf_amp_x_err':.004,
                              'rf_amp_y_err':.004,
                              'mw_amp_err':.008,
                              'phase_err':.1*np.pi/180}
    

        return super(WaveformParams,cls).__new__(cls)

    def __init__(self,*,SPIN_UP,DIM_UP,SPIN_DN,DIM_DN,DIM,DT,STEPS,stop_iter,I44,I33,rf_det_error,mw_amp_error,grel,hf_freq,mw_amp,mw_det,rf_det,rf_bias,rwa_order,rf_amp_x,rf_amp_y,rf_freq,save_dict,params_key,subspace_vec=None,subspace_type='isolated'):
        self.mw_det       = mw_det
        self.mw_amp       = mw_amp
        self.rf_det       = rf_det
        self.rf_bias      = rf_bias
        self.rf_amp_x     = rf_amp_x
        self.rf_amp_y     = rf_amp_y
        self.phase_error  = 0

    @staticmethod
    def prepare_params(*,STEPS:int=150,DT:float=4e-6,stop_iter:float=1e-3,subspace_vec:Union[None,list[int]]=None,subspace_type:str='isolated'):
        """Prepare dictionary of constants for use in other functions
        
        Arguments
            STEPS int   = 150  : number of timesteps
            DT    float = 4e-6 : duration of timestep in seconds
        
        Return
            params : dictionary of constants
        """
        # save input arguments for easy loading from file
        
        save_dict = {PARAMS_KEY:{'STEPS'        :STEPS,
                                 'DT'           :DT,
                                 'stop_iter'    :stop_iter,
                                 'subspace_vec' :subspace_vec,
                                 'subspace_type':subspace_type}}
        # call data.update(params.save_dict)
        # before calling np.savez(filename,**data)

        SPIN_UP = 4
        SPIN_DN = 3
        DIM_UP  = 2*SPIN_UP+1
        DIM_DN  = 2*SPIN_DN+1
        DIM     = DIM_UP+DIM_DN

        rwa_order = 1

        # moved to arguments
        # DT = 4e-6   # duration of phase step
        # STEPS = 150 # number of phase steps in waveform

        mw_amp = 27.5e3 * (2*np.pi) # rabi freq for for stretched state transition in Hz
        rf_freq = 1e6 * (2*np.pi)  # rf freq in Hz
        rf_amp_x = 25e3 * (2*np.pi) # rf larmor frequency x in Hz
        rf_amp_y = 25e3 * (2*np.pi) # rf larmor frequency y in Hz

        rf_det_error = 40 # inhomo grid param for rf detuning in Hz
        mw_amp_error = 0  # inhomo grid param for mw amp in Hz

        rf_det = 0 # constant rf detuning

        rf_bias = rf_freq - rf_det
        grel = -1.0032
        hf_freq = 9.19263e9 * (2*np.pi)
        freq_44_33_0det = hf_freq - (7*grel*rf_freq**2*(1/hf_freq)) + ((4-3*grel)*rf_freq)
        mw_det = freq_44_33_0det - ( hf_freq - (7*grel*rf_bias**2*(1/hf_freq)) + ((4-3*grel)*rf_bias) )

        I44 = 8
        I33 = 9

        params_dict = {'SPIN_UP':SPIN_UP,
                       'DIM_UP':DIM_UP,
                       'SPIN_DN':SPIN_DN,
                       'DIM_DN':DIM_DN,
                       'DIM':DIM,
                       'subspace_vec':subspace_vec,
                       'subspace_type':subspace_type,
                       'DT':DT,
                       'STEPS':STEPS,
                       'stop_iter':stop_iter,
                       'I44':I44,
                       'I33':I33,
                       'rf_det_error':rf_det_error,
                       'mw_amp_error':mw_amp_error,
                       'grel':grel,
                       'hf_freq':hf_freq,
                       'mw_amp':mw_amp,
                       'mw_det':mw_det,
                       'rf_det':rf_det,
                       'rf_bias':rf_bias,
                       'rwa_order':rwa_order,
                       'rf_amp_x':rf_amp_x,
                       'rf_amp_y':rf_amp_y,
                       'rf_freq':rf_freq,
                       'save_dict':save_dict,
                       'params_key':PARAMS_KEY}
        params = WaveformParams(**params_dict)
        return params

    def load_from_npz(data_dict):
        if isinstance(data_dict,dict):
            return WaveformParams.prepare_params(**data_dict)
        else:
            
            try:
                # assume it is a numpy npz file
                data = data_dict[PARAMS_KEY]
                data = data.item() # must convert back from zero-d array
                return WaveformParams.prepare_params(**data)
            except ValueError as e:
                if 'pickle' in e.args[0]: # failed to load object when allow_pickle==False
                    # try to reload with 
                    fname = data_dict.zip.filename
                    data_dict = np.load(fname,allow_pickle=True)
                    data = data_dict[PARAMS_KEY]
                    data = data.item() # must convert back from zero-d array
                    return WaveformParams.prepare_params(**data)
                else:
                    raise e

    def mutate_self_and_waveform(self,waveform:np.ndarray,evolver_errors:Union[None,dict]=None) -> np.ndarray:
        # can try to assert that waveforms is list of np.ndarray
        errors = copy.deepcopy(self.default_errors)
        if evolver_errors is not None:
            errors.update(evolver_errors)

        rand_normal = np.random.randn(5)

        self.set_detuning(rand_normal[0]*errors['rf_det_err'])
        self.set_rf_amplitude_scale(x=1+rand_normal[1]*errors['rf_amp_x_err'],
                                    y=1+rand_normal[2]*errors['rf_amp_y_err'])
        self.set_mw_amplitude_scale(1+rand_normal[3]*errors['mw_amp_err'])

        waveform3d = waveform.reshape(3,-1)
        waveform3d[1,:] += rand_normal[4]*errors['phase_err']
        return waveform3d.flatten()

    def reset_errors(self):
        self.set_detuning(0)
        self.set_rf_amplitude_scale(x=1,y=1)
        self.set_mw_amplitude_scale(1)

    def set_detuning(self,rf_det:float) -> None:
        self.rf_det = rf_det
        self.rf_bias = self.rf_freq - self.rf_det
        freq_44_33_0det               = ( self.hf_freq - (7*self.grel*self.rf_freq**2*(1/self.hf_freq)) + ((4-3*self.grel)*self.rf_freq) )
        self.mw_det = freq_44_33_0det - ( self.hf_freq - (7*self.grel*self.rf_bias**2*(1/self.hf_freq)) + ((4-3*self.grel)*self.rf_bias) )
        
    def set_rf_amplitude_scale(self,*,x:float,y:float) -> None:
        self.rf_amp_x = x*self.rf_amp_x_nom
        self.rf_amp_y = y*self.rf_amp_y_nom

    def set_mw_amplitude_scale(self,scale:float) -> None:
        self.mw_amp = scale*self.mw_amp_nom
        
    def calc_b(self,waveform):
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
        phases = waveform.reshape(self.NPAR,-1) # 3 is a constant
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
                        (self.rwa_order*self.rf_amp_x*self.rf_det*spx + self.rwa_order*self.rf_amp_y*self.rf_det*cpy + 2*self.rf_freq*(self.rf_amp_x*cpx + self.rf_amp_y*spy))/(4*self.rf_freq),
                    (-self.rwa_order*self.rf_amp_x*self.rf_det*cpx + self.rwa_order*self.rf_amp_y*self.rf_det*spy + 2*self.rf_freq*(-self.rf_amp_x*spx + self.rf_amp_y*cpy))/(4*self.rf_freq),
                        -self.rwa_order*(self.rf_amp_x**2*(2*cp2x - 1) - 2*self.rf_amp_x*self.rf_amp_y*spdt + self.rf_amp_y**2*(2*cp2y - 1))/(16*self.rf_freq),
                        self.grel*(self.rwa_order*self.rf_amp_x*self.rf_det*cpx - self.rwa_order*self.rf_amp_y*self.rf_det*spy - self.rf_amp_x*(self.rwa_order*self.rf_bias*(self.grel + 1) - 2*self.rf_freq)*cpx - self.rf_amp_y*(self.rwa_order*self.rf_bias*(self.grel + 1) + 2*self.rf_freq)*spy)/(4*self.rf_freq),
                        self.grel*(self.rwa_order*self.rf_amp_x*self.rf_det*spx - self.rwa_order*self.rf_amp_y*self.rf_det*cpy + self.rf_amp_x*(self.rwa_order*self.rf_bias*(self.grel + 1) + 2*self.rf_freq)*spx - self.rf_amp_y*(self.rwa_order*self.rf_bias*(self.grel + 1) - 2*self.rf_freq)*cpy)/(4*self.rf_freq),
                        self.rwa_order*self.grel**2*(self.rf_amp_x**2*(2*cp2x - 1) + 2*self.rf_amp_x*self.rf_amp_y*spdt + self.rf_amp_y**2*(2*cp2y - 1))/(16*self.rf_freq),
                        self.mw_amp*cpmw/2,
                        -self.mw_amp*spmw/2,
                        ])
        
        z = np.zeros((phases.shape[1],1))

        db_dx = np.hstack([z,
                        self.rf_amp_x*(self.rwa_order*self.rf_det*cpx - 2*self.rf_freq*spx)/(4*self.rf_freq),
                        self.rf_amp_x*(self.rwa_order*self.rf_det*spx - 2*self.rf_freq*cpx)/(4*self.rf_freq),
                        self.rwa_order*self.rf_amp_x*(2*self.rf_amp_x*sp2x + self.rf_amp_y*cpdt)/(8*self.rf_freq),
                        self.grel*self.rf_amp_x*(self.rwa_order*self.rf_bias*(self.grel + 1) - self.rwa_order*self.rf_det - 2*self.rf_freq)*spx/(4*self.rf_freq),
                        self.grel*self.rf_amp_x*(self.rwa_order*self.rf_bias*(self.grel + 1) + self.rwa_order*self.rf_det + 2*self.rf_freq)*cpx/(4*self.rf_freq),
                        self.rwa_order*self.grel**2*self.rf_amp_x*(-2*self.rf_amp_x*sp2x + self.rf_amp_y*cpdt)/(8*self.rf_freq),
                        z,
                        z,
                        ])
        
        db_dy = np.hstack([z,
                        self.rf_amp_y*(-self.rwa_order*self.rf_det*spy + 2*self.rf_freq*cpy)/(4*self.rf_freq),
                        self.rf_amp_y*(self.rwa_order*self.rf_det*cpy - 2*self.rf_freq*spy)/(4*self.rf_freq),
                        self.rwa_order*self.rf_amp_y*(-self.rf_amp_x*cpdt + 2*self.rf_amp_y*sp2y)/(8*self.rf_freq),
                        -self.grel*self.rf_amp_y*(self.rwa_order*self.rf_bias*(self.grel + 1) + self.rwa_order*self.rf_det + 2*self.rf_freq)*cpy/(4*self.rf_freq),
                        self.grel*self.rf_amp_y*(self.rwa_order*self.rf_bias*(self.grel + 1) + self.rwa_order*self.rf_det - 2*self.rf_freq)*spy/(4*self.rf_freq),
                        -self.rwa_order*self.grel**2*self.rf_amp_y*(self.rf_amp_x*cpdt + 2*self.rf_amp_y*sp2y)/(8*self.rf_freq),
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
                        -self.mw_amp*spmw/2,
                        -self.mw_amp*cpmw/2,
                        ])
        
        db = np.stack((db_dx,db_dy,db_dmw),axis=1)

        return b,db

    def calc_ops(self):
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

        ACZ = np.zeros((self.DIM,self.DIM),dtype=np.complex128)
        for m in range(2,-4,-1):
            # iterate over all m ≠ 3

            V3m  = np.zeros(self.DIM,dtype=np.complex128) # projector onto |3,m⟩   index is 12-m
            V3m[11-m] = 1 # this needs to be flipped
            V4m1 = np.zeros(self.DIM,dtype=np.complex128) # projector onto |4,m+1⟩ index is m+4
            V4m1[6+m] = 1
            ACZ += (np.diag(V3m-V4m1))*(clebsch_gordan(3,1,4,m,1,m+1)**2)/(3-m) #minus sign corrected 4 June 2012
        

        ACZ *= (self.mw_amp**2)/(8*self.rf_bias)

        # ACZ = np.diag(ACZ)

        jx_up,jy_up,jz_up = [direct_sum(j,self.DIM_DN) for j in ang_mom(spin=self.SPIN_UP,convention='Reversed')]   # reverse ordering to align with eigenstate ordering convention
        jx_dn,jy_dn,jz_dn = [direct_sum(self.DIM_UP,j) for j in ang_mom(spin=self.SPIN_DN,convention='Standard')]
        
        proj_up = direct_sum(np.eye(self.DIM_UP),self.DIM_DN,dtype=np.complex128)
        proj_dn = direct_sum(self.DIM_UP,np.eye(self.DIM_DN),dtype=np.complex128)

        mw_x = np.zeros((self.DIM,self.DIM),dtype=np.complex128)
        mw_y = np.zeros((self.DIM,self.DIM),dtype=np.complex128)

        mw_x[self.I44,self.I33] = 1
        mw_x[self.I33,self.I44] = 1
        mw_y[self.I44,self.I33] = -1j
        mw_y[self.I33,self.I44] = 1j
        


        H0 =  (((3/2)*self.rf_bias*(1+self.grel))-((25/2)*self.grel*(self.rf_bias**2)*(1/self.hf_freq))-((1/2)*(self.mw_det-7*self.rf_det))) * (proj_up-proj_dn)\
            +(self.rf_bias*(1+self.grel)*jz_dn)\
            +(self.grel*(self.rf_bias**2)*(1/self.hf_freq)*(jz_up@jz_up-jz_dn@jz_dn))\
            +((-1*self.rf_det*(jz_up+self.grel*jz_dn)))\
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

        ops = np.zeros((N_OPS,self.DIM,self.DIM),dtype=np.complex128)
        ops[0,:,:] = H0
        ops[1,:,:] = jx_up
        ops[2,:,:] = jy_up
        ops[3,:,:] = jz_up
        ops[4,:,:] = jx_dn
        ops[5,:,:] = jy_dn
        ops[6,:,:] = jz_dn
        ops[7,:,:] = mw_x
        ops[8,:,:] = mw_y

        d  = np.zeros((N_OPS,self.DIM),dtype=np.float64)
        ld = np.zeros((N_OPS,self.DIM-1),dtype=np.complex128)

        # separating diagonal and lower diagonal makes calculations faster
        for i in range(N_OPS):
            d[i,:] = np.diag(ops[i,:,:]).real
            ld[i,:] = np.diag(ops[i,:,:],-1)

        return ops, d, ld
