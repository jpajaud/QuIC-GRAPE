import numpy as np

# naming convention
# u is unitary at a single phase step
# U is final unitary or forward and backward propagations
# uU is derivative with respect to cumulative

def calc_M(U_f, ks, vs):
    """M is the robustness metric defined in
    Pablo M. Poggi, Gabriele De Chiara, Steve Campbell, and Anthony Kiely
    Phys. Rev. Lett. 132, 193801 
        
        M = |(∑_ U_i.T⊗U_i.T.conj())@P/STEPS|**2
        
    This function assumes that the projector P is a diagonal matrix. In order to
    evaluate the metric M quickly, an intermediate matrix needs to be defined as:
    
    mask = np.diag(P).reshape(DIM,DIM,order='F').T

    "mask" defines a subspace of perturbation operators contained within the projector
        
    Arguments
        U_f[STEPS,DIM,DIM]  : Cumulative unitary at each step in a waveform
                              U_f[i] = U[i]@U[i-1]@...@U[1]@U[0]
        ks[q]               : Non-zero eigenvalues of mask
        vs[q,DIM]           : vs[q,:] is the eigenvector associated with eigenvalu ks[q]
    
    Return
        M                   : Robustness metric constrained to a subspace of operators        
    """

    STEPS = U_f.shape[0]

    d = U_f[:,None,:,:]*U_f.conj()

    # A_{q,i,j} = ∑_{n,m} vs_{q,n}*U_f_{i,n,m}*U_f.conj()_{j,n,m}
    A_qij = np.tensordot(vs[:,None,:,None]*U_f[None,:,:,:],U_f.conj(),axes=([3,2],[2,1]))
    # final equation is ∑_{q,i,j} ks_{q}*A_{q,i,j}*A_{q,j,i}
    a_q = np.apply_over_axes(np.sum,A_qij*A_qij.conj(),axes=[2,1]).squeeze(axis=(2,1)).real
    return (np.dot(ks,a_q).real/(STEPS*STEPS))

def calc_dM(dU_s, U_f, U_b_conj, ks, vs): # ks, vs represent subspace projection
    """dM is the derivative of the robustness metric defined in
    Pablo M. Poggi, Gabriele De Chiara, Steve Campbell, and Anthony Kiely
    Phys. Rev. Lett. 132, 193801 
        
        M = |(∑_ U_i.T⊗U_i.T.conj())@P/STEPS|**2
        
    This function assumes that the projector P is a diagonal matrix. In order to
    evaluate the metric M quickly, an intermediate matrix needs to be defined as:
    
    mask = np.diag(P).reshape(DIM,DIM,order='F').T

    "mask" defines a subspace of perturbation operators contained within the projector
        
    Arguments
        dU_s[STEPS,NPAR,DIM,DIM]  :
        U_f[STEPS,DIM,DIM]        : Cumulative unitary at each step in a waveform
                                    U_f[i] = U[i]@U[i-1]@...@U[1]@U[0]
        U_b_conj[STEPS+1,DIM,DIM] : Cumulative unitary defined in reverse
                                    U_b_conj[-1] = np.eye(DIM)
                                    U_b_conj[-i] = I@U[STEPS]@U[STEPS-1]@...@U[1-i]
        ks[q]                     : Non-zero eigenvalues of mask
        vs[q,DIM]                 : vs[q,:] is the eigenvector associated with eigenvalu ks[q]
    
    Return
        dM                        : Gradient of robustness metric constrained to a subspace of 
                                    operators        
    """

    # STEPS,NPAR,DIM,_ = dU_s.shape # derivative of final unitary
    STEPS = dU_s.shape[0]
    NPAR  = dU_s.shape[1]
    DIM   = dU_s.shape[2]

    
    # derivative of cumulative unitary at each step, 
    # s and r are indices of step and phase of derivative variable
    # i is index of cumulative step that the derivative is acting on
    # n and m are indices of unitary operator in Hilbert space
    #                  s     r    i     n   m
    duU_sr = np.zeros((STEPS,NPAR,STEPS,DIM,DIM),dtype=np.complex128)
    
    for r in range(NPAR):
        for s in range(STEPS):
            # x@y pattern
            duU_sr[s,r,s:,:,:] = np.tensordot(U_b_conj[(s+1):,:,:],dU_s[s,r,:,:],axes=(2,0)) # output shape is (STEPS-i,DIM,DIM)
    
    U_f_c_scaled = vs[:,None,:,None]*U_f.conj()[None,:,:,:]
    A_qij_c = np.tensordot(U_f_c_scaled,U_f,axes=([3,2],[2,1]))
    dA_srqij = np.tensordot( duU_sr,U_f_c_scaled, axes=([4,3],[3,2])   ).transpose(0,1,3,2,4)
    dA_srqij += dA_srqij.transpose(0,1,2,4,3).conj()

    da_srq = np.real(np.apply_over_axes(np.sum,dA_srqij*A_qij_c[None,None,:,:,:],axes=(4,3)).squeeze(axis=(4,3)))
    return 2*(da_srq@ks)/(STEPS*STEPS)



### The following code is obsolete as it is much slower, but it is easier to read
# # good
# def M_proj(Ui, Uj, ks, vs):
#     #    ks is [1] and vs is [np.ones(DIM)]  for no projection
#     # or ks is [1] and vs is [subspace_vec]  for projecting solely into subspace
#     # or ks is [kc,kv] and vs is [c,d]       for including connecting spaces as well

#     x = (Ui*Uj.conj()).sum(axis=1)
    
#     M = 0.0

#     for k,v in zip(ks,vs):
#         vx = np.dot(x,v)
#         M += np.real(k*np.conjugate(vx)*vx)

#     return M

# def dM_proj(Ui, dUi, Uj, dUj, ks, vs):
#     #    ks is [1] and vs is [np.ones(DIM)]  for no projection
#     # or ks is [1] and vs is [subspace_vec]  for projecting solely into subspace
#     # or ks is [kc,kv] and vs is [c,d]       for including connecting spaces as well

#     x = (Ui*Uj.conj()).sum(axis=1)
#     dx = (dUi*Uj.conj() + Ui*dUj.conj()).sum(axis=1)
    
#     dM = 0.0
#     for k,v in zip(ks,vs):
#         vx = np.dot(x,v)
#         vdx = np.dot(dx,v)
#         dM += 2*np.real(k*np.conjugate(vdx)*vx)

#     return dM

# def calc_M(U_f, ks, vs):

#     STEPS = U_f.shape[0]

#     # count all the diagonal terms first where i==j because they are all equal to sum(mask)
#     diag = 0.0
#     for k,v in zip(ks,vs):
#         diag += k*v.sum()**2

#     # count nontrivial off diagonal term
#     # Mij = Mji.conj(), but they are all real when mask is symmetric
#     upper_triang = 0.0
#     for i in range(STEPS):
#         for j in range(i+1,STEPS):
#             upper_triang += M_proj(U_f[i,:,:],U_f[j,:,:],ks,vs)

#     return 2*upper_triang + STEPS*diag

# # this is stupid slow
# def calc_dM(dU_s, U_f, U_b_conj, ks, vs): # ks, vs represent subspace projection

#     # STEPS,NPAR,DIM,_ = dU_s.shape # derivative of final unitary
#     STEPS = dU_s.shape[0]
#     NPAR  = dU_s.shape[1]
#     DIM   = dU_s.shape[2]

#     dM = np.zeros((STEPS,NPAR))

#     # for each step/npar, we need all U_s and dU_s_i # derivative of cumulative unitary not final

#     # need to calculate M outside
#     for j in range(NPAR): # do this first to reduce size of duU_si
#         duU_si = np.zeros((STEPS,STEPS,DIM,DIM),dtype=np.complex128) # for DIM = 16 this is 88 MB (92_160_160 bytes)
#         for i in range(STEPS):
#             # derivative is zero if step > i            
#             # x@y pattern
#             duU_si[i,i:,:,:] = np.tensordot(U_b_conj[(i+1):,:,:],dU_s[i,j,:,:],axes=(2,0)) # output shape is (STEPS-i,DIM,DIM)

#         for k in range(STEPS):
#             # evaluate M_k,j and dM_k,j here
#             uU = U_f[k:,:,:]
#             duU = duU_si[k,k:,:,:] # (STEPS-k,DIM,DIM), represents all non zero derivatives of cumulative unitaries for steps >= k with respect to phase k
#             # (STEPS-k by STEPS-k) for loop
#             # diagonal elements will all evaluate to sum(mask)
            
#             # for diagonal terms, the derivative is identically 0
#             # for off diagaonal terms, the derivative is symmetric under exchange on n and m
#             for m in range(STEPS-k):
#                 print(k,m)
#                 for n in range(1,STEPS-k):
#                     dM[k,j] += dM_proj(uU[m],duU[m],uU[n],duU[n],ks,vs) # k,v are list of coefficients and vectors
#             # now for every off diagonal combination, we can find dM
    
#     dM *= 2.0 # account for other triangle
#     return dM