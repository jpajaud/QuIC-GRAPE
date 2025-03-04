import numpy as np

# helper function to do direct sum of two matrices
def direct_sum(A,B,dtype=None):
    """Calculates the direct sum of 2 matrices
    A⊕B = [[A 0],
            [0 B]]

    Arguments
        A [:,:] or int : matrix in upper left of direct sum, A is np.zeros((A,A),B.dtype) if int
        B [:,:] or int : matrix in lower right of direct sum, B is np.zeros((B,B),A.dtype) if int
    only one argument can be int

    Return
        A⊕B

    """
    if type(A)==int:
        A = np.zeros((A,A),dtype=B.dtype)
        assert type(B)!=int, 'A and B must not both be integers'
    if type(B)==int:
        B = np.zeros((B,B),dtype=A.dtype)
        assert type(A)!=int, 'A and B must not both be integers'
    
    i,j = A.shape    
    k,l = B.shape
    if dtype is None:
        dtype = A.dtype
    M = np.zeros((i+k,j+l),dtype=dtype)
    M[:i,:j] = A
    M[i:,j:] = B
    return M

def calc_hessenberg(ld):
    """Calculates the diagonal transformation matrix from Hessenberg form for an array of lower diagonal vectors
    
    Arguments
        ld[STEPS,DIM-1]   : each row contains the lower diagonal of the Hamiltonian at that step
        
    Return
        phases[STEPS,DIM] : np.diag(phases[i,:]) gives transformation matrix from Hessenberg form to standard basis
    """
    # R[i] = np.diag(ld[:,i])
    # v_true = R@v
    
    phases = np.hstack((np.ones((ld.shape[0],1)),ld/np.abs(ld)))
    return np.multiply.accumulate(phases,axis=1) # return is [STEPS,DIM]

