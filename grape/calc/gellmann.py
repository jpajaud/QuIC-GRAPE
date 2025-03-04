import numpy as _np

def gen_GellMann_traceless(d):
    #
    # gellmann.gen_basis(dim)
    # Generates basis vectors of Gell-Mann basis as array of orthogonal
    # Hermitian matrices
    #    
    # Arguments:
    #     dim   : dimension of Hilbert space
    # Output:
    #     basis : 3d array of basis elements of shape (dim^2,dim,dim) 
    #                    (output from gellmann.gen_basis(dim))
    # 

    basis = _np.zeros((d*d-1,d,d),dtype=_np.complex128)
    # basis[0,:,:] = _np.eye(d)/_np.sqrt(d)
    for i in range(1,d):
        basis[i-1,:,:] = _np.diag(_np.hstack((_np.ones(i), -i, _np.zeros(d-1-i))))/_np.sqrt((i+1)*i)
    ind = d-1
    for i in range(1,d):
        for j in range(i):
            basis[ind,i,j] = 1/_np.sqrt(2)
            basis[ind,j,i] = 1/_np.sqrt(2)
            basis[ind+1,i,j] = 1j/_np.sqrt(2)
            basis[ind+1,j,i] = -1j/_np.sqrt(2)
            ind += 2
    return basis

# turn vector into Hamiltonian with np.tensordot(vector,basis,axes=(0,0))