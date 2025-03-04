from typing import List
import numpy as np

def prepare_subspace(subspace_indices:List[int],DIM:int=16):
    """Helper function used to create a vector for use in run.optimize_* functions
    The return value is of length DIM and has a 1 in every column specified in 
    subspace_indices list
    Example use:
        prepare_subspace([0,1],DIM=4)
        returns np.array([1,1,0,0])

    Arguments
        subspace_indices list[int] : indices of eigenstate
        DIM int                    : total dimension of system

    Return
        subspace_vec

    """
    subspace_vec = np.zeros(DIM)
    subspace_vec[subspace_indices] = 1
    return subspace_vec

def expand_from_subspace(M,subspace_vec):
    """Helper function to take a matrix of dimension S and embed it in an 
    operator of dimension D in a subspace defined by subspace_vec

    Arguments
        M[S,S]           : Matrix of dimension S
        subspace_vec[D,] : Subspace as returned by prepare_subspace(subspace_indices,DIM)
    
    Return
        M_expanded[D,D]  : Matrix with M placed in subspace and zeros everywhere else
    """
    DIM = len(subspace_vec)
    SUBSPACE_DIM = sum(subspace_vec)
    assert M.shape[0]==SUBSPACE_DIM, "Dimension of subspace must match dimension of unitary"
    mask = np.outer(subspace_vec,subspace_vec).astype(bool).flatten(order='C')
    M_expand = np.zeros(DIM*DIM,dtype=np.complex128)
    M_expand[mask] = M.flatten(order='C')
    return M_expand.reshape(DIM,DIM,order='C')