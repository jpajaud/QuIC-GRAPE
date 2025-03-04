import ctypes,platform
from pathlib import Path
import numpy as np

_system = platform.system()
if _system == "Linux":
    p = Path(__file__).parent/'libcg.so'
elif _system == "Windows":
   p = Path(__file__).parent/'libcg.dll'
else:
    raise RuntimeError(f"Platform <<{_system}>> is not supported")
clebsch_gordan = ctypes.cdll.LoadLibrary(p.as_posix()).clebsch_gordan
clebsch_gordan.argtypes = [ctypes.c_double]*6
clebsch_gordan.restype = ctypes.c_double
clebsch_gordan.__doc__ = """clebsch_gordan(double j1, double j2, double j, double m1, double m2, double m)"""

def ang_mom(spin=7.5,convention="Standard"):
    """Calculates spin operators J_x, J_y, and J_z
    
    Arguments
        spin = 7.5                           : half integer representing spin
        convention ∈ ["Standard","Reversed"] : ordering of eigenstates
                                               "Standard" |J,J>...|J,-J>
                                               "Reversed" |J,-J>...|J,J>

    Return
        Jx
        Jy
        Jz
    """

    # norm of spin matrix is J*(J+1)*(2*J+1)/3
    m = np.arange(spin-1,-spin-1,-1)
    v = np.sqrt( spin*(spin+1) - m*(m+1)  )

    jx = (np.diag(v,1) + np.diag(v,-1))/2

    if convention.lower()=="standard":
        jy = (np.diag(v,1) + np.diag(-v,-1))/(2j)
        jz = np.diag(np.arange(spin,-spin-1,-1))
    elif convention.lower()=="reversed":
        jy = (np.diag(-v,1) + np.diag(v,-1))/(2j)
        jz = np.diag(np.arange(-spin,spin+1,1))
    else:
        raise ValueError("Invalid convention")

    return jx,jy,jz
