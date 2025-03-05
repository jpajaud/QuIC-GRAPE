# QuIC-GRAPE
Python library that implements gradient ascent pulse engineering (GRAPE) [[1]](#1) for use in the Quantum Information and Control Group lab at the University of Arizona. The package implements several types of optimization.

Conventional control allows us to specify a target unitary to find a waveform that implements a full arbitrary unitary dynamics[[2]](#2). Conventional control also allows us to specify a target isometry to find a waveform that only maps an input state to an output state[[2]](#2).

Eigenvalue only control allows us to specify a target unitary without regard to the basis in which the simulation is performed. This allows for the use of shorter waveforms to achieve the same dynamics[[3]](#3).

Universal robust control (URC) is a new protocol that seeks to generate robust waveforms without knowledge of the system architecture that defines the quantum processor[[4]](#4).

## References
<a id="1">[1]</a>
N. Khaneja, T. Reiss, C. Kehlet, T. Schulte-Herbruggen, and S. J. Glaser, Journal of Magnetic Resonance 172, 296 (2005).

<a id="2">[2]</a>
Brian Eric Anderson. Unitary transformations in a large Hilbert space. PhD thesis, University of Arizona, Tucson, AZ, 2013.

<a id="3">[3]</a>
Nathan Kenneth Lysne. Sensitivity to imperfections of analog quantum simulation on atomic qudits. PhD thesis, University of Arizona, Tucson, AZ, 2020.

<a id="4">[4]</a>
Pablo M. Poggi, Gabriele De Chiara, Steve Campbell, and Anthony Kiely. Phys. Rev. Lett. 132, 193801