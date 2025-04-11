import pytest
import AtomicSnap
from AtomicSnap import AtomicSnapshots
import numpy as np
import os
import ase, ase.io
import matplotlib
import matplotlib.pyplot as plt
# import julia, julia.Main
import os, sys


def test_fold():
    """
    CHECK IF THE FOLDING OF THE DIOLES WORKS
    ========================================
    """
    snap = AtomicSnapshots.AtomicSnapshots()
    snap.init('brines-1', unit_cell=np.eye(3) * 20, calc_type = 'MD',
              ext_pos='.xyz',  ext_dipoles=".dipoles")

    # Copy the initial result
    original_d = np.copy(snap.dipoles)
    # Refold the dipoles
    snap.refold_all_dipoles(Nmax = 2, tol = 1.0, debug = False)

    print()
    for i in range(3):
        plt.plot(original_d[:,i], lw = 3, label = "BEFORE")
        plt.plot(snap.dipoles[:,i], lw = 2, label = "AFTER")
        plt.legend()
        plt.show()
    
        delta = np.abs(snap.dipoles[1:,i] - snap.dipoles[:-1,i])
        print('|| DIFF ', delta.max())
        if np.any(delta > 1.0):
            raise ValueError("The refolding for component {} is not correct".format(i))

        


if __name__ == '__main__':
    """
    TEST FOLDING
    ============
    """
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)


    test_fold()