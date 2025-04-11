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


def test_read():
    """
    CHECK IF THE READING OF THE DIPOLES WORKS
    =========================================
    """

    snap = AtomicSnapshots.AtomicSnapshots()
    snap.init('brines-1', unit_cell=np.eye(3) * 20, calc_type = 'MD',
              ext_pos='.xyz',  ext_dipoles=".dipoles")

    r_1       = np.array([-0.65419431,      0.43728347,     -0.27230561])
    shift_1   = np.eye(3) *  90.85995436
    dip_1    = np.array([-0.3840670E+02 , -0.2034359E+02,   0.2612150E+02])


    r_5       = np.array([-0.62548945  ,    0.43714206  ,   -0.27241192])
    shift_5   = np.eye(3) *  90.82109529 
    dip_5    = np.array([-0.3725231E+02 , -0.1852588E+02 , 0.2440124E+02 ])

    r_1000 = np.array([0.71837816 ,     0.48393653 ,    -0.58131862])
    shift_1000 = np.eye(3) * 91.74083893
    dip_1000 = np.array([ 0.3571053E+02 , -0.6925751E+01,  -0.7651333E+01])

    delta = np.abs(snap.dipoles_origin[0,:] - r_1 * AtomicSnapshots.BOHR_TO_ANGSTROM).max()
    if delta > 1e-10:
        raise ValueError("Reading of the dipole origin is not correct")

    delta = np.abs(snap.dipoles_quantum[0,:] - shift_1 * AtomicSnapshots.DEBEYE_TO_AU).max()
    if delta > 1e-10:
        raise ValueError("Reading of the dipole quantum is not correct")
        
    delta = np.abs(snap.dipoles[0,:] - dip_1 * AtomicSnapshots.DEBEYE_TO_AU).max()
    if delta > 1e-10:
        raise ValueError("Reading of the dipole is not correct")



    delta = np.abs(snap.dipoles_origin[5,:] - r_5 * AtomicSnapshots.BOHR_TO_ANGSTROM).max()
    if delta > 1e-10:
        raise ValueError("Reading of the dipole origin is not correct")

    delta = np.abs(snap.dipoles_quantum[5,:] - shift_5 * AtomicSnapshots.DEBEYE_TO_AU).max()
    if delta > 1e-10:
        raise ValueError("Reading of the dipole quantum is not correct")
        
    delta = np.abs(snap.dipoles[5,:] - dip_5 * AtomicSnapshots.DEBEYE_TO_AU).max()
    if delta > 1e-10:
        raise ValueError("Reading of the dipole is not correct")



    delta = np.abs(snap.dipoles_origin[-1,:] - r_1000 * AtomicSnapshots.BOHR_TO_ANGSTROM).max()
    if delta > 1e-10:
        raise ValueError("Reading of the dipole origin is not correct")

    delta = np.abs(snap.dipoles_quantum[-1,:] - shift_1000 * AtomicSnapshots.DEBEYE_TO_AU).max()
    if delta > 1e-10:
        raise ValueError("Reading of the dipole quantum is not correct")
        
    delta = np.abs(snap.dipoles[-1,:] - dip_1000 * AtomicSnapshots.DEBEYE_TO_AU).max()
    if delta > 1e-10:
        raise ValueError("Reading of the dipole is not correct")
        


if __name__ == '__main__':
    """
    SCRIPT TO PLOT THE EVOLUTION OF NPT SIMULATIONS IN CP2k
    =======================================================
    """
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)


    test_read()