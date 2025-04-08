import pytest
import AtomicSnap
from AtomicSnap import AtomicSnapshots
import numpy as np
import os

def test_read_basic():
    snap = AtomicSnapshots.AtomicSnapshots()
    ucell = np.eye(3) * 10
    snap.init('water-1', ucell, calc_type = 'MD', ext_ener = '.ener', ext_force='.force', ext_pos='.xyz', ext_stress='.stress', ext_vel='.vel')


    # Hard CODED PART DO NOT MODIFY IT
    E_1 = -34.3355531335
    forces_1_at3 = np.array([0.0349178145, 0.1128674195, 0.1257836003])
    pos_1_at5 = np.array([0.8242796529,  4.3159449448, 4.6279041827])
    vel_0_at4 = np.array([0.0011907260 , -0.0003322850, 0.0001860094])
    stress_1_yx = 69869.6169197316  
    V_pot_1 = -34.335553134 
    T_1 = 950.211599401 
    print('SNP', snap.snapshots)
    if np.abs(snap.energies[1] - E_1) > 1e-8:
        raise ValueError('The energy is not correct')

    if np.abs(snap.forces[1,3,:] - forces_1_at3).max() > 1e-8:
        raise ValueError('The force is not correct')

    if np.abs(snap.positions[1,5,:] - pos_1_at5).max() > 1e-8:
        raise ValueError('The position is not correct')
        
    if np.abs(snap.velocities[0,4,:] - vel_0_at4).max() > 1e-8:
        raise ValueError('The velocity is not correct')
        
    if np.abs(snap.stresses[1,1,0] - stress_1_yx * AtomicSnap.AtomicSnapshots.BAR_TO_HA_BOHR3).max() > 1e-8:
        raise ValueError('The stress is not correct')
       
    if np.abs(snap.potential_energies[1] - V_pot_1) > 1e-8:
        raise ValueError('The potential energy is not correct')
        
    if np.abs(snap.temperatures[1] - T_1) > 1e-8:
        raise ValueError('The temtperature is not correct')
  
    
    
if __name__ == "__main__":
    
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    test_read_basic()
