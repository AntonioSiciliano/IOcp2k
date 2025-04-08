import pytest
import AtomicSnap
from AtomicSnap import AtomicSnapshots
import numpy as np
import os

def test_merge():
    """
    CHECK IF THE MERGING WORKS FINE
    ===============================
    """
    all_snap = []

    for i in range(2):
        snap = AtomicSnapshots.AtomicSnapshots()
        ucell = np.eye(3) * 17.19
        snap.init('BATCH_{}_MD_T_673_P_1000_steps_1000_dt_0.5_NH2O_128_NaCl_4_abc_17.19_17.19_17.19_wt_9.20/brines-1'.format(i+1),
                  ucell, calc_type = 'MD',
                  ext_ener = '.ener',    ext_force='.force',   ext_pos='.xyz',
                  ext_stress='.stress',  ext_cell='.cell_xyz', ext_vel='.vel')

        all_snap.append(snap)

    # First get all the attributes to check that if the snapshots object have the same attributes
    all_attr = []
    for attr in all_snap[0].__dict__:
        if not attr in all_snap[1].__dict__:
            raise ValueError("{} not found in the other snapshots".format(attr))
        if not attr in ['__total_attributes__', 'fixed_attributes']:
            all_attr.append(attr)

    # Now merge the two snapshots
    merge_snap = all_snap[0].merge_snapshots(all_snap[1])
    print(id(merge_snap) - id(all_snap[0]))

    # Check that all the attribute of the merged snapshots are correct
    for attr in all_attr:
        if not(attr in merge_snap.__dict__):
            raise ValueError("{} not found in the merged snapshots".format(attr))

    # Check the attributes of the merged object coincide with those of the single snapshots object
    for i in range(merge_snap.snapshots):
        if i < 1000:
            for attr in all_attr:
                # Check that the types coicide
                if type(getattr(merge_snap, attr)) != type(getattr(all_snap[0], attr)):
                    raise ValueError("The type does not coincides")
                
                if isinstance(getattr(merge_snap, attr), np.ndarray):
                    # Check if is it an array with shape (snapshots, x, y, z) or (snapshots, x)
                    if getattr(merge_snap, attr).shape[0] > 100:
                        if np.abs(getattr(merge_snap, attr)[i] - getattr(all_snap[0], attr)[i]).max() > 1e-10:
                            raise ValueError('The attribute {} was not concatenated properly'.format(attr))
                    # It an array with shape (x) or (x, y)
                    else:
                        if np.abs(getattr(merge_snap, attr) - getattr(all_snap[0], attr)).max() > 1e-10:
                            raise ValueError('The attribute {} was not concatenated properly'.format(attr))

        else:
            for attr in all_attr:
                # Check that the types coicide
                if type(getattr(merge_snap, attr)) != type(getattr(all_snap[1], attr)):
                    raise ValueError("The type does not coincides")
                
                if isinstance(getattr(merge_snap, attr), np.ndarray):
                    # Check if is it an array with shape (snapshots, x, y, z) or (snapshots, x)
                    if getattr(merge_snap, attr).shape[0] > 100:
                        if np.abs(getattr(merge_snap, attr)[i] - getattr(all_snap[1], attr)[i - 1000]).max() > 1e-10:
                            raise ValueError('The attribute {} was not concatenated properly'.format(attr))
                    # It an array with shape (x) or (x, y)
                    else:
                        if np.abs(getattr(merge_snap, attr) - getattr(all_snap[1], attr)).max() > 1e-10:
                            raise ValueError('The attribute {} was not concatenated properly'.format(attr))

if __name__ == "__main__":
    
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    test_merge()