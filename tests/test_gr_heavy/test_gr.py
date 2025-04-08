import AtomicSnap
from AtomicSnap import AtomicSnapshots
import numpy as np
import os
import ase, ase.io
import matplotlib
import matplotlib.pyplot as plt
# import julia, julia.Main
import os, sys, json

def retrive_snapshots(directory):
    """
    RETRIVE THE SNAPSHOTS
    =====================
    """

    os.chdir(directory)

    # Get all the directories that start with BATCH and has an outputfile
    all_dirs = []
    for mydir in os.listdir():
        if mydir.startswith('BATCH'):
            if 'output.out' in os.listdir(mydir):
                all_dirs.append(mydir)
    # Get the number of directories
    Ndir = len(all_dirs)

    # Get all the numeric indices of the directories
    all_indices = []
    uc_l = 0
    for mydir in all_dirs:
        all_indices.append(mydir.split('_')[1])
        uc_l = float(mydir.split('_')[-3])
    all_indices = np.asarray(all_indices, dtype = int)
    # Sort all the indices
    mask = np.argsort(all_indices)
    all_dirs = [all_dirs[i] for i in mask]

    # I will compute all the averages after Nstart steps
    Nstart = 5000
    if len(sys.argv[2:]) > 0:
        Ndir = int(sys.argv[2])
        
    dirs = [all_dirs[i] for i in range(Ndir)]
    
    all_snaps = []
    
    for mydir in dirs:
        snap = AtomicSnapshots.AtomicSnapshots()
        snap.init(os.path.join(mydir, 'brines-1'), unit_cell=np.eye(3) * uc_l, calc_type = 'MD',
                  ext_force='.force', ext_pos='.xyz', ext_ener='.ener', ext_stress = '.stress', ext_cell = '.cell_xyz')
        #print(np.abs(snap.energies - snap.potential_energies).max())
        all_snaps.append(snap)
    
    
    if len(all_snaps) > 1:
        print('MERGING')
        merge_snap = all_snaps[0].merge_multiple_snapshots(all_snaps)
    else:
        merge_snap = all_snaps[0]

    os.chdir(total_path)

    return merge_snap
    
def get_gr_original(snapshots):
    """
    TEST GR WITH THE ORIGINAL UNIT CELLS
    ====================================
    """

    # In picoseconds
    t_ini = 2.5
    T = snapshots.snapshots * snapshots.dt * 1e-3
    snapshots.get_pair_correlation_functions(selected_atoms = [["O", "O"]],
                                              t_ini = t_ini,
                                              my_r_range = (0.01, 9.0), bins = 500, 
                                              wrap_positions = True, use_pbc = True,
                                              json_file_result = "gr_pbc_original_tini_{:.1f}_T_{:.1f}.json".format( t_ini, T), 
                                              save_ase_atoms_file = False,
                                              show_results = True, save_plot = False)


def get_gr_modified(snapshots):
    """
    TEST GR CHANGES WITH DIFFERENT UNIT CELLS
    =========================================
    """
    # Modify the unit cell shape to see if there are any differences
    snapshots.unit_cells[:,:,:] = np.eye(3) * 17.05
    # In picoseconds
    t_ini = 2.5
    T = snapshots.snapshots * snapshots.dt * 1e-3
    snapshots.get_pair_correlation_functions(selected_atoms = [["O", "O"]],
                                              t_ini = t_ini,
                                              my_r_range = (0.01, 9.0), bins = 500, 
                                              wrap_positions = True, use_pbc = True,
                                              json_file_result = "gr_pbc_modified_tini_{:.1f}_T_{:.1f}.json".format(t_ini, T), 
                                              save_ase_atoms_file = False,
                                              show_results = True, save_plot = False)


def test():
    """
    RUN THE TEST
    ============
    """
    # The directory contianing the snapshots of the MD simulations
    directory = "/home/antonio/POSTDOC/MagBrinesSimulations/NPT/128_H2O_4_NaCl_10_wt_673K_1000bar"


    all_snapshots = retrive_snapshots(directory)

    
    get_gr_original(all_snapshots)

    get_gr_modified(all_snapshots)


    all_gr = []
    for file in os.listdir():
        if file.endswith('json'):
            with open(file, 'r') as f:
                gr = json.load(f)
                all_gr.append(gr)

    for gr in all_gr:
        plt.plot(gr['OO'][0], gr['OO'][1])
    plt.show()

    if np.sum(all_gr[0]['OO'][1]) * (gr['OO'][0][1] - gr['OO'][0][0])  == np.sum(all_gr[1]['OO'][1]) * (gr['OO'][1][1] - gr['OO'][1][0]):
        raise ValueError('The gr are insensible to the unit cell')

    
    
if __name__ == '__main__':
    """
    TEST GR CHANGING THE UNIT CELL SIZE
    ===================================
    """
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    test()


