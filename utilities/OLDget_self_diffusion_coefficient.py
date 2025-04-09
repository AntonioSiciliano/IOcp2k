import numpy as np
import ase, ase.io
from matplotlib.ticker import MaxNLocator
from ase import Atoms
import os, sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import copy
from mpi4py import MPI
import time
import multiprocessing
# import psutil
ANG2_PS_TO_SI = 1e-8
FAST = False
def get_selected_positions(atoms):
    """
    GET THE POSITIONS OF THE type1 ATOMS
    ====================================
    """
    #Get the total number of atoms
    N_at_tot = len(atoms.positions[:,0])
    
    # Get all the chemical symbols
    chem_symb = np.asarray(atoms.get_chemical_symbols()).ravel()
    # Select only the atoms I want
    selected_atoms = np.arange(N_at_tot)[np.isin(chem_symb, [type1])]
    
    # Return the corresponding position
    return atoms.positions[selected_atoms,:]


def get_MSD_snapshot(atoms, R_0):
    """
    GET THE MSD FOR A SINGLE SNAPSHOT
    """
    # get the positions for the selected atoms (N_at_selected, 3)
    R_t = get_selected_positions(atoms)
    
    # The number of selected atoms
    N_at_selected = len(R_t[:,0])
    
    # The distance travelled ANGSTROM for each of the selected atoms (N_at_selected, 3)
    d = R_t - R_0
    
    # Get the scalar product squared between the current positions and the initial ones
    # ANGSTROM^2
    return np.sum(np.einsum('ia, ia -> i', d, d)) /(6. * N_at_selected)

def get_MSD(ase_atoms_slice, debug = True, verbose = True, using_pbc = True):
    """
    RETURNS THE MEAN SQUARE DISPLACEMENT FOR ONLY ONE TRAJECTORY
    ============================================================
    
    Rember that ase use ANGSTROM
    
    We compute the Mean Square Displacement on one trajectory

    .. math::  MSD(t) =\\frac{1}{N} \sum_{i=1}^{N} \\frac{1}{6} |r_i(t) - r_i(0)|^2

    where t are the timesteps of the current trajectories

    Parameters:
    -----------
        -ase_atoms_slice: a list of ase atoms objects on which we compute the MSD

    Returns:
    --------
        -MSD_chunk: the MSD for the current trajectory
    """
    # The mean square dispacement for this slice of the trajectory
    MSD_chunk = np.zeros(len(ase_atoms_slice), dtype = type(dt))
    
    # Get the first position in ANGSTROM (N_at_selected, 3)
    R_0 = get_selected_positions(ase_atoms_slice[0])

    if FAST:
        # Create a pool of processes for parallel computation
        with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
            results = pool.starmap(get_MSD_snapshot, [(atoms, R_0) for atoms in ase_atoms_slice])
            print(len(results))
            
        # Collect results back and populate MSD_chunk
        for i, msd_value in enumerate(results):
            MSD_chunk[i] = msd_value
    else:
        for i, atoms in enumerate(ase_atoms_slice):
            MSD_chunk[i] = get_MSD_snapshot(atoms, R_0)

    return MSD_chunk
        

def plot_results(x, y):
    """
    PLOT THE RESULTS
    =================
    """
    #Width and height
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(x, y, lw = 3, color = 'k')
    slope, intercept = np.polyfit(x, y , 1)
    ax.plot(x, x * slope + intercept, lw = 3, color = 'green', label = "Fit {:.4e} m$^2$/s".format(slope * ANG2_PS_TO_SI))
    print(slope * ANG2_PS_TO_SI)
    ax.set_xlabel('Time [ps]', size = 15)
    ax.set_ylabel('MSD [Angstrom$^2$]', size = 15)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.legend(fontsize = 15)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax = fig.add_subplot(gs[0,1])
    # Linear fit (degree=1)
    dy_dx = np.gradient(y, x[1] - x[0])
    # mask = t > t_tol
    
    ax.plot(x, dy_dx, lw = 3, color = 'red')
    
    ax.set_xlabel('Time [ps]', size = 15)
    ax.set_ylabel('$\\frac{d MSD(t)}{d t}$ [Angstrom$^2$/ps]', size = 15)
    ax.tick_params(axis = 'both', labelsize = 12)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize = 15)
    plt.tight_layout()

    # if not(img_name is None):
    #     plt.savefig(img_name, dpi = 500)
    plt.show()



    
if __name__ == '__main__':
    """
    A PYTHON SCRIPT TO GET THE SELF DIFFUSION COEFFICIENT
    =====================================================

    Uses ANGSTROM units as in ASE
    """
    # Check the time
    t1=time.time()
    matplotlib.use('tkagg')
    debug = True
    
    # where to find the ase atoms
    dir_ase_atoms = sys.argv[1]
    # The atomic type
    type1 = sys.argv[2] 
    # The time steps in FEMPTOSECOND
    dt = float(sys.argv[3])
    # Now in PICOSECOND
    dt *= 1e-3 
    # Slices, the number of subtrajectories we want to analyze to get the Mean Square Displacement
    Nslices = int(sys.argv[4])
    
    # Read all the ase atoms objects
    all_ase_atoms = ase.io.read(dir_ase_atoms, index = ':')
    # The size of the full trajectory
    len_full_trajectory = len(all_ase_atoms)
    if len_full_trajectory % Nslices != 0:
        raise ValueError("Please choose Nslices as a divisor of the full trajectory leght {}".format(dir_ase_atoms))
    # The size of the sub trajectories
    len_sub_trajectories = len(all_ase_atoms)//Nslices
    print("Dividing the full trajectory of {} snapshots in {} sub trajectories of {} snapshots".format(len_full_trajectory,
                                                                                                        Nslices,
                                                                                                        len_sub_trajectories))

    print("The full trajectory was long {} ps while the sub trajectories are of {} ps".format(len_full_trajectory * dt, len_sub_trajectories * dt))

    # Now get the time in PICOSECOND, the length is the one of the subtrajectories
    t = np.arange(len_sub_trajectories) * dt
    # Get the Mean Square Displacement in ANGSTROM^2
    MSD = np.zeros(len_sub_trajectories, dtype = type(dt))

    # Cycle on the number of subtrajectories
    for i in range(Nslices):
        print("\n\nSUBTRAJECTORY #{}".format(i))
        # Get the ase atoms slices for the subtrajectory
        start_index, final_index = i * len_sub_trajectories, i *len_sub_trajectories + len_sub_trajectories
        ase_atoms_slice = all_ase_atoms[start_index:final_index]
        if debug:
            print('start index {} final index {}'.format(start_index, final_index))
            print('N slices analyzed ', len(ase_atoms_slice))
            
        # Get the MSD for the current slice
        res = get_MSD(ase_atoms_slice)
        MSD += res

    MSD /= Nslices

    t2=time.time()
    print("TIME ELAPSED", t2 - t1)

    plot_results(t, MSD)
    
    