import numpy as np

import ase, ase.io
from ase import Atoms

import os, sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import scipy, scipy.optimize

import copy

from mpi4py import MPI

import time

ANG2_PS_TO_SI = 1e-8
KELVIN_TO_EV = 8.6173e-5
_e_charge_ = 1.602176634

def get_Vcom_from_ase_atoms(atoms):
    """
    GET THE CENTER OF MASS VELOCITIES
    ================================

    Parameters:
    -----------
        -atoms: an ase atoms object

    Returns:
    --------
        -V_com: np.array with shape 3, the center of mass velocity in ANGSROM/PICOSECOND
    """
    # Get the masses
    masses = np.asarray(atoms.get_masses())

    # Get the center of mass position
    V_com = np.einsum('i, ia -> a', masses, atoms.get_velocities()[:,:]) /np.sum(masses)

    return V_com


def get_selected_velocities(atoms, atom_types, atom_qs, subtract_vcom = True):
    """
    GET THE CURRENT FOR THE SELECTED ATOMS
    =======================================

    Parameters:
    -----------
        -atoms: an ase atoms object
        -atom_types: list of string, the selected atoms types
        -atom_qs: np.array with the oxidation charge for each atom
        -subtract_vcom: bool, if True we subtract the center of mass velocities

    Returns:
    --------
        -J: np.array with shape (N_at_selected, 3), the current for each selected atomic types in the simulation box
    """
    # Get the total number of atoms
    N_at_tot = len(atoms.positions[:,0])
    
    # Get all the chemical symbols
    chem_symb = np.asarray(atoms.get_chemical_symbols()).ravel()
    
    # Select only the atoms I want
    selected_atoms = []
    selected_qs = []
    for i, atom in enumerate(atom_types):
        my_sel = np.arange(N_at_tot)[np.isin(chem_symb, [atom])]
        # Select the atoms contrbuting to the ionic conductivity
        selected_atoms.append(my_sel)
        # Get the corresponding oxidations charges
        selected_qs.append([atom_qs[i]] * len(my_sel))
    # Transform everything in array
    selected_atoms = np.asarray(selected_atoms).ravel()
    selected_qs    = np.asarray(selected_qs, dtype = int).ravel()
    # print(chem_symb[selected_atoms])
    # print(selected_qs)

    # Get the velocities of the selected atoms
    selected_velocities = atoms.get_velocities()[selected_atoms,:]
    
    if subtract_vcom:
        # Subtract the COM 
        selected_velocities[:,:] -= get_Vcom_from_ase_atoms(atoms)

    J = np.einsum('i, ia -> ia', selected_qs, selected_velocities)

    return J

def get_JJ(m, N, atom_types, atom_qs, debug = True, verbose = True):
    """
    RETURNS THE CURRENT CURRENT CORRELATION
    ========================================
    
    Rember that ase use EV, ANGSTROM
    
    We compute the Ionic Conductivity as a windowed average, i.e. we compute the correlations for all possible lag times

    .. math::  C(t) = \vec{J}(t)\vec{J}(0)

    Parameters:
    -----------
        -m: int, the corresponding time is dt * m PICOSECOND
        -N: int, the full length of the trajectory (N * dt is the time length in PICOSECOND)
        -atom_types: list of string, the selected atom types
        -atom_qs: list of string, the selected atom types
        -debug: bool
        -verbose: bool

    Returns:
    --------
        -JJ_average: float, the value of the MSD(m * dt)
        -JJ_error:   float, the value of the error on MSD(m * dt)
    """
    # The mean square displacement for all the possible starting positions
    JJ_chunk = np.zeros(N - m, dtype = float)

    # In this way the sum goes from 0 to N - m - 1 (python convention)
    for k in np.arange(0, N - m):

        # Get the current in ANGSTROM^/PICOSECOND (N_at_selected, 3) at step k
        J_k = get_selected_velocities(all_ase_atoms[k], atom_types, atom_qs)
        
        # Get the currentin ANGSTROM^/PICOSECOND (N_at_selected, 3) at step k + m
        J_k_plus_m = get_selected_velocities(all_ase_atoms[k + m], atom_types, atom_qs)
        
        # The scalar product between the velocities (ANGSTROM/PICOSECOND)^2
        d =  np.einsum('ia -> a', J_k_plus_m[:,:]).dot(np.einsum('ia -> a', J_k[:,:]))

        # Get  the current-current correlators
        JJ_chunk[k] = d

    # Get average and error
    JJ_average = np.sum(JJ_chunk[:]) / (N - m)

    JJ_error = np.sqrt(np.sum(JJ_chunk[:] - JJ_average)**2) / (N - m)
    
    return JJ_average, JJ_error
        

def save_plot_results(x, y, z, show_error = True):
    """
    PLOT THE RESULTS
    ================

    Parameters:
    -----------
        -x: np.array, the MD time in pPICOSECOND
        -y: np.array, the JJ in ANGSTROM^2/PICOSECOND^2
        -z: np.array the error on MSD in  ANGSTROM^2/PICOSECOND^2
    """
    def f(var, a, b):
        return a * var + b
    print("\nANALYSIS OF THE CONDUCTIVITY")
    # Width and height
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 1, figure=fig)

    # Plot the MSD with its error
    ax = fig.add_subplot(gs[0,0])
    ax.plot(x, y, lw = 3, color = 'blue')
    if show_error:
        ax.fill_between(x, y - z, y + z, color='blue', alpha=0.3)


    fracs = [0.2, 0.4]
    dx = x[1] - x[0]
    for frac in fracs:
        mask = x > (frac * x[0])
        sigma_SI = _e_charge_ * 1e+3 * np.sum(y[mask]) * dx /(3 * V_mean * T * KELVIN_TO_EV)
        print(x[mask].min(), sigma_SI)
    
    ax.set_xlabel('Time [ps]', size = 15)
    ax.set_ylabel('J(t)J(0) [Angstrom$^2$/ps$^2$]', size = 15)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.legend(fontsize = 12)

    plt.tight_layout()

    plt.show()

    with open("cond.txt", "w") as file:
        file.write(f"#t [ps] JJ [ANGSTROM^2/PICOSECOND^2] err JJ [ANGSTROM^2/PICOSECOND^2]\n")
        for i, j, k in zip(x, y, z):
            file.write(f"{i} {j} {k}\n")


if __name__ == '__main__':
    """
    A PYTHON SCRIPT TO GET THE CONDUCTIVITY
    =======================================

    The unitsof INPUTS used are ANGSTROM and PICOSECONDS. 

    Paramters:
    ----------
        -dir_ase_atoms: the path to the ase objects saved as xyz
        -dt: float, the time step in FEMPTOSECOND
        -T: float, the temperature in KELVIN
        -at, q_at: a series of atomic types with oxidations charges
    """
    matplotlib.use('tkagg')
    debug = True

    # Initialize MPI communicator
    comm = MPI.COMM_WORLD  
    # Process ID
    rank = comm.Get_rank()  
    # Number of processes
    size = comm.Get_size()

    if rank == 0:
        t1 = time.time()
        if len(sys.argv[1:]) < 2:
            raise ValueError("The usage is python get_self_diffusion_coefficient.py dir_ase_atoms dt at1 at2")
    
    # Where to find the ase atoms
    dir_ase_atoms = sys.argv[1]
    
    # The time steps in FEMPTOSECOND
    dt = float(sys.argv[2])
    # Now in PICOSECOND
    dt *= 1e-3

    # The temperature in KELVIN
    T = float(sys.argv[3])
    
    # The atomic types and charges
    selected_atoms = []
    selected_qs    = []
    for i in range(len(sys.argv[4:])//2):
        selected_atoms.append(sys.argv[2*i + 4])
        selected_qs.append(sys.argv[2*i + 5])
    # Transform into array
    selected_qs = np.asarray(selected_qs)

    
    # Read all the ase atoms objects
    all_ase_atoms = ase.io.read(dir_ase_atoms, index = ':')
    all_ase_atoms = all_ase_atoms[:3000]
    # The size of the full trajectory
    len_full_trajectory = len(all_ase_atoms)

    if rank == 0:
        if len_full_trajectory % size != 0:
            raise ValueError("Choose a number of processes which is a integer divisor of {}".format(len_full_trajectory))
    
    if rank == 0:
        print("\n========CONDUCTIVITY ANALYSIS========")
        print("Ionic conductivity for {}".format(selected_atoms))
        print("with ox charges {}".format(selected_qs))
        print("Number of frames is {} the T {} ps".format(len(all_ase_atoms), len(all_ase_atoms) * dt))
        # Now get the lagtime in PICOSECOND for the trajectory
        t = np.arange(len_full_trajectory) * dt
        # print('MASTER t {}'.format(t))
        
        # The lagtimes to be processed by each processor PICOSECOND
        chunks_t = np.array_split(t, size)
        # The lagtimes inidices to be processed by each processor, useful to get the ase atoms easily
        chunks_t_index = np.array_split(np.arange(len_full_trajectory), size)
        print()
    else:
        chunks_t = None
        chunks_t_index = None
       
    
    # Scatter the chunks to each processorss
    chunk_t       = comm.scatter(chunks_t, root = 0)
    chunk_t_index = comm.scatter(chunks_t_index, root = 0)
    print('RANK {} t {} {}'.format(rank, chunk_t, chunk_t_index))
    
    # The size of the chunk trajectory
    len_chunk_trajectory = len(chunk_t)

    if len(chunk_t) != len(chunk_t_index):
        raise ValueError("The chunks sizes are not correct")
        
    # Get the J(t) J(0) correlation ANGSTROM^2 /PICOSECOND^2
    JJ       = np.zeros(len_chunk_trajectory, dtype = float)
    error_JJ = np.zeros(len_chunk_trajectory, dtype = float)
    # Get the average of volume ANGSTROM^3
    V_mean = 0.
    
    # Cycle on the possible time distances (lag times)
    for i, t_index in enumerate(chunk_t_index):
        # Get the JJ correlation for the current lag time  (ANGSTROM/PICOSECOND)^2
        mean, error = get_JJ(t_index, len_full_trajectory, selected_atoms, selected_qs)
        if rank == 1:
            print('RANK {} t {} jj {}'.format(rank, t_index, mean))
        if rank == 2:
            print('RANK {} t {} jj {}'.format(rank, t_index, mean))
        JJ[i] = mean
        error_JJ[i] = error
        # Get the average volume ANGSTROM^3
        V_mean += all_ase_atoms[t_index].get_volume() /len_chunk_trajectory

    comm.Barrier()

    # print('RANK {} {}'.format(rank, V_mean))
    # print('RANK {} jj {}'.format(rank, JJ))

    # Gather all the volumes
    all_V = comm.gather(V_mean, root = 0)

    # Gather all JJ arrays at rank 0
    gathered_JJ = comm.gather(JJ, root=0)
    # Gather all JJ arrays at rank 0
    gathered_err_JJ = comm.gather(error_JJ, root=0)

    
    if rank == 0:
        t2=time.time()
        
        print("Gathered V values at rank 0:", all_V)
        print('\n\nTIME ELAPSED FOR COMPUTING THE CURRENT-CURRENT CORRELATION is {} sec\n'.format(t2 - t1))

        # The mean volume in ANGSTROM3
        V_mean = np.sum(all_V) /size

        # The final JJ correlation
        final_JJ = np.concatenate(gathered_JJ)
        err_final_JJ = np.concatenate(gathered_err_JJ)
        
        # Save all the results
        save_plot_results(t, final_JJ , err_final_JJ)