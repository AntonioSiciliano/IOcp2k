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

def get_selected_positions(atoms):
    """
    GET THE POSITIONS OF THE type1 ATOMS
    ====================================

    Parameters:
    -----------
        -atoms: an ase atoms object
    """
    # Get the total number of atoms
    N_at_tot = len(atoms.positions[:,0])
    
    # Get all the chemical symbols
    chem_symb = np.asarray(atoms.get_chemical_symbols()).ravel()
    # Select only the atoms I want
    selected_atoms = np.arange(N_at_tot)[np.isin(chem_symb, [type1])]

    return atoms.positions[selected_atoms,:]

def get_MSD(ase_atoms_slice, debug = True, verbose = True):
    """
    RETURNS THE MEAN SQUARE DISPLACEMENT
    ====================================
    
    Rember that ase use EV, ANGSTROM
    
    We compute the Mean Square Displacement on one trajectory

    .. math::  MSD(t) =\\frac{1}{N} \sum_{i=1}^{N} \\frac{1}{6} |r_i(t) - r_i(0)|^2

    where N is the number of targeted atoms

    Parameters:
    -----------
        -ase_aatoms_slice: a list of ase atoms objects
        -debug: bool
        -verbose: bool

    Returns:
    --------
        -MSD_chunk: np.array, the MSD(t) for the current trajectory
    """

    # The mean square dispacement for this slice of the trajectory
    MSD_chunk = np.zeros(len_sub_trajectories, dtype = type(dt))
    
    # Get the first position in ANGSTROM (N_at_selected, 3)
    R_0 = get_selected_positions(ase_atoms_slice[0])
    
    for i, atoms in enumerate(ase_atoms_slice):
        # at the current time get the positions for the selected atoms (N_at_selected, 3)
        R_t = get_selected_positions(atoms)
        # The number of selected atoms
        N_at_selected = len(R_t[:,0])
        # The distance travelled ANGSTROM for each of the selected atoms (N_at_selected, 3)
        d = R_t[:,:] - R_0[:,:]
        # Get the scalar product squared between the current positions and the initial ones ANGSTROM^2
        # First I sum on the Cartesian coordinate then on the number of selected atoms
        MSD_chunk[i] = np.sum(np.einsum('ia, ia -> i', d[:,:], d[:,:])) /(6. * N_at_selected)
    
    return MSD_chunk
        

def save_plot_results(x, y, z, show_error = True):
    """
    PLOT THE RESULTS
    ================

    Parameters:
    -----------
        -x: np.array, the MD time in picosecond
        -y: np.array, the MSD in ANGSTROM^2
        -z: np.array the error on MSD in ANGSTROM^2
    """
    def f(var, a, b):
        return a * var + b
    print("\nANALYSIS OF THE MSD")
    # Width and height
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # Plot the MSD with its error
    ax = fig.add_subplot(gs[0,0])
    ax.plot(x, y, lw = 3, color = 'blue', label = 'MSD')
    if show_error:
        ax.fill_between(x, y - z, y + z, color='blue', alpha=0.3)

    # To get the diffusion consant we make a linear fit of the MSD starting from t_tol
    my_colors = ['k', 'red', 'purple', 'green']
    selected_times = np.linspace(0.1, 0.9, 4) * x[-1]
    for ic, t_tol in  enumerate(selected_times):
        # Select time larger than t_tol
        mask = x > t_tol
        # Make the linear fit, if the MSD at large time is linear then the slope is exactly the diffusion constant
        slope1, intercept1 = np.polyfit(x[mask], y[mask] , 1)
        data_fit, cov = scipy.optimize.curve_fit(f, x[mask], y[mask], sigma = z[mask], absolute_sigma = True)
        slope, intercept = data_fit[0], data_fit[1]
        err_slope = np.sqrt(cov[0][0])
        ax.plot(x[mask], x[mask] * slope + intercept, lw = 3, color = my_colors[ic],
                ls = ':', label = "D={:.4e} {:.4e} m$^2$/s".format(slope * ANG2_PS_TO_SI, err_slope * ANG2_PS_TO_SI))
        # Mark the time after which we made the linear extrapolation
        ax.axvline(t_tol, color = my_colors[ic], ls = ":", alpha = 0.3)
        print("\nTmin={:.3f}ps D={:.4e} {:.4e} m$^2$/s".format(t_tol, slope* ANG2_PS_TO_SI, err_slope * ANG2_PS_TO_SI))
        chisqr = np.sum((y[mask] - f(x[mask], slope, intercept))**2/z[mask]**2)
        dof = len(y[mask]) - 2
        print("Reduced chi^2 {}".format(chisqr/dof))
    
    ax.set_xlabel('Time [ps]', size = 15)
    ax.set_ylabel('MSD(t) [Angstrom$^2$]', size = 15)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.legend(fontsize = 12)

    # Show the gradient of the MSD(t)
    ax = fig.add_subplot(gs[0,1])
   
    dy_dx = np.gradient(y, x[1] - x[0])
    err_dy_dx = np.gradient(z, x[1] - x[0])
    ax.plot(x, dy_dx, lw = 3, color = 'red')
    if show_error:
        ax.fill_between(x, dy_dx - err_dy_dx, dy_dx + err_dy_dx, color='red', alpha=0.3)
    ax.set_xlabel('Time [ps]', size = 15)
    ax.set_ylabel('$\\frac{d MSD(t)}{d t}$ [Angstrom$^2$/ps]', size = 15)
    ax.tick_params(axis = 'both', labelsize = 12)
    plt.tight_layout()

    plt.show()

    with open("MSD_{}.txt".format(type1), "w") as file:
        file.write(f"#t [ps] MSD [ANGSTROM^2] err MSD [ANGSTROM^2]\\n")
        for i, j, k in zip(x, y, z):
            file.write(f"{i} {j} {k}\n")


if __name__ == '__main__':
    """
    A PYTHON SCRIPT TO GET THE SELF DIFFUSION COEFFICIENT
    =====================================================

    The unitsof INPUTS used are ANGSTROM and PICOSECONDS. 

    Paramters:
    ----------
        -dir_ase_atoms: the path to the ase objects saved as xyz
        -type1: string, atomic type
        -dt: the time step in FEMPTOSECOND
        -Nslices: how many times we divide the trajectory to get the self diffusion coefficient
    """
    # Check the execution time
    t1 = time.time()
    matplotlib.use('tkagg')
    debug = True

    if len(sys.argv[1:]) < 2:
        raise ValueError("The usage is python get_self_diffusion_coefficient.py dir_ase_atoms at_type dt Nslices")
    
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

    if Nslices == 1:
        raise ValueError("To compute the error we need to divide the full trajectory in at least 2 slices")
    
    # Read all the ase atoms objects
    print("\nReading ase atoms")
    all_ase_atoms = ase.io.read(dir_ase_atoms, index = ':')
    print("Finish to read ase atoms")
    
    # The size of the full trajectory
    len_full_trajectory = len(all_ase_atoms)
    if len_full_trajectory % Nslices != 0:
        raise ValueError("Please choose Nslices {} as a divisor of the full trajectory snapshots {}".format(Nslices, len_full_trajectory))
        
    # The size of the sub trajectories
    len_sub_trajectories = len(all_ase_atoms)//Nslices
    print("\n\nSELF-DIFFUSION COEFFICIENT")
    print("Dividing the full trajectory of {} snapshots in {} sub trajectories of {} snapshots".format(len_full_trajectory,
                                                                                                       Nslices,
                                                                                                       len_sub_trajectories))

    print("The full trajectory was long {} ps while the sub trajectories are of {} ps\n".format(len_full_trajectory  * dt,
                                                                                              len_sub_trajectories * dt))
   
    
    # Now get the time in PICOSECOND for the subtrajectories
    t = np.arange(len_sub_trajectories) * dt
    
    # Get the Mean Square Displacement in ANGSTROM^2
    MSD  = np.zeros(len_sub_trajectories, dtype = type(dt))
    MSD_all = np.zeros((Nslices, len_sub_trajectories), dtype = type(dt))
    
    # Cycle on the number of subtrajectories
    for i in range(Nslices):
        print("\n\nSUBTRAJECTORY #{}".format(i))
        # Get the ase atoms slices for the subtrajectory
        start_index, final_index = i * len_sub_trajectories, i * len_sub_trajectories + len_sub_trajectories
        ase_atoms_slice = all_ase_atoms[start_index:final_index]
        if debug:
            print('\nStart index {} final index {}'.format(start_index, final_index))
            print('N slices analyzed {}'.format(len(ase_atoms_slice)))
            
        # Get the MSD(t) for the current slice
        mean = get_MSD(ase_atoms_slice)
        MSD[:] += mean[:]
        MSD_all[i,:] = mean[:]
        
    # I sum over Nslices trajectories, now MSD is the average in ANGSTROM^2
    MSD /= Nslices
    
    # Get the standard deviation
    # Get the error in ANGSTROM^2
    error_MSD = np.sqrt(np.einsum('it -> t', (MSD_all[:,:] - MSD[:])**2) /(Nslices - 1))/np.sqrt(Nslices)
    
    
    t2=time.time()
    print('\n\nTIME ELAPSED FOR COMPUTING THE MSD(t) is {} sec\n'.format(t2 - t1))

    # Save all the results
    save_plot_results(t, MSD, error_MSD)