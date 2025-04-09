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

def get_Rcom_from_ase_atoms(atoms):
    """
    GET THE CENTER OF MASS POSITIONS
    ================================

    Parameters:
    -----------
        -atoms: an ase atoms object

    Returns:
    --------
        -R_com: np.array with shape 3, the center of mass positions
    """
    # Get all the chemical symbols
    chem_symb = np.asarray(atoms.get_chemical_symbols()).ravel()

    # Get the masses
    masses = np.asarray(atoms.get_masses())

    # Get the center of mass position
    R_com = np.einsum('i, ia -> a', masses, atoms.positions[:,:]) /np.sum(masses)

    return R_com

def get_selected_positions(atoms, atom_type, subtract_com = True):
    """
    GET THE POSITIONS OF THE selected_atom_type ATOMS
    ====================================

    Parameters:
    -----------
        -atoms: an ase atoms object
        -atom_type: string, the selected atom type

    Returns:
    --------
        -selected_positions: np.array with shape (N_at_selected, 3), the positions of the selected atoms shifted by the center of mass positions
    """
    # Get the total number of atoms
    N_at_tot = len(atoms.positions[:,0])
    
    # Get all the chemical symbols
    chem_symb = np.asarray(atoms.get_chemical_symbols()).ravel()
    # Select only the atoms I want
    selected_atoms = np.arange(N_at_tot)[np.isin(chem_symb, [atom_type])]

    # Get the positions of the selected atoms
    selected_positions = atoms.positions[selected_atoms,:]
    
    if subtract_com:
        # Get the center of mass postions
        Rcom = get_Rcom_from_ase_atoms(atoms)

        # Subtract the COM positons
        selected_positions[:,:] -= Rcom[:]

    return selected_positions

def get_MSD(m, N, atom_type, debug = True, verbose = True):
    """
    RETURNS THE MEAN SQUARE DISPLACEMENT
    ====================================
    
    Rember that ase use EV, ANGSTROM
    
    We compute the Mean Square Displacement as a windowed average, i.e. we compute MSD for all possible lag times

    .. math::  MSD(t) =\\frac{1}{N} \sum_{i=1}^{N} \\frac{1}{6} |r_i(t) - r_i(0)|^2

    where N is the number of targeted atoms

    Parameters:
    -----------
        -m: int, the corresponding time is dt * m PICOSECOND
        -N: int, the full length of the trajectory (N * dt is the time length in PICOSECOND)
        -atom_type: string, the selected atom type
        -debug: bool
        -verbose: bool

    Returns:
    --------
        -MSD_average: float, the value of the MSD(m * dt)
        -MSD_error:   float, the value of the error on MSD(m * dt)
    """
    # The mean square displacement for all the possible starting positions
    MSD_chunk = np.zeros(N - m, dtype = float)

    # In this way the sum goes from 0 to N - m - 1 (python convention)
    for k in np.arange(0, N - m):

        # Get the position in ANGSTROM (N_at_selected, 3) at step k
        R_k = get_selected_positions(all_ase_atoms[k], atom_type)
        
        # Get the position in ANGSTROM (N_at_selected, 3) at step k + m
        R_k_plus_m = get_selected_positions(all_ase_atoms[k + m], atom_type)
        
        # The number of selected atoms
        N_at_selected = len(R_k_plus_m[:,0])
        
        # The distance travelled ANGSTROM for each of the selected atoms (N_at_selected, 3)
        d = R_k_plus_m[:,:] - R_k[:,:]
        
        # Get the scalar product squared
        # First sum on the Cartesian coordinates to get the scalar product
        # Second sum on the number of selected atoms
        MSD_chunk[k] = np.sum(np.einsum('ia, ia -> i', d[:,:], d[:,:])) /(6. * N_at_selected)

    # Get average and error
    MSD_average = np.sum(MSD_chunk[:]) / (N - m)

    MSD_error = np.sqrt(np.sum(MSD_chunk[:] - MSD_average)**2) / (N - m)
    
    return MSD_average, MSD_error
        

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
    fracs = [0.2]
    for ic, frac in  enumerate(fracs):
        # Select time larger than t_tol
        t_tol_min, t_tol_max = x[len(x)//2] - x[-1] * frac, x[len(x)//2] + x[-1] * frac
        print(x[-1], t_tol_max, t_tol_min)
        mask = (t_tol_min < x)  & (x < t_tol_max)
        # Make the linear fit, if the MSD at large time is linear then the slope is exactly the diffusion constant
        slope1, intercept1 = np.polyfit(x[mask], y[mask] , 1)
        data_fit, cov = scipy.optimize.curve_fit(f, x[mask], y[mask], sigma = z[mask], absolute_sigma = True)
        slope, intercept = data_fit[0], data_fit[1]
        err_slope = np.sqrt(cov[0][0])
        # ax.plot(x[mask], x[mask] * slope + intercept, lw = 3, color = my_colors[ic],
                # ls = ':', label = "D={:.2e} {:.2e} m$^2$/s".format(slope * ANG2_PS_TO_SI, err_slope * ANG2_PS_TO_SI))
        ax.plot(x[mask], y[mask], lw = 3, color = my_colors[ic])
        # Mark the time after which we made the linear extrapolation
        ax.axvline(t_tol_max, color = my_colors[ic], ls = ":", alpha = 0.3)
        ax.axvline(t_tol_min, color = my_colors[ic], ls = ":", alpha = 0.3)
        # print("\nTmin={:.3f}ps D={:.4e} {:.4e} m$^2$/s".format(t_tol, slope* ANG2_PS_TO_SI, err_slope * ANG2_PS_TO_SI))
        # chisqr = np.sum((y[mask] - f(x[mask], slope, intercept))**2/z[mask]**2)
        # dof = len(y[mask]) - 2
        # print("Reduced chi^2 {}".format(chisqr/dof))
    
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
        ax.fill_between(x, dy_dx - err_dy_dx, dy_dx + err_dy_dx, color = 'red', alpha=0.3)
    ax.set_xlabel('Time [ps]', size = 15)
    ax.set_ylabel('$\\frac{d MSD(t)}{d t}$ [Angstrom$^2$/ps]', size = 15)
    ax.tick_params(axis = 'both', labelsize = 12)
    plt.tight_layout()

    plt.show()

    with open("MSD_{}.txt".format(selected_atom_type), "w") as file:
        file.write(f"#t [ps] MSD [ANGSTROM^2] err MSD [ANGSTROM^2]\n")
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
        -selected_atom_type: string, atomic type
        -dt: the time step in FEMPTOSECOND
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
    selected_atom_type = sys.argv[2] 
    # The time steps in FEMPTOSECOND
    dt = float(sys.argv[3])
    # Now in PICOSECOND
    dt *= 1e-3 
    
    # Read all the ase atoms objects
    print("\n========MDS ANALYSIS========")
    print("\nReading ase atoms")
    all_ase_atoms = ase.io.read(dir_ase_atoms, index = ':')
    all_ase_atoms = all_ase_atoms[:500]
    print("Finish to read ase atoms")
    print("Number of frames is {} the T {} ps".format(len(all_ase_atoms), len(all_ase_atoms) * dt))
    
    # The size of the full trajectory
    len_full_trajectory = len(all_ase_atoms)
   
    # Now get the time in PICOSECOND for the trajectory
    t = np.arange(len_full_trajectory) * dt
    
    # Get the Mean Square Displacement in ANGSTROM^2
    MSD = np.zeros(len_full_trajectory, dtype = float)
    error_MSD = np.zeros(len_full_trajectory, dtype = float)
    
    # Cycle on the possible time distances (lag times)
    for m in np.arange(len_full_trajectory):
        # Get the MSD(t) for the current lag time
        mean, error = get_MSD(m, len_full_trajectory, selected_atom_type)
        MSD[m] = mean
        error_MSD[m] = error
    
    
    t2=time.time()
    print('\n\nTIME ELAPSED FOR COMPUTING THE MSD(t) is {} sec\n'.format(t2 - t1))

    # Save all the results
    save_plot_results(t, MSD, error_MSD)