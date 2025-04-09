import numpy as np

import ase, ase.io
from ase import Atoms

from julia import Julia
jl = Julia(compiled_modules=False)  # Avoid precompile issues

# Import a Julia module
from julia import Main

Main.include("/home/antonio/IOcp2k/utilities/conductivity/time_correlation.jl")

import os, sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import scipy, scipy.optimize, scipy.fft

import copy

from mpi4py import MPI

import time

ANG2_PS_TO_SI = 1e-8
ANG_PS_TO_SI = 1e+2
KELVIN_TO_EV = 8.6173e-5
_e_charge_ = 1.602176634


def get_Vcom_from_ase_atoms(atoms):
    """
    GET THE CENTER OF MASS VELOCITIES
    =================================

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


def get_J_from_ase_atoms(atoms, atom_types, atom_qs, subtract_vcom = True):
    """
    GET THE CURRENT
    ===============

    Get the total current for the single snapshopts ANGSTROM/PICOSECOND

    ..math: J(t) = \sum_{i=1}^{N} q_{i} v_{i}(t)
    
    Parameters:
    -----------
        -atoms: an ase atoms object
        -atom_types: list of string, the selected atoms types
        -atom_qs: np.array, the oxidation charge for each atom
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
    selected_qs    = []
    for i, target_atom in enumerate(atom_types):
        # Select the atomic indices corresponding to the atoms that I want
        my_sel = np.arange(N_at_tot)[np.isin(chem_symb, [target_atom])]
        # Select the atoms contributing to the ionic conductivity
        selected_atoms.append(my_sel)
        # Get the corresponding oxidations charges
        selected_qs.append([atom_qs[i]] * len(my_sel))
    # Transform everything in array
    selected_atoms = np.asarray(selected_atoms).ravel()
    selected_qs    = np.asarray(selected_qs, dtype = int).ravel()
    if debug:
        print('sel atoms', chem_symb[selected_atoms])
        print('sel qs   ', selected_qs)

    # Get the velocities of the selected atoms
    selected_velocities = atoms.get_velocities()[selected_atoms,:]
    if debug:
        print(atoms.get_velocities()[-8:,:] - selected_velocities)
    
    if subtract_vcom:
        # Subtract the COM velocities
        selected_velocities[:,:] -= get_Vcom_from_ase_atoms(atoms)

    J = np.einsum('i, ia -> ia', selected_qs, selected_velocities)

    if debug:
        print('q', selected_qs)
        print('J', J)
        print('V', selected_velocities)

    J_all = np.einsum('ia -> a', J)

    return J_all

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
    # Check the execution time
    t1 = time.time()
    debug = False

    if len(sys.argv[1:]) < 2:
        raise ValueError("The usage is python prepare_inputs.py dir_ase_atoms dt at1 at2")
    
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
    print("\n========CONDUCTIVITY ANALYSIS========")
    print("\nReading ase atoms...")
    all_ase_atoms = ase.io.read(dir_ase_atoms, index = ':')
    print("Finish to read ase atoms")
    print("Ionic conductivity for {}".format(selected_atoms))
    print("with ox charges {}".format(selected_qs))
    print("Number of frames is {} the T {} ps".format(len(all_ase_atoms), len(all_ase_atoms) * dt))

    # Get all the currents # Angstrom/picosecond
    all_J = np.zeros((len(all_ase_atoms), 3), dtype = type(dt))
    # Picoseconds
    times = np.arange(len(all_ase_atoms)) * dt

    for i in range(len(all_ase_atoms)):
        all_J[i] = get_J_from_ase_atoms(all_ase_atoms[i], selected_atoms, selected_qs)

    # Get the julia windowed average (BRUTE FORCE)
    corr, err = Main.get_time_correlation_vector(all_J, len(all_ase_atoms))


    
    # The J J correlation in Fourier
    J_omega  = scipy.fft.fft(all_J[:,:], axis = 0)

    omega    = scipy.fft.fftfreq(len(all_ase_atoms), dt * 1e-12)
    J2_omega = np.einsum('ia, ia -> i', np.conjugate(J_omega), J_omega)
    C_t = scipy.fft.ifft(J2_omega)

    if np.imag(C_t).max() > 1e-5:
        print("Discarting imaginary part...")
    C_t_real = np.real(C_t)

    matplotlib.use('tkagg')
    # Create base plot
    fig, ax1 = plt.subplots()
    
    # Plot on left y-axis
    print(err)
    ax1.errorbar(times[:-1], corr[:-1], yerr = err[:-1], color='k', label = "Julia windowed")
    ax1.legend()
    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    mask = times < times[-1]/2
    ax2.plot(times[mask], C_t_real[mask], color = 'red', label = "python Fourier")
    # ax2.plot(times[mask], C_t_real[~mask][::-1],  ls = ':', color = 'purple', label = "python Fourier")
    ax2.legend(loc = 'upper left')
    plt.show()

    print(dt * np.sum(corr))
    print(dt * np.sum(C_t_real[mask]))
    print(dt * np.sum(corr) /(dt * np.sum(C_t_real[mask])))

    plt.plot(corr[mask] /C_t_real[mask])
    plt.show()
    
    
