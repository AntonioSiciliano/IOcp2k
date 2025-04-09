import numpy as np

import ase, ase.io
from ase import Atoms

import os, sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import copy

from mpi4py import MPI

import time

ANG_PS_TO_SI = 1e+2
KELVIN_TO_EV = 8.6173E-5
e_charge = 1.602176634


def get_ox_charges(N_at_tot, chemical_symbols):
    """
    GET THE OXIDATION CHARGES FOR THE ATOMS THAT ARE INVOLVED 
    =========================================================

    Parameters
    ----------
        -N_at_tot: the total number of atoms in the simulation box
        -checmical_symbols: np.array of strings with chemical symbols one for each atom in the simulation box

    Returns:
    --------
        -ox_charges: np.array of len N_at_tot with zeros only for the atoms that do not contrbute to the current
    """
    # Prepare the array with the oxidation charges
    ox_charges = np.zeros(N_at_tot)
    
    for i, atm_type in enumerate(types_q.keys()):
        ox_charges[np.where(chemical_symbols == atm_type)] = types_q[atm_type]

    # print(ox_charges)
    # stop

    return ox_charges

def get_selected_J(atoms):
    """
    GET THE CURRENT OPERATOR
    ========================

    Parameters:
    -----------
        -atoms: an ase atoms object

    Returns:
    --------
        -J_all: a np.array with shape 3, the total current
    """
    # Get the total number of atoms
    N_at_tot = len(atoms.positions[:,0])
    
    # Get all the chemical symbols
    chem_symb = np.asarray(atoms.get_chemical_symbols()).ravel()
    # print(chem_symb)
   

    # Get the oxidation charges array, it has zero on the atoms that do not contribute to the total currrent
    ox_charges = get_ox_charges(N_at_tot, chem_symb)
    # print(ox_charges)
    # print(ox_charges != 0.)
    # stop

    J = np.einsum('i, ia -> ia', ox_charges[ox_charges != 0.], atoms.get_velocities()[ox_charges != 0.,:])

    J_all = np.einsum('ia -> a', J)
    # print(J)
    # stop
    return J_all

def get_JJ(ase_atoms_slice, debug = True, verbose = True):
    """
    RETURNS THE CURRENT CURRENT CORRELATION FUNCTION
    ================================================
    
    Rember that ase use EV, ANGSTROM
    
    We compute the current current correlation on one trajectory

    .. math::  \vec{J}(t) \cdot \vec{J}(0)

    Parameters:
    -----------
        -ase_atoms_slice: a list of ase atoms object
        -debug: bool
        -verbose: bool

    Returns:
    --------
        -JdotJ_chunk: the J(t) J(0) scalar product
    """

    # The mean square dispacement for this slice of the trajectory
    JdotJ_chunk = np.zeros(len_sub_trajectories, dtype = type(dt))
    
    # Get the first current for the selected atoms, the shape is (3)
    J_0 = get_selected_J(ase_atoms_slice[0])
    
    for i, atoms in enumerate(ase_atoms_slice):
        # Get the current for the selected atoms, the shape is (3)
        J_t = get_selected_J(atoms)
        # Get the scalar product J(t)J(0) 
        JdotJ_chunk[i] = J_t.dot(J_0)
    
    return JdotJ_chunk
        

def save_plot_results(x, y, z):
    """
    PLOT THE RESULTS
    ================
    """
    # with open("_{}.txt".format(type1), "w") as file:
    #     file.write(f"#t [ps] MSD [ANGSTROM^2]\n")
    #     for i, j in zip(x, y):
    #         file.write(f"{i} {j}\n")
    
    # Width and height
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(x, y, lw = 3, color = 'blue', label = '<J(t)J(0)>')
    ax.fill_between(x, y - z, y + z, color='blue', alpha=0.3)
    for t_tol in  np.linspace(0.1,0.9, 4) * x[-1]:
        mask = x > t_tol
        cond = np.sum(y[mask] * (x[1] - x[0]))
        # ax.plot(x[mask], x[mask] * slope + intercept, lw = 3, ls = ':', label = "D={:.4e} m$^2$/s".format(slope* ANG2_PS_TO_SI))
        print("Tmin={:.1f}ps sigma={:.4e} S/m".format(t_tol, cond * e_charge * 1e+3))
    
    ax.set_xlabel('Time [ps]', size = 15)
    ax.set_ylabel('<J(t)J(0)>/(3 V k T) [1/(eV Angstrom ps$^2$]', size = 15)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.legend(fontsize = 12)
    plt.show()


def get_average_volume(ase_atoms):
    """
    GET THE AVERAGE VOLUME in ANGTROM^3
    ===================================
    """
    volume = 0.
    volumes = np.zeros(len(ase_atoms))
    for i, atom in enumerate(ase_atoms):
        volume += atom.get_volume()
        volumes[i] = atom.get_volume()

    volume /= len(ase_atoms)
    err_volume = np.sum((volumes[:] - volume)**2)/len(ase_atoms)
    return volume, err_volume


if __name__ == '__main__':
    """
    A PYTHON SCRIPT TO GET THE CONDUCTIVITY
    =======================================

    The units used are ANGSTROM and PICOSECONDS. 

    Paramters:
    ----------
        -dir_ase_atoms: the ase atoms object with our trajectory
        -dt: the time step FEMPTOSECOND
        -T: the temperature in KELVIN
        -Nslices: the number of slices in which we want to cut our trajectory
        -type q: atomic type and oxidation charge

    """
    # Check the execution time
    t1 = time.time()
    matplotlib.use('tkagg')
    debug = True

    if len(sys.argv[1:]) < 2:
        raise ValueError("The usage is python get_conductivity.py dir_ase_atoms dt T Nslices atm1 q1 atm2 q2")
    
    # THE DIR TO THE ASE ATOMS OBJECT
    dir_ase_atoms = sys.argv[1]
    
    # FEMPTOSECOND
    dt = float(sys.argv[2])
    # NOW IN PICOSWECOND
    dt *= 1e-3
    
    # TEMPERATURE KELVIN
    T = float(sys.argv[3])

    # Slices, the number of subtrajectories we want to analyze to get the Mean Square Displacement
    Nslices = int(sys.argv[4])

    if Nslices == 1:
        raise ValueError("The trajectory should be divided in at least 2 subtrajectories")

    if len(sys.argv[5:]) == 0:
        raise ValueError('Please specify type1 q1 etc')
    
    # GET THE ATOMIC TYPES WITH THE OXIDATION CHARGES
    types_q = {}
    for i in range(len(sys.argv[5:])//2):
        types_q.update({sys.argv[5 + 2*i] : int(sys.argv[5 + 2*i + 1])})

    if debug:
        print("You choose to compute the conductivity for {} with ox charge".format(types_q))

    
    # Read all the ase atoms objects
    all_ase_atoms = ase.io.read(dir_ase_atoms, index = ':')
    # The size of the full trajectory
    len_full_trajectory = len(all_ase_atoms)
    if len_full_trajectory % Nslices != 0:
        raise ValueError("Please choose Nslices as a divisor of the full trajectory lenght {}".format(dir_ase_atoms))
        
    # The size of the sub trajectories
    len_sub_trajectories = len(all_ase_atoms)//Nslices
    print("\nDividing the full trajectory of {} snapshots in {} sub trajectories of {} snapshots".format(len_full_trajectory,
                                                                                                         Nslices,
                                                                                                         len_sub_trajectories))

    print("The full trajectory was long {} ps while the sub trajectories are of {} ps".format(len_full_trajectory * dt,
                                                                                              len_sub_trajectories * dt))

    print("I will compute the conducitivty for the following atoms {} with ox states {}".format(types_q.keys(), types_q.items()))
    print()
    
    # Now get the time in PICOSECOND for each trajectory, the length is the one of the subtrajectories
    t = np.arange(len_sub_trajectories) * dt
    # Get the J(t) J(0) scalar product 
    JJ  = np.zeros(len_sub_trajectories, dtype = type(dt))
    # Get also J(t) J(0) scalar product to compute the error 
    JJ_all = np.zeros((Nslices, len_sub_trajectories), dtype = type(dt))
    
    # Cycle on the number of subtrajectories
    for i in range(Nslices):
        print("\n\nSUBTRAJECTORY #{}".format(i))
        # Get the ase atoms slices for the subtrajectory
        start_index, final_index = i * len_sub_trajectories, i *len_sub_trajectories + len_sub_trajectories
        ase_atoms_slice = all_ase_atoms[start_index:final_index]
        if debug:
            print('\nStart index {} final index {}'.format(start_index, final_index))
            print('N slices analyzed ', len(ase_atoms_slice))
            
        # Get the J(t) J(0) scalar product for the current slice
        mean = get_JJ(ase_atoms_slice)
        JJ[:] += mean[:]
        JJ_all[i,:] = mean[:]


    # I sum over Nslices trajectories, Now J(t)J(0) is the average, ANGSTROM^2/PICOSECOND^2
    JJ[:] /= Nslices
    
    # Get the standard deviation for the J(t)J(0) correlator
    error_JJ =  np.sqrt(np.einsum('it -> t', (JJ_all[:,:] - JJ[:])**2) /(Nslices - 1)) /np.sqrt(Nslices)

    # Volume and the error in ANGSTROM^3
    V, error_V = get_average_volume(all_ase_atoms)

    # Now we get the conductivity
    # (ANGSTROM^2/PICOSECOND^2) /(ANGSTROM^3 * EV) = 1 /(PICOSECOND^2 * ANGSTROM * EV)
    sigma_t = JJ[:] /(3 * V * T * KELVIN_TO_EV)
    error_sigma_t = sigma_t * np.sqrt((error_JJ/JJ)**2 + (error_V/V)**2) /(3 * T * KELVIN_TO_EV)

    sigma_t *= e_charge
    error_sigma_t *= e_charge


    # Get the timing
    t2=time.time()
    print('\n\nTIME ELAPSED {} sec\n'.format(t2 - t1))

    # Get the conductiovit
    # cond = JJ

    # Save all the results
    save_plot_results(t, sigma_t, 0*error_sigma_t)