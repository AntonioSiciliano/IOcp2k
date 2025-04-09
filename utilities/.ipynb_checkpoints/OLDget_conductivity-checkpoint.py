import numpy as np
import ase, ase.io
from matplotlib.ticker import MaxNLocator
from ase import Atoms
import os, sys
import matplotlib
import matplotlib.pyplot as plt
import copy
from mpi4py import MPI
import time
# import psutil

KELVIN_TO_HA = 3.1668E-6
ANGSTROM_TO_BOHR = 1.88973
AU_TIME_TO_PICOSEC = 2.4188843265864003e-05
PICOSECOND_TO_AU_TIME = 1/AU_TIME_TO_PICOSEC


def get_initial_current(atoms):
    """
    GET THE INITIAL CURRENT
    =======================

    The units are ANGSTROM/PICOSECOND
    """
    # Get the total number of atoms
    N_at_tot = len(atoms.positions[:,0])

    # Get all the chemical symbols len N_at_tot
    chem_symb = np.asarray(atoms.get_chemical_symbols()).ravel()

    # Get a mask that selects only the atoms that contribute to the current len N_at_tot
    selected_atoms = np.arange(N_at_tot)[np.isin(chem_symb, [myatom for myatom in types])]

    # Get the oxidations charges for all  the atoms len N_at_tot
    ox_charges = get_ox_charges(N_at_tot, chem_symb)

    # Only the nonzero oxidation charges contrbute to the conductivity
    return np.einsum('i, ia -> a', ox_charges[ox_charges != 0.], atoms.get_velocities()[selected_atoms,:])

def get_ox_charges(N_at, chemical_symbols):
    """
    GET THE OXISATION CHARGES FOR THE ATOMS THAT ARE INVOLVED 
    =========================================================
    """
    # Prepare the array with the oxidation charges
    ox_charges = np.zeros(N_at)
    
    for i, atm_type in enumerate(types):
        ox_charges[np.where(chemical_symbols == atm_type)] = q_types[i]

    return ox_charges
    
def get_conductivity(debug = True, verbose = True, using_pbc = True):
    """
    RETURNS THE CONDUCTIVITY
    ========================
    
    Rember that ase use EV, ANGSTROM
    
    It computes the pair correlation function
    
    .. math::  \sigma = \\frac{1}{3 V k T} \int dt \left\langle j(t) j(0) \right\rangle
    
     Paramters:
    ----------
        -type1, type2: the pair of atoms you want to consider in the g(r)
        -dr: the radial spacing in ANGSTROM
        -Nr: the number of points in the grid

    Returns:
    --------
        -r: np.array, with the radial grid
        -g: np.array, the pair correlation function
    """
    if debug:
        print('Hello from {}'.format(rank))
    comm.Barrier()

    # Read all the ase atoms objects
    all_ase_atoms = ase.io.read(dir_ase_atoms, index = ':')

    # Time in PICOSECOND
    t = dt * np.arange(len(all_ase_atoms))
    
    # Get the ase_objects
    if rank == 0:
        print("Get the conductivity for {} with ox charges {}\n".format(types, q_types))
        print("Splitting {} ase atoms into chunks among {} processors\n".format(len(all_ase_atoms), size))
        ase_atoms_chunks = [all_ase_atoms[i::size] for i in range(size)]
    else:
        ase_atoms_chunks = None

    atoms_chunks = comm.scatter(ase_atoms_chunks, root = 0)

    print('=> RANK {} processes {} ase atoms'.format(rank, len(atoms_chunks)))
    comm.Barrier()

    # Get the initial velocity in ANGSTROM/PICOSECOND
    j_t0 = get_initial_current(all_ase_atoms[0])
    
    comm.Barrier()
    
    # Get all the scalar products ANGSTROM^2/PICOSECOND
    j_j_prod = 0.0
        
    for i, atoms in enumerate(atoms_chunks):
        if verbose:
            if rank == 0:
                if i % 500 == 0:
                    print("\n==> MASTER DOING SNAPSHOT #{}".format(i))
        # Get the total number of atoms
        N_at_tot = len(atoms.positions[:,0])

        # Get all the chemical symbols len N_at_tot
        chem_symb = np.asarray(atoms.get_chemical_symbols()).ravel()

        # Get the checmial symbols len N_at_tot
        ox_charges = get_ox_charges(N_at_tot, chem_symb)

        # Get a mask that selects only the atoms that contribute to the current len N_at_tot
        selected_atoms = np.arange(N_at_tot)[np.isin(chem_symb, [myatom for myatom in types])]

        #  Get the current in ANGSTROM/PICOSECOND
        j_t = np.einsum('i, ia -> a',  ox_charges[ox_charges != 0.], atoms.get_velocities()[selected_atoms,:])
        # Get the PICOSECOND ANGSTROM^2/PICOSECOND^2 1/(KELVIN * ANGSTROM^3)
        j_j_prod += dt * np.einsum('a, a', j_t, j_t0) /(3 * T * atoms.get_volume() * len(atoms_chunks))
        # Get the 1/(KELVIN * PICOSECOND * ANGSTROM)

    # Convert in HARTREE UNITS
    j_j_prod *= 1/(KELVIN_TO_HA * PICOSECOND_TO_AU_TIME * ANGSTROM_TO_BOHR)
    comm.Barrier()
    if debug:
        print('RANK {}| JJ CHUNK \n{}'.format(rank, j_j_prod))
        # print(rdf_chunks)

    # Now we will gather all the correlation functions
    if rank == 0:
        sigma = np.empty(size, dtype = type(dt)) 
    else:
        sigma = None

    # Gather all arrays at rank 0
    comm.Gather(j_j_prod, sigma, root=0)
    if debug:
        if rank == 0:
            print('FINAL RANK {}\n{}'.format(rank, sigma))
            print("\n\nELAPSED TIME {}".format(time.time() - t1))
    if rank == 0:
        with open("sigma_.txt", "w") as file:
            file.write(f"#COND in HARTREE UNITS\n")
            file.write("{}".format(np.sum(sigma)/len(sigma)))


if __name__ == '__main__':
    """
    A PYTHON SCRIPT TO GET THE CONDUCTIVITY
    =======================================

    Uses ANGSTROM PICOSECONDS units as in ASE
    """
    comm = MPI.COMM_WORLD  # Initialize MPI communicator
    rank = comm.Get_rank()  # Process ID
    size = comm.Get_size()

    matplotlib.use('tkagg')
    if rank == 0:
        if len(sys.argv) == 1:
            raise ValueError('USAGE IS python3 get_conductivity dir_to_ase_atoms dt T type1 q1 type2 q2...')

    # THE DIR TO THE ASE ATOMS OBJECT
    dir_ase_atoms = sys.argv[1]
    
    # FEMPTOSECOND
    dt = float(sys.argv[2])
    # NOW IN PICOSWECOND
    dt *= 1e-3
    
    # TEMPERATURE KELVIN
    T = float(sys.argv[3])

    if len(sys.argv[4:]) == 0:
        raise ValueError('Please specify type1 q1 etc')
    
    # GET THE ATOMIC TYPES WITH THE OXIDATION CHARGES
    types = []
    q_types = np.zeros(len(sys.argv[4:])//2)
    for i in range(len(sys.argv[4:])//2):
        types.append(sys.argv[4 + 2*i])
        q_types[i] = int(sys.argv[4 + 2*i + 1])

    # print(types)
    # print(q_types)
    # Get the timing
    if rank == 0:
        t1 = time.time()

    # Get the result
    get_conductivity()

    
    