import numpy as np
import ase
from matplotlib.ticker import MaxNLocator
from ase import Atoms
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import AtomicSnap
from AtomicSnap import AtomicSnapshots
import copy
from mpi4py import MPI


def get_pair_correlation_function(ase_atoms, type1, type2, dr, Nr):
    """
    RETURNS THE PAIR CORRELATION FUNCTION
    =====================================
    
    Rember that ase use EV, ANGSTROM
    
    It computes the pair correlation function
    
    .. math::  g(r) = \\frac{V}{4 \pi r^2 N^2} \left\langle \delta(r - R_{IJ}) \right\rangle
    
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
    if not __JULIA__:
        raise ValueError('You need Julia')

    comm = MPI.COMM_WORLD  # Initialize MPI communicator
    rank = comm.Get_rank()  # Process ID
    size = comm.Get_size()

    # Split the list into chunks
    if rank == 0:
        print("Get the pair correlation function for {} {}".format(type1, type2))
        print("Splitting into chunks")
        ase_atoms_chunks = [ase_atoms[i::size] for i in range(size)]
        # Check that the maxium is compatible with the PBC
        for i in range(3):
            if dr * Nr > ase_atoms[0].unit_cell[i,i]:
                raise ValueError('Reduce the number of points Nr or reduce the dr')
    else:
        ase_atoms_chunks = None

    ase_atoms_chunks = comm.scatter(ase_atoms_chunks, root = 0)
        
    # Exclude the zero ANGTROM
    r_grid = dr * np.arange(1, Nr + 1)
    # RDF
    g_r = np.zeros(len(r_grid), dtype = type(dr))
    

    def generate_mask(input_list, target_elements):
        mask = [elem in target_elements for elem in input_list]
        return np.asarray(mask, dtype = bool)


    for ir, r in enumerate(r_grid):
        for isnap, _ in enumerate(ase_atoms_chunks):
            # Total number of atoms for the current snapshots
            N_at        = ase_atoms_chunks[isnap].get_number_of_atoms()
            # Get the distances ANGSTROM
            distances   = ase_atoms_chunks[isnap].get_all_distances(mic = True)
            # Get the atomic types
            atoms_types = ase_atoms_chunks[isnap].get_chemical_symbols()
            # Get the mask to select only type1 and type2 atoms
            mask = generate_mask(atoms_types, [type1, type2])

            # In this way the julia code runs only on a subset of atoms
            N_at_mask   = np.arange(N_at)[mask]
            # We average on the snapshots
            g_r[ir]     += julia.Main.pair_correlation_function(type1, type2, r, dr, N_at_mask, atoms_types, distances, np.linalg.det(ase_atoms[isnap].cell), N_at)

    # Divide by the number of snapshots
    g_r[:] /= len(ase_atoms_chunks)

    # Rank 0 will gather all arrays into an (size x len(r_grid)) matrix
    if rank == 0:
        final_g_r = np.empty((size, len(r_grid)), dtype = type(dr))
    # Gather data from all processes
    comm.Gather(g_r, final_g_r, root=0)
    
    if rank == 0:
        return r_grid, np.einsum('sr -> r', final_g_r) /size