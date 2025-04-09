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

def get_pair_correlation_function(debug = True, verbose = True, using_pbc = True):
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
    if debug:
        print('Hello from {}'.format(rank))
    comm.Barrier()

    # Read all the ase atoms objects
    all_ase_atoms = ase.io.read(dir_ase_atoms, index = ':')
    # Get the ase_objects
    if rank == 0:
        print("Get the pair correlation function for {} {}\n".format(type1, type2))
        print("Splitting {} ase atoms into chunks among {} processors\n".format(len(all_ase_atoms), size))
        ase_atoms_chunks = [all_ase_atoms[i::size] for i in range(size)]
    else:
        ase_atoms_chunks = None

    atoms_chunks = comm.scatter(ase_atoms_chunks, root = 0)

    print('=> RANK {} processes {} ase atoms'.format(rank, len(atoms_chunks)))
    comm.Barrier()
    
    # For each snapshot we will compute an histogram of distances then we will take the average
    all_counts = np.zeros(len(r) - 1, dtype = type(r[0]))
    
    for i, atoms in enumerate(atoms_chunks):
        if verbose:
            if rank == 0:
                if i % 100 == 0:
                    print("\n==> MASTER DOING SNAPSHOT #{}".format(i))
        # Get the total number of atoms
        N_at_tot = len(atoms.positions[:,0])

        # Get all the chemical symbols
        chem_symb = np.asarray(atoms.get_chemical_symbols()).ravel()

        # Get all the distances using pbc or not ANGSTROM
        all_distances = atoms.get_all_distances(mic = using_pbc)
        
        # select only the pair of atoms of which we want the RDF
        selected_atoms1 = np.arange(N_at_tot)[np.isin(chem_symb, [type1])]
        selected_atoms2 = np.arange(N_at_tot)[np.isin(chem_symb, [type2])]
        # Mix the two types of atoms using meshgrid
        mix = np.meshgrid(selected_atoms1, selected_atoms2)

        # Select only the distances of the two atoms kinds we want
        # (len(selecte_atoms_2), len(selected_atoms_1))
        selected_distances = all_distances[mix[0],mix[1]]

        # if debug:
        #     if rank == 0:
        #         # shape are (len(selecte_atoms_2), len(selected_atoms_1))
        #         print("Selected atoms type1") 
        #         # print(selected_atoms1)
        #         # print(mix[0])
        #         print(selected_atoms1.shape)
        #         print("Selected atoms type2") 
        #         # print(selected_atoms2)
        #         print(mix[1].shape)
        #         print(selected_atoms2.shape)
                
        #         print("Selected distances") 
        #         # shape are (len(selecte_atoms_2), len(selected_atoms_1))
        #         print(all_distances[mix[0],mix[1]].shape)
        #         # print(all_distances)

        # Get rid of small distances (equal atoms)
        d = selected_distances[selected_distances > 0.].ravel()

        # Make the histrogram
        counts, x = np.histogram(d, bins = r)
        # Add the hstrogram multiplied by the volume and divided by two so to cosider double counts
        all_counts += atoms.get_volume() * counts/2

    
    rdf_chunks = all_counts /(4 * np.pi * r[1:]**2 * int(len(atoms_chunks)))

    comm.Barrier()
    if debug:
        print('RANK {}| RDF CHUNK \n{}'.format(rank, rdf_chunks))
        # print(rdf_chunks)

    # Now we will gather all the rdf
    if rank == 0:
        rdf = np.empty((size, len(rdf_chunks)))  # Container for all arrays
    else:
        rdf = None

    # Gather all arrays at rank 0
    comm.Gather(rdf_chunks, rdf, root=0)
    if debug:
        if rank == 0:
            print('FINAL RANK {}\n{}'.format(rank, rdf))
            print("\n\nELAPSED TIME {}".format(time.time() - t1))
    if rank == 0:
        # Return the radial cooridnate and the g(r)
        x_values, y_values = r[1:], np.einsum('sr -> r', rdf) /size
        plt.plot(x_values, y_values, label = "g(r) {} {}".format(type1, type2))
        plt.xlabel("r [ANGSTROM]")
        plt.legend(fontsize = 15)
        plt.show()
        # Save to a .daat file (space-separated)
        with open("gr_{}_{}.txt".format(type1, type2), "w") as file:
            file.write(f"#r gr\n")
            for x, y in zip(x_values, y_values):
                file.write(f"{x} {y}\n")
        
        return x_values, y_values
        


if __name__ == '__main__':
    """
    A PYTHON SCRIPT TO GET THE PAIR CORRELATION FUNCTION
    =====================================================

    Uses ANGSTROM units as in ASE
    """
    comm = MPI.COMM_WORLD  # Initialize MPI communicator
    rank = comm.Get_rank()  # Process ID
    size = comm.Get_size()

    matplotlib.use('tkagg')
    if rank == 0:
        if len(sys.argv) == 1:
            raise ValueError('USAGE IS python3 get_pair_correlation_function.py dir_to_ase_atoms type1 type2 dr Nr')

    
    dir_ase_atoms = sys.argv[1]
    type1 = sys.argv[2] 
    type2 = sys.argv[3]
    # The dr step in ANGSTROM
    dr = float(sys.argv[4])
    Nr = int(sys.argv[5])

    # The radial variable in ANGSTROM
    r = dr * np.arange(1, 1 + Nr)

    # Get the timing
    if rank == 0:
        t1 = time.time()

    # Get the result
    get_pair_correlation_function()

    
    