import ase, ase.io, ase.visualize
import cellconstructor as CC
import cellconstructor.Structure
import numpy as np
import ase.atoms, ase.build
import time
import tkinter as tk
import networkx as nx
import sys
import mpi4py
from mpi4py import MPI
import os, gc
import pickle
import psutil

EXCLUDED_POS = np.ones(3) * -0.1

TIMER = False

def get_ram():
    """
    GET THE RAM MEMORY
    ==================
    """
    mem = psutil.virtual_memory()

    return mem.available/1e+9
    
def print_mem():
    """
    PRINT THE MEMORY
    ================
    """
    print("RAM {:.2f}".format(get_ram()))


def chunk_list(lst, chunk_size):
    """
    DIVIDE THE LIST INTO CHUNKS
    ===========================
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


    

def create_sphere_centers(L, distance, view = False, def_chem_type = "X", verbose = True):
    """
    GENERATE THE SPHERE CENTERS POSITIONS
    =====================================

    We generate the spheres centers in a cubic cell of length L. 
    The lattice constant is given by distance.

    Parameters:
    -----------
        -L: float, the leght of the MD simulation cubic box
        -distance: float, the lattice constant of the cubic superstructure, ie the distance between the sphere centers
        -view: bool
        -def_chem_type: str, the fictitious type of the sphere
    """
    # Create the cubic unit cell
    unit_cell = ase.build.bulk(name = def_chem_type, crystalstructure = "sc", a = distance)
    unit_cell.center()
        
    # Generate the supercell structure to tile the real simulation box
    structure = CC.Structure.Structure()
    structure.generate_from_ase_atoms(unit_cell)
    size = [int(L//distance), int(L//distance), int(L//distance)]
    super_structure = structure.generate_supercell(size, QE_convention = False)
    
    # The ase atoms with the postions of the sphere centers
    sphere_centers = super_structure.get_ase_atoms()
    # Overwrite the cell shape with the one of the MD simulation
    sphere_centers.set_cell(np.eye(3) * L)
    sphere_centers.center()

    if verbose and rank == 0:
        print("\n")
        print("FILLING THE MD BOX WITH CAVITIES")
        print("The lattice spacing is {:.1f} Angstrom".format(distance))
        print("The number of spheres is {}".format(len(sphere_centers)))
        print("The distance matrix in Angstrom")
        print(sphere_centers.get_all_distances(mic = True))
        print("The box has shape {} {} {} Angstrom".format(L, L, L))
        print("COM of the voids Angstrom")
        print(np.einsum('ab -> b', sphere_centers.get_positions()) /len(sphere_centers))
        print("DONE FILLING THE MD BOX WITH CAVITIES")
        print("\n")
        
    if view and rank == 0:
        ase.visualize.view(sphere_centers)

    return sphere_centers




def get_types_idx(atoms, my_type, debug = False):
    """
    GET THE INDICES OF THE ATOMS FORMING THE CLUSTER
    ==================================================

    Parameters:
    -----------
        -atoms: ase atom object
        -my_type: chemical symbol of which we want the corresponding indices

    Retunrs:
    --------
        -sel_types: np.arrays with the indices of the target atoms
    """

    # Get all the atomic types, np.array of len N_AT_TOT
    types = np.asarray(atoms.get_chemical_symbols())

    # Select only the atomic types I want
    sel_types = np.where(types == my_type)[0]

    if rank == 0 and debug:
        # if not excluded_position is None:
        print("\n    We are looking for {} atom".format(my_type))
        print("    The indices are")
        print(np.asarray(sel_types, dtype = int))
        print()

    return np.asarray(sel_types, dtype = int)


def get_mask(atom, ids_sph, ids_at, max_r, min_r = 0, debug = False, selected_atoms = None):
    """
    GET THE ADIACENCY MASK
    ======================

    Given the indices of the atoms we look for their relative distances

    and define an adjacency mask

    We return the id mask of the spheres that are filled if True is filled

    Parameters:
    -----------
        -atom: ase atoms, MD snapshot
        -ids_sph: np array of int, the ids of the spheres
        -ids_at:  np array of int, the ids of the atoms
        -min_r: float
        -max_r: float
        -debug: bool
    """
    if rank == 0 and TIMER:
        ta = time.time()
        
    if selected_atoms is None:
        if rank == 0 and debug:
            print("\n  -The adjacency mask will be computed with all the atoms")
    else:
        # Get the new indices
        sel_ids_at = []
        for chem_typ in selected_atoms:
            sel_ids_at.append(get_types_idx(atom, chem_typ, debug = debug))
        # Concatenate the items in the list of arrays
        sel_ids_at = np.concatenate([item for item in sel_ids_at])
        sel_ids_at = np.asarray(sel_ids_at).ravel()
        # Overwrite the input
        ids_at = sel_ids_at
        if rank == 0 and debug:
            print("\n  -The adjacency mask will be computed with atoms")
            print(selected_atoms)
            print("  -The corresponding indices are {}".format(len(sel_ids_at)))
            print(ids_at)

        
    # Get the distances between the ids we selected Angstrom
    d = atom.get_distances(ids_sph[:, None], ids_at[None, :], mic = True)
    # Reshape to a 2D array (N_sph, N_at)
    d = d.reshape((len(ids_sph), len(ids_at)))
    # Get the distances that are smaller than a cutoff
    mask_d = (d > min_r) & (d < max_r)
        
    # Now sum to counts how many atoms we have in each sphere
    ids_mask = np.einsum('ab -> a', mask_d.astype(int))

    if rank == 0 and TIMER:
        tb = time.time()
        print("MASTER | TIMER | to get distances {}".format(tb - ta))

    if len(ids_mask) != len(ids_sph):
        raise ValueError("The shape of ids mask is not correct")

    if rank == 0 and debug:
        print("  -GET ADJACENCY MATRIX between {} and {} Angstrom-".format(min_r, max_r))
        print("  Spheres ids")
        print(ids_sph)
        print("  Atoms   ids")
        print(ids_at)
        # print("  DIST MAT [ANG] shape{}".format(d.shape))
        # print(d)
        # print("  MASK DIST MAT")
        # print(mask_d)
        print("  Counts on spheres")
        print(ids_mask)
        print("  Counts on sphere as bool type")
        print(~ids_mask.astype(bool))
        print()

    # THE CLEANING WAS SLOWING DOWN THE CODE
    # if rank == 0 and TIMER:
    #     ta = time.time()
    #     # print("MASTER | TIMER | to finish {}".format(tc - tb))
    # # clean the memory
    # del d
    # del mask_d
    # gc.collect()
    # if rank == 0 and TIMER:
    #     tc = time.time()
    #     print("MASTER | TIMER | to clean {}".format(tc - ta))

    return ids_mask 


    
def atoms_in_cavity(atom, L = 1., radius = 2.5, sphere_centers = None, debug = True, def_chem_type = "X", selected_attyp = None):
    """
    ATOMS IN CAVITY
    ===============

    Given an MD snaphot, rappresented by atom, we generate a cubic pattern of spheres that tiles the MD cell.
    Then we look for spheres that are filled with selected_attyp atoms.

    Parameters:
    -----------
        -atom: ase atom object, an MD snapshot
        -L: float, the size of the cubic box of the MD snapshots Angstrom
        -radius: float, the radius of the spheres Angstrom
        -sphere_centers: ase atom with the spheres centers, useful if we analyze NVT simulation where the volume is fixed
        -debug: bool
        -def_chem_type: str, the default chemical type we assign to empty spheres
    """
    # Get the number of atoms
    Nat = len(atom)
    
    # The lenght of the MD simulation box Angstrom
    if L < 1e-3:
        raise ValueError("The cell is zero")

    # The centers of the sphere
    if sphere_centers is None:
        print("No sphere centers is given, building it")
        sphere_centers = create_sphere_centers(L, 2.0 * radius)
        
    # Check the consistency between the sphere centers distance and the radius
    if np.abs(sphere_centers.get_distance(0, 1, mic = True) - 2.0 * radius) > 1e-3:
        raise ValueError("Check the distance between the spheres centers. It must be {}".format(radius))

    # The number of spheres
    Nspheres = len(sphere_centers)
    # The ids of the sphere centers
    all_spheres_id = np.arange(Nspheres)
    
    # The real atoms of the MD snapshots + the center of the spheres
    if rank == 0 and TIMER:
        ta = time.time()
        
    atom_sphere = ase.Atoms(symbols = atom.get_chemical_symbols() + [def_chem_type] * Nspheres,
                            pbc = True, cell = np.eye(3) * L,
                            positions = np.concatenate((atom.positions,	sphere_centers.positions)))
    if rank == 0 and TIMER:
        tb = time.time()
        print("\nMASTER | TIMER | To get the atom sphere objects {}".format(tb - ta))
    

    # Reeturns an array of len Nspheres counting how many atoms are contained in each cavity 
    counts = get_mask(atom_sphere, all_spheres_id + Nat, np.arange(Nat), radius,
                      debug = debug, selected_atoms = selected_attyp)

    if rank == 0 and TIMER:
        tc = time.time()
        print("MASTER | TIMER | To get the counts {}".format(tc - tb))


    del atom_sphere
    gc.collect()

    
    return counts



if __name__ == '__main__':
    """
    SCRIPT TO STUDY THE EFFECTIVE SIZE OF A WATER MOLECULE
    ======================================================

    The use is

    mpirun -np 2 python3 new_cavity.py path/aseatoms.traj calc_type radius exec_path DEBUG

    Given an MD snapshots, we look for the probability that a cavity of volume V is filled with N water molecules

    TODO It can be useful to have an option where we save just the trajectory with only the empty spheres
    
    """
    # total_path = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(total_path)
    
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ###### INPUT OF THE CODE ######
    # Get the ase atoms
    atoms_path = sys.argv[1]
    
    # Get the cells
    # cell_path = sys.argv[2]
    
    # Calc type NVT or NPT
    calc_type = sys.argv[2]
    
    # choose the radius of the sphere in Angstrom
    # The distance between spheres centers will be twice the RADIUS
    RADIUS = float(sys.argv[3])
    
    # The path where the code is executed
    total_path = sys.argv[4]
    
    # The debug variable
    DEBUG = bool(sys.argv[5])
    DEBUG = sys.argv[5].lower() in ("true", "1", "yes", "t")

    if len(sys.argv[6:]) == 0:
        atoms_type_target = ['O']
    else:
        # The atoms we want to check 
        atoms_type_target = [item for item in sys.argv[6:]]

    ###### END INPUT OF THE CODE ######

    if not calc_type in ["NVT", "NPT"]:
        raise ValueError("calc_type msu be either NVT or NPT")

    
    comm.Barrier()
    
    # Split the configurations among processors
    if rank == 0:
        # Get the time
        t1 = time.time()
        # print_mem()
        print("\n================== FILLED CAVITY SCRIPT ==================")
        print("MASTER | We are in {}\nand loading {}".format(os.getcwd(), atoms_path))
        print("MASTER | CALC {} RADIUS {}\n TOTAL PATH {} DEBUG {}".format(calc_type, RADIUS, total_path, DEBUG))
        # Load the trajectory
        atoms = ase.io.Trajectory(atoms_path)
        # Convert to list
        atoms = list(atoms)
        # Load the cells
        cells = [atoms[i].cell for i in range(len(atoms))]
        cells = np.asarray(cells)

        if len(atoms) != len(cells[:,0,0]):
            raise ValueError("Choose correctly the size of the unit cells and the ase atoms traj file")
            
        if np.abs(np.linalg.det(cells[0,:,:])) < 1e-3:
            raise ValueError("The cells are not loaded correctly")

        if cells[0,0,0] != cells[0,1,1] or cells[0,0,0] != cells[0,2,2] or np.any(cells[0,~np.eye(3, dtype=bool)] != 0.0):
            raise ValueError("We can handle only cubic boxes!")

        # Check that we are working with a number of configurations that is an integer multiple of the processors
        if len(atoms) % size != 0:
            raise ValueError("Choose a divisor of {}".format(len(atoms)))
        
        # Get the chunks size
        chunk_size = len(atoms) //size
        
        print("MASTER| Looking for empty spheres in {}.\nThe spacing between the spheres is {} Ang\n".format(atoms_path,
                                                                                                             2.0 * RADIUS))
        # Split the configurations 
        atoms_chunks   = chunk_list(atoms, chunk_size) 
        cells_chunks   = chunk_list(cells, chunk_size) 
        configs_chunks = chunk_list(list(np.arange(len(atoms))), chunk_size) 
        
        # Clean the memory
        del atoms 
        del cells 
        gc.collect()
        # print_mem()
    else:
        atoms          = None
        cells          = None
        atoms_chunks   = None
        cells_chunks   = None
        configs_chunks = None
        
    # Scatter the configurations ids that have to be computed to each rank
    # Local thingks
    local_atoms   = comm.scatter(atoms_chunks, root = 0)
    local_cells   = comm.scatter(cells_chunks, root = 0)
    local_configs = comm.scatter(configs_chunks, root = 0)
    
    print("RANK {} will compute {} configurations from #{} to #{}".format(rank, len(local_atoms),
                                                                          local_configs[0],
                                                                          local_configs[-1]))

    comm.Barrier()


    # # The ids of the empty spheres
    # all_empty_spheres_id = []
    # # All the atoms with the positions of the empty spheres 
    # # (the ones that are filled we put the outside of the box)
    # all_atom_with_spheres = []

    all_counts = []

    if calc_type == "NPT":
        if rank == 0:
            print("MASTER | This is a NPT calculation")
        sphere_centers = None
    
    for i, atom in enumerate(local_atoms):
        
        # if rank == 0 and i % 100 == 0:
        if rank == 0 and ((not DEBUG and i % 100 == 0) or (DEBUG and i % 1 == 0)):
            # Get the time
            t2 = time.time()
            
            print("\n\n\nMASTER | CONFIGURATION {} | Time spent {:.2f} sec | RAM avail {:.2f} Gb".format(local_configs[i],
                                                                                                  t2 - t1, get_ram()))


        if i == 0 and calc_type == "NVT":
            if rank == 0:
                print("\nMASTER | This is a NVT calculation with cubic cell")
                print("         Cell a = {:.3f} b = {:.3f} c = {:.3f} [ANG]".format(local_cells[i,0,0],
                                                                                    local_cells[i,1,1],
                                                                                    local_cells[i,2,2]))
                print("MASTER | RAM avail {:.2f} Gb".format(get_ram()))
            sphere_centers = create_sphere_centers(local_cells[i,0,0], 2.0 * RADIUS, verbose = True)
            if rank == 0:
                print("MASTER | The number of sphere is {}\n".format(len(sphere_centers)))
        
        # Look for atoms in cavity in the MD snapshot
        # counts is a np.array of len Nspheres
        counts = atoms_in_cavity(atom, radius = RADIUS, 
                             sphere_centers = sphere_centers, debug = DEBUG,
                             L = local_cells[i,0,0], selected_attyp = atoms_type_target)

        if rank == 0 and ((not DEBUG and i % 100 == 0) or (DEBUG and i % 1 == 0)):
            print("MASTER | STEP {}".format(i))
            print("MASTER | Counts of {} for each sphere".format(atoms_type_target))
            print(counts)
            print("MASTER | TOTAL Count of {} in the spheres".format(atoms_type_target))
            print(np.sum(counts))
            # print("MASER | Total number of {} atomsms ")
        
        # Append the result
        all_counts.append(counts)

    # Reshape in np array
    all_counts = np.asarray(all_counts).reshape((len(local_atoms), len(sphere_centers)))
    
    comm.Barrier()

    if DEBUG:
        print(rank, all_counts.shape)

    # Gather the results
    if rank == 0:
        print("MASTER | Putting all together...")
        final_counts = np.empty((len(local_atoms) * size, len(sphere_centers)), dtype = type(all_counts[0,0]))
    else:
        final_counts = None
    # Gather all the results
    comm.Gather(all_counts, final_counts, root = 0)

    
    comm.Barrier()
    
    if rank == 0:
        print("MASTER | Total counts shape")
        print(final_counts.shape)
        print("MASTER | Average counts for each sphere")
        print(np.einsum("ij -> j", final_counts) /len(final_counts[:,0]))

        # print(spheres_id)
        print("MASTER| Save everything...")
        # save all the files in a specific dir
        if len(atoms_type_target) == 1:
            if atoms_type_target[0] == 'O':
                DIR = "CAVITY_WATER_O_rc{:.1f}_".format(RADIUS) + atoms_path.split('/')[-1].split('.')[0]
            else:
                DIR = "CAVITY_{}_rc{:.1f}_".format(atoms_type_target[0], RADIUS) + atoms_path.split('/')[-1].split('.')[0]
        else:
            string = "".join(i for i in atoms_type_target)
            DIR = "CAVITY_{}_rc{:.1f}_".format(string, RADIUS) + atoms_path.split('/')[-1].split('.')[0]
            
        DIR = os.path.join(total_path, DIR)
        os.mkdir(DIR)
        os.chdir(DIR)

        np.save("all_counts", final_counts)

        os.chdir(total_path)
