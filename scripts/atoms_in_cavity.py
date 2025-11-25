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
# from filled_cavity import get_types_idx

EXCLUDED_POS = np.ones(3) * -0.1



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




def BRUTE_FORCE_ATOM_IN_CAV(atom, ids_at, ids_cav, radius, tol):
    """
    DEBUG FUNCTION
    """
    all_at = [] 
    for at in ids_at:
        for cav in ids_cav:
            d = atom.get_distance(at, cav, mic = True)
            if d <= radius + tol:
                all_at.append(at)

    return np.asarray(all_at)


def HARD_CORE_DIST_MAT(atom, ids_at, ids_cav):
    """
    DEBUG FUNCTION
    """
    DBG_D = np.zeros((len(ids_at), len(ids_cav)))
    for ia_, i_ in enumerate(ids_at):
        for ja_, j_ in  enumerate(ids_cav):
            DBG_D[ia_,ja_] = atom.get_distance(i_,j_,mic=True)

    return DBG_D


def filled_cavity(atom, cluster_info, radius = 2.0, debug = True, selected_atoms = ["Na", "Cl"], tol = 0.0):
    """
    FILLED CAVITY FUNCTION
    ======================

    Look for the cavities that are filled by selected_atoms types atoms

    Parameters:
    -----------
        -atoms: ase atoms objects containing atoms and cavities
        -cluster_info: a dictionary with all the infos of the clusters cavities
                        id_X
                        id_XX
                        num_clust
                        sizes
                        composition
        -radius: float in ANGSTROM, the radius of the cavity
        -debug: bool
        -selected_atoms: list of str, the types of atoms we want to include in the analysis

    Returns:
    --------
        -all_V_Natom: list of dictionaries with volumes and counts for each 
    """
    # All the ions list of arrays
    # the len of the list depends on the number of selected atoms
    all_ids_atoms = []
    for i, ty in enumerate(selected_atoms):
        all_ids_atoms.append(np.where(np.asarray(atom.get_chemical_symbols()) == ty)[0])
    NTOT_AT = np.sum(np.asarray([1 for item in all_ids_atoms for i in item]))

    if debug and rank  == 0:
        print('   Number of cavities clusters {}'.format(len(cluster_info['sizes'])))
        for i, ty in enumerate(selected_atoms):
            print('   Selected atoms {}'.format(ty))
            print(all_ids_atoms[i])
            
    # The result, list of dictionaries
    # Each dictionary contains "V" and "counts"
    all_V_Natom= []
    
    # Check each cavity cluster if it contains the atoms with the targeted type
    for id_cluster, size in enumerate(cluster_info['sizes']):
        # Volume of the cavity cluster in Angstrom3
        V_cluster = size * 4.0 * np.pi * radius**3 /3.0

        # Get the ids of the cavities forming the cluster
        ids_cavities = cluster_info['id_XX'][id_cluster]

        # How many atoms we have in the cavity for each atom types
        counts_all_ions = np.zeros(len(selected_atoms))

        # Ids of the atoms inside the cavity cluster
        ids_atoms_in_cavity = []
        
        # Check each atomic type
        for i, type_atom in enumerate(selected_atoms):
            
            # Relative distance in Angstrom shape (N_atoms, N_cavities, )
            dist = atom.get_distances(all_ids_atoms[i].astype(int)[:, None], ids_cavities.astype(int)[None,:], mic = True)
            # Reshape  (N_atoms, N_cavities, )
            dist = dist.reshape((len(all_ids_atoms[i]), len(ids_cavities)))

            if rank == 0 and debug:
                dist_dbg = HARD_CORE_DIST_MAT(atom,
                                              all_ids_atoms[i].astype(int),
                                              ids_cavities.astype(int))
                if np.abs(np.sum((dist_dbg - dist).ravel())) > 1e-10:
                    raise ValueError("Dist mat fail")

            # print(dist.shape)
            # Mask bool type (N_atoms, N_cavities, )
            mask_d = (dist >= 0.0) & (dist <= (radius + tol))
            # Cast to int
            mask_d = mask_d.astype(int)
            # print(mask_d)
            # Sum over the cavities, array of len N_atoms
            # How many cavities there are close to the atom a
            mask_cav = np.einsum("ab -> a", mask_d)

            # Append the atoms that are inside the cavity
            ids_atoms_in_cavity.append(all_ids_atoms[i][mask_cav.astype(bool)])

            # Recast in bool then Recast to int then sum to avoid double counting
            counts_all_ions[i] = np.sum(mask_cav.astype(bool).astype(int))
            
            if rank == 0 and debug:
                print("\n   === CLUSTER + ATOM {} DEBUG === ".format(type_atom))
                # print(dist[ids_atoms_in_cavity[-1],:])
                print("   In cluster #{} with size {} and volume {:.1f} ANG3".format(id_cluster, size, V_cluster))
                print("   Ids of the voids")
                print(ids_cavities)
                print("   Ids of atoms type {}".format(type_atom))
                print(all_ids_atoms[i])
                # print("   Distances ANG")
                # print(dist)
                # print("   Mask")
                # print(mask_cav)
                print("   Atoms in the cavity")
                print(ids_atoms_in_cavity[-1])
                res = BRUTE_FORCE_ATOM_IN_CAV(atom,
                                              all_ids_atoms[i].astype(int), 
                                              ids_cavities.astype(int), radius, tol)
                if np.sum(res - ids_atoms_in_cavity[-1]) != 0.0:
                    print('   DEBUG | Atoms in the cavity ')
                    print(res)
                    raise ValueError("The identification of atoms in cavities is wrong")
                print("   There are {} atoms of type {}\n".format(counts_all_ions[i], type_atom))
        #
        if rank == 0 and debug:
            print()

        all_V_Natom.append({"V" : V_cluster,
                            "ids_X" : cluster_info['id_XX'][id_cluster],
                            "counts" : counts_all_ions,
                            "ids_atoms" : np.concatenate([ids.ravel() for ids in ids_atoms_in_cavity])})

    if rank == 0 and debug:
            counts_per_cluster = np.asarray([np.sum(item["counts"]) for item in all_V_Natom])
            print("  I have found {} atoms out of {} in the cavities".format(counts_per_cluster.sum(),
                                                                             NTOT_AT))
            print()
            print()
            print()

    return all_V_Natom
            


if __name__ == '__main__':
    """
    SCRIPT TO STUDY THE CAVITY
    ==========================

    The use is

    mpirun -np 2 python3 atoms_in_cavity.py path/aseatoms.traj cluster_info_path radius exec_path DEBUG type1 type2 ..

    We look for atoms of type1, type2 in the cluster of cavities 
    
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

    # Path to pkl file
    # The info on the clustering of cavities
    cluster_info_path = sys.argv[2]
    
    # choose the radius of the sphere cavity in Angstrom
    RADIUS = float(sys.argv[3])
    
    # The path where the code is executed
    total_path = sys.argv[4]
    
    # The debug variable
    DEBUG = sys.argv[5].lower() in ("true", "1", "yes", "t")

    if len(sys.argv[6:]) > 0:
        selected_atoms = [item for item in sys.argv[6:]]
    else:
        selected_atoms = ['Na', 'Cl']

    ###### END INPUT OF THE CODE ######
    
    # comm.Barrier()
    
    # Split the configurations among processors
    if rank == 0:
        # Get the time
        t1 = time.time()
        # print_mem()
        print("\n================== ATOMS IN CAVITY SCRIPT ==================")
        print("MASTER | We are in {}\nand loading ATOMS {}\nand CLUSTER INFO {}".format(os.getcwd(),
                                                                                        atoms_path,
                                                                                        cluster_info_path))
        print("MASTER | RADIUS {} ANG \nTOTAL PATH {}\n DEBUG {}".format( RADIUS, total_path, DEBUG))
        # Load the trajectory
        atoms = ase.io.Trajectory(atoms_path)
        # Convert to list
        atoms = list(atoms)
        
        # # Load the cells
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
            raise ValueError("Choose a trajectory which is a divisor of {}".format(len(atoms)))

        # Load the PKL file with the clustering info
        # cluster_info is a list
        with open(cluster_info_path, "rb") as file:
            cluster_info =  pickle.load(file)
            
        # Check consistency
        if len(cluster_info) != len(atoms):
            raise ValueError("The len of atoms is different from the len of cluster_info {} {}".format(len(atoms),
                                                                                                       len(cluster_info)))
        
        # Get the chunks size
        chunk_size = len(atoms) //size
        
        # print("MASTER| Looking for empty spheres in {}.\nThe spacing between the spheres is {} Ang\n".format(atoms_path,
        #                                                                                                      2.0 * RADIUS))
        
        # Split the configurations 
        atoms_chunks        = chunk_list(atoms, chunk_size) 
        cluster_info_chunks = chunk_list(cluster_info, chunk_size)
        cells_chunks        = chunk_list(cells, chunk_size) 
        configs_chunks      = chunk_list(list(np.arange(len(atoms))), chunk_size) 
        
        # Clean the memory
        del atoms 
        del cells 
        del cluster_info
        gc.collect()
        # print_mem()
    else:
        atoms          = None
        cluster_info   = None
        cells          = None
        atoms_chunks   = None
        cells_chunks   = None
        cluster_info_chunks = None
        configs_chunks = None
        
    # Scatter the configurations ids that have to be computed to each rank
    # Local thingks
    local_atoms        = comm.scatter(atoms_chunks, root = 0)
    local_cluster_info = comm.scatter(cluster_info_chunks, root = 0)
    local_cells        = comm.scatter(cells_chunks, root = 0)
    local_configs      = comm.scatter(configs_chunks, root = 0)

    # comm.Barrier()
    
    print("RANK {} will compute {} configurations from #{} to #{}".format(rank, len(local_atoms),
                                                                          local_configs[0],
                                                                          local_configs[-1]))

    comm.Barrier()
    

    all_V_N = []
    for i, atom in enumerate(local_atoms):
        if rank == 0 and ((not DEBUG and i % 100 == 0) or (DEBUG and i % 1 == 0)):
            # Get the time
            t2 = time.time()
            
            print("\n\n\n\nMASTER | CONFIGURATION {} | Time spent {:.2f} sec | RAM avail {:.2f} Gb".format(local_configs[i],
                                                                                                  t2 - t1, get_ram()))

        # List of dictionaries
        # with volumes of the cavity number of atoms in the cavity and ids of the atoms inside
        res_V_N = filled_cavity(atom, local_cluster_info[i],
                                radius = RADIUS,
                                debug = DEBUG, selected_atoms = selected_atoms)

        all_V_N.append(res_V_N)

                        
    comm.Barrier()

    # Gather all the results
    V_N = comm.gather(all_V_N, root = 0)

    comm.Barrier()
    
    if rank == 0:
        print("MASTER| Putting all together...")
        # Put everrything back together by unravelling with respect to the size
        # spheres_id = [subitem for item in spheres_id for subitem in item]
        # atoms_with_spheres_complete = [subitem for item in atoms_with_spheres_complete for subitem in item]
        V_N = [subitem for item in V_N for subitem in item]

        # print(spheres_id)
        print("MASTER| Save everything...")
        # save all the files
        DIR = "CAVITY_ATOMS_rc{:.1f}_".format(RADIUS) + atoms_path.split('/')[-1].split('.')[0]
        DIR = os.path.join(total_path, DIR)
        os.mkdir(DIR)
        os.chdir(DIR)

    
        # Save 
        with open('counts_V_N.pkl', 'wb') as f:
            pickle.dump(V_N, f)

        os.chdir(total_path)
