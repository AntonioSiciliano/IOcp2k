import numpy as np
import ase, ase.io
import os, sys, json, pickle
import mpi4py
from mpi4py import MPI
import networkx as nx
import gc, psutil
import time

# Hardcoded variable
EXCLUDED_POS = np.ones(3) * -0.1
PRINT_EVERY_N = 100

def get_types_idx(atoms, my_types = ["Na", "Cl"], excluded_position = EXCLUDED_POS, debug = False):
    """
    GET THE INDICIES OF THE ATOMS FORMING THE CLUSTER
    ==================================================

    Parameters:
    -----------
        -atoms: ase atom object
        -my_types: list of chemical types, the atoms forming the cluster

    Retunrs:
    --------
        -sel_types_idx_0, sel_types_idx_1: np.arrays with the indices of the target atoms
    """
    if len(my_types) != 2:
        raise ValueError("The atomic types must be 2")

    # Get all the atomic types, np.array of len N_AT_TOT
    types = np.asarray(atoms.get_chemical_symbols())

    if rank == 0 and debug:
        if not excluded_position is None:
            print("We are excluding the atoms in {} [Ang]".format(excluded_position))
            

    # Select only the atomic types I want
    sel_types_0 = types == my_types[0]
    # Get the corresponding indices
    sel_types_idx_0 = np.where(sel_types_0)[0]

    # Select only the atomic types I want
    sel_types_1 = types == my_types[1]
    # Get the corresponding indices
    sel_types_idx_1 = np.where(sel_types_1)[0]



    if not excluded_position is None:
        # Exclude the atoms in a given position (useful to study cavity formation)
        idx_to_exclude_0 = np.where((atoms.positions[:,:] == excluded_position).all(axis=1))[0]
        idx_to_exclude_1 = np.where((atoms.positions[:,:] == excluded_position).all(axis=1))[0]
    
        if rank == 0 and debug:
            print("We are excluding the atoms {}".format(my_types[0]))
            print(idx_to_exclude_0)
            print("We are excluding the atoms {}".format(my_types[1]))
            print(idx_to_exclude_1)
    
        # Get rid of the other atoms
        sel_types_idx_0 = [item for item in sel_types_idx_0   if item not in idx_to_exclude_0]
        sel_types_idx_1 = [item for item in sel_types_idx_1   if item not in idx_to_exclude_1]  
        # Back to arrays
        sel_types_idx_0, sel_types_idx_1 = np.asarray(sel_types_idx_0), np.asarray(sel_types_idx_1)
    
        if rank == 0 and debug:
            print("Some atoms have been excluded. These are the ones we will analyze...")
            print("Atom {} #{} indices {}".format(my_types[0], len(sel_types_idx_0), sel_types_idx_0))
            print("Atom {} #{} indices {}".format(my_types[1], len(sel_types_idx_1), sel_types_idx_1))

    return np.asarray(sel_types_idx_0, dtype = int), np.asarray(sel_types_idx_1, dtype = int)



def cluster_analysis(atoms, index = 0, 
                     adjacency = None, 
                     rc_min = 0.5, rc_max = 1, 
                     types = ["Na", "Cl"], indices_types = None,
                     debug = False):
    """
    CLUSTER_ANALYSIS
    ================

    Look for clusters formed by types[0] and types[1]. 

    The atoms are considered to be clustered if the pairwise distances satisfy rc_min < d < rc_max

    Angstrom units are used
    
    
    Parameters:
    ----------
        -atoms: a single ase atoms object, eg an MD snapshots
        -index: int, the id of atoms object
        
        #-my_dist: np.array with size (Nat, Nat), the matrix of distances among all atoms in the atoms object
        -adjacency: np.array of bool with size (Nat_1, Nat_2), where Nat_1 Nat_2 are the number of types[0] and types[1] atoms
        
        -rc_min: float, the minimum distance (Angstrom)
        -rc_max: float, the maximum distance (Angstrom)
        
        -types: list, the atoms forming the cluster
        -indices_types: np.array, the indices of the atoms forming the cluster

        -excluded_position: np.array of size 3, default None. If not None, we look for the atoms that are in this position and we exclude them.
        
        -debug: bool

    Returns:
    --------
        -len_clusters: int, the number of clusters found for this configruation
        -all_cluster_size: np.array, the size of all clusters, eg the number of atoms in each cluster
        -all_cluster_composition: list of string, the composition of all the cluster, eg ["Na1Cl2", "Na2Cl3"]
        -all_at0_ind: a list of len len_clusters, with the indices of atoms types[0] forming the cluster
        -all_at1_ind: a list of len len_clusters, with the indices of atoms types[1] forming the cluster
        -all_at0_at1_ind: a list of len len_clusters with the indices of all the atoms forming the cluster
    """
    # Identify at1 and at2 indices
    sel_types_idx_0, sel_types_idx_1 = indices_types
    if sel_types_idx_0 is None or sel_types_idx_1 is None:
        raise ValueError("Give me the indices of the atoms for which")


    if rank==0 and debug:
        print("Atom {} #{} indices {}".format(types[0], len(sel_types_idx_0), sel_types_idx_0))
        print("Atom {} #{} indices {}".format(types[1], len(sel_types_idx_1), sel_types_idx_1))
        
    
    # Define a graph and its edges by using the atomic indices
    G = nx.Graph()
    for i, at0 in enumerate(sel_types_idx_0):
        for j, at1 in enumerate(sel_types_idx_1):
            if adjacency[i, j]:
                # G.add_edge(f'A{at1}', f'B{at2}')
                G.add_edge('{}{}'.format(types[0], at0), '{}{}'.format(types[1], at1))
                if rank==0 and debug:
                    edge = ('{}{}'.format(types[0], at0), '{}{}'.format(types[1], at1))
                    print("Graph edge {} d {:.4f} Ang".format(edge, atoms.get_distance(at0, at1, mic = True)))
    
    # Find clusters (connected components)
    clusters = list(nx.connected_components(G))
    if rank==0 and debug:
        print()
        print(f"Number of {types[0]}{types[1]} clusters: {len(clusters)}\n")


    # The size of all the cluster
    all_cluster_size = []
    # The compositon of all the cluster
    all_cluster_composition = []
    # the at1 and at2 indices for each cluster
    all_at0_ind, all_at1_ind = [], []
    # the at1 and at2 indices together for each cluster
    all_at0_at1_ind = []
    
    # Store the info about all the cluster
    for i, cluster in enumerate(clusters):
        # Get the size of the cluster
        all_cluster_size.append(len(cluster))

        # The number of atoms0 and atoms1 present in the cluster
        at0_count, at1_count = 0, 0
        # The indices of atoms0 and atoms1 composing the cluster
        at0_ind, at1_ind = [], []
        
        # Get the composition
        for element in cluster:
            # To fix in case we have H-Na clusters, works with Na-Cl or X-X
            if len(my_types[0]) == len(my_types[1]):
                if len(my_types[0]) == 1:
                    ind = int(element[1:])
                elif len(my_types[0]) == 2:
                    ind = int(element[2:])
                else:
                    #To fix in case we have H Na
                    raise ValueError("To fix in case we have Naa Naa") 
            else:
                #TODO
                #To fix in case we have H Na
                raise ValueError("To fix in case we have H Na")
            
            if atoms[ind].symbol == types[0]:
                at0_count += 1
                at0_ind.append(ind)
            elif atoms[ind].symbol == types[1]:
                at1_count += 1
                at1_ind.append(ind)
            else:
                raise ValueError("The type {} does not match any of the input one {}".format(atoms[ind].symbol, types))
                
        # Get the composition of the cluster At1NAt2M with N and M integers
        composition = "{}{}{}{}".format(types[0], at0_count, types[1], at1_count) 
        # Store the compositon
        all_cluster_composition.append(composition)
        # Store the atomic ids
        all_at0_ind.append(np.sort(np.asarray(at0_ind)))
        all_at1_ind.append(np.sort(np.asarray(at1_ind)))

        if rank==0 and debug:
            print(f"Cluster #{i} with {len(cluster)} atoms")
            print(f"and composition {composition}")
            print(f"{types[0]} | ids {at0_ind}")
            print(f"{types[1]} | ids {at1_ind}")
            # Cycle on the atomic indicices of the cluster to check connectivity
            for i in at0_ind:
                for j in at1_ind:
                    print("{}[{}] {}[{}] d {:.3f} Ang".format(types[0], i, types[1], j, atoms.get_distance(i, j, mic = True)))
            print()

        # Put all the indices together
        tmp = np.concatenate((all_at0_ind[-1], all_at1_ind[-1]))
        tmp = np.sort(tmp)
        all_at0_at1_ind.append(tmp)

    
    if rank==0 and debug:
        print("The ids of the atoms forming all the cluster")
        print(all_at0_at1_ind)
        print()
        
    return len(clusters), np.asarray(all_cluster_size), all_cluster_composition, all_at0_ind, all_at1_ind, all_at0_at1_ind


def get_mask(atom, ids0, ids1, min_r, max_r, debug = False):
    """
    GET THE ADIACENCY MASK
    ======================

    Parameters:
    -----------
        -atom: ase atoms, MD snapshot
        -ids0: np array of int, the ids of the atomic type 1
        -ids1: np array of int, the ids of the atomic type 2
        -min_r: float
        -max_r: float
        -debug: bool
    """
    # Get the distances between the atoms ids we selected Angstrom
    d = atom.get_distances(ids0[:, None], ids1[None, :], mic = True)
    d = d.reshape((len(ids0), len(ids1)))

    mask_d = (d > min_r) & (d < max_r)

    if rank == 0 and debug:
        print("  -GET ADJACENCY MATRIX-")
        print("  Type 1 ids", ids0)
        print("  Type 2 ids", ids1)
        print("  DIST MAT [ANG]")
        print(d)
        print()

    # clean the memory
    del d
    gc.collect()

    return mask_d
    
def look_for_clusters(configs, atoms, min_r = 0.9, max_r = 1.0, my_types = ["Na", "Cl"], verbose = True, debug = False):
    """
    CHECK DISTANCES
    ===============

    Check which atoms of types my_types[0] and my_types[1] have distances between min_r and max_r

    Parameters:
    -----------
        -configs: a list of int, configurations to go through
        -atoms: a list of ase atoms objects
        -min_r: float, the minimum distance (Angstrom)
        -max_r: float, the maximum distance (Angstrom)
        -my_types: list, the chemical types forming the clusters
        -verbose: bool,
        -debug: bool,
    """
    # A list of dictiornary. One dictionary for each configuration that has clusters
    results = []


    if len(configs) != len(atoms):
        raise ValueError("The number of snpshots does not coincide with the list of configurations")

    # Check all the atomic configurations
    for i, atom in enumerate(atoms):
        # NB the i is not the configuration index
        # The configuration id is configs[i]

        if rank == 0 and verbose and configs[i] % PRINT_EVERY_N == 0:
            print("\n\n=========================== MASTER | START CONF {} ===========================".format(configs[i]))

        # Get the indices of the selected atoms of len N_at_1 and N_at_2
        # !!!! We also filter out the atoms outside of the box !!!!
        sel_types_idx_0, sel_types_idx_1 = get_types_idx(atom, my_types = my_types, debug = debug)

        # #### MEMORYY UNEFFICIENT PART ####
        # # Get all the distances ANGSTROM
        # d =  atom.get_all_distances(mic = True)

        # # Select only the distances I want, the shape is (N_at_1, N_at_2)
        # sel_d = (d[sel_types_idx_0,:])[:,sel_types_idx_1]

        # # Clean the memory
        # del d
        # gc.collect()
        
        # # Choose only atoms that are in the range min_r max_r, the shape is (N_at_1, N_at_2)
        # mask_d = (sel_d > min_r) & (sel_d < max_r)
        # #### MEMORYY UNEFFICIENT PART ####

        # TEST OF THE NEW CODE
        # # This avoids to compute all the distances and store them
        # # Choose only atoms that are in the range min_r max_r, the shape is (N_at_1, N_at_2)
        # new_mask_d = get_mask(atom, sel_types_idx_0, sel_types_idx_1, min_r, max_r)
        # print(new_mask_d.astype(int) - mask_d.astype(int))

        # Get the adjaceny mask (memory efficient implementation)
        mask_d = get_mask(atom, sel_types_idx_0, sel_types_idx_1, min_r, max_r)

        if rank == 0 and verbose and configs[i] % PRINT_EVERY_N == 0:
            print("\n---------------------------------------")
            print("  CONF {}".format(configs[i]))
            print("  NAT TOT {}".format(len(atom)))
            print("  RAM avail {:.2f} Gb".format(get_ram()))
            print("  <<<<< Min R {:.2f} Ang Max R {:.2f} Ang >>>>>>".format(min_r, max_r))
            print()
            print("  ATOM {}".format(my_types[0]))
            print("  NAT SEL {}".format(len(sel_types_idx_0)))
            print("  ID  SEL\n{}".format(sel_types_idx_0))
            print()
            print("  ATOM {}".format(my_types[1]))
            print("  NAT SEL {}".format(len(sel_types_idx_1)))
            print("  ID  SEL\n{}".format(sel_types_idx_1))
            print()
            # OBSOLETE PART
            # try:
            #     print("Min d {:.2f} [Ang]".format(sel_d[sel_d != 0].min()))
            # except:
            #     print("miss")
            print("---------------------------------------\n")

        # Get the number of cluster the size and its composition
        num_clusters, sizes, compositions, at0_inds, at1_inds, at0_at1_inds = cluster_analysis(atom, index = configs[i],
                                                                                               adjacency = mask_d,
                                                                                               rc_min = min_r, rc_max = max_r,
                                                                                               types = my_types,
                                                                                               indices_types = [sel_types_idx_0, sel_types_idx_1],
                                                                                               debug = debug)
        # Information about the configuration with clusters
        dictionary = {"conf" : configs[i],         # the current configuration
        # OBSOLETE PART  #"dist_mat" : None, # sel_d, #  the matrix distances between my_types[0] and my_types[1] # THIS COULD GIVE RAM ISSUES
                      "id_{}".format(my_types[0]) : at0_inds, # the indices of type0 in the cluster
                      "id_{}".format(my_types[1]) : at1_inds, # the indices of type1 in the cluster
                      "id_{}{}".format(my_types[0], my_types[1]) : at0_at1_inds, # the indices of type0 and type1 in the cluster
                      "num_clust" : num_clusters,   # the number of clusters
                      "sizes" : sizes,              # The size of the cluster
                      "composition" : compositions} # The composition
        
        # Store the dictionary 
        results.append(dictionary)


        if rank == 0 and verbose and configs[i] % PRINT_EVERY_N == 0:
            print("=========================== MASTER | END OF CONF {} ===========================\n\n".format(configs[i]))


    return results

def get_ram():
    """
    GET THE RAM MEMORY Gb
    =====================
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

    
    
if __name__ == '__main__':
    """
    SCRIPT TO STUDY THE FORMATION OF CLUSTERS
    =========================================

    The usage is

    mpirun -np 2 python3 parallel_clusters.py aseatoms.traj rc_min rc_max debug_bool_as_int types1 types2 path_where_execute
    """
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    
    if rank==0 and len(sys.argv[1:]) != 7:
        raise ValueError("mpirun -np 2 python3 path/find_clusters.py atoms.traj rc_min rc_max debug_bool_as_int type1 type2 path_where_execute")

    #########################################
    # Requesed inputs
    # The ase atoms .traj file
    ase_atoms_dir = sys.argv[1]

    # The min distance cutoff in Angstrom
    RCUT_min = float(sys.argv[2])

    # The max distance cutoff in Angstrom
    RCUT_max = float(sys.argv[3])

    # Debug variable 1=True
    DBG = bool(int(sys.argv[4]))

    # The atoms forming the clusters
    my_types = [sys.argv[5], sys.argv[6]]

    # The path where we excute the code
    total_path = sys.argv[7]
    #########################################


    
    if rank == 0:
        print("\n=========================================")
        print("THIS IS THE OPTIMIZED FIND CLUSTER SCRIPT")
        print("=========================================\n\n")

    # Split the configurations among processors
    if rank == 0:
        print("MASTER| Looking for {} clusters in {} within {:.2f} {:.2f} Ang\n".format(my_types, ase_atoms_dir, RCUT_min, RCUT_max))
        
        # Read the ase atoms (only the master does this)
        atoms = ase.io.Trajectory(os.path.join(total_path, ase_atoms_dir))
        atoms = list(atoms)
    
        # Get the total number of configurations
        total_configs = len(atoms)

        # Check that we are working with a number of configurations that is an integer multiple of the processors
        if total_configs % size != 0:
            raise ValueError("Choose a divisor of {}".format(total_configs))

        # The legth of the snapshots that will be ananlyzed
        chunk_size = total_configs //size
        # Split the configurations 
        atoms_chunks   = chunk_list(atoms, chunk_size) 
        configs_chunks = chunk_list(list(np.arange(total_configs)), chunk_size)

        # Clean the memory
        del atoms 
        gc.collect()
    else:
        atoms = None
        atoms_chunks = None
        configs_chunks = None

    # Scatter the configurations ids that have to be computed to each rank
    local_atoms   = comm.scatter(atoms_chunks, root = 0)
    local_configs = comm.scatter(configs_chunks, root = 0)

    print("RANK {} will compute {} configurations from #{} to #{}".format(rank, len(local_configs),
                                                                          local_configs[0], local_configs[-1]))


    comm.Barrier()

    local_results = look_for_clusters(local_configs, local_atoms, my_types = my_types,
                                      min_r = RCUT_min, max_r = RCUT_max,
                                      verbose = True, debug = DBG)

    comm.Barrier()

    print("RANK {} HAS FOUND {} CONFIGURATIONS BONDED".format(rank, len(local_results)))

    # Gather all the results
    results = comm.gather(local_results, root = 0)



    # Save everything
    if rank == 0:
        total_configs_bonded = 0
        
        for i in range(len(results)):
            print("MASTER| PROC {} HAS FOUND {} CONFIGURATIONS BONDED".format(i, len(results[i])))
            total_configs_bonded += len(results[i])
        
        if total_configs_bonded > 0:
            # Get the name of the ase atoms file
            label_ase_atoms = (ase_atoms_dir.split('/')[-1]).split('.')[0]
            # Greate a specific direcory
            DIR_RES = "CLUSTER_{}_rc{:.2f}_{:.2f}_Nc_{}".format(label_ase_atoms, RCUT_min, RCUT_max, total_configs)
            DIR_RES = os.path.join(total_path, DIR_RES)

            # Create the directory to store all the results
            if not os.path.isdir(DIR_RES):
                os.mkdir(DIR_RES)

            # Ravel the gathered list (useful for saving)
            ravel_results = [item for sublist in results for item in sublist]
            # print(ravel_results[0])
            print()
            for item_ind, item in enumerate(ravel_results):
                print("ITEM {}".format(item_ind))
                print(item)
            print()
            
            name_file_pkl = os.path.join(DIR_RES, "CLUSTER_confs_atoms_bonded_rc{:.2f}_{:.2f}_Nc_{}.pkl".format(RCUT_min, RCUT_max, total_configs))
            with open(name_file_pkl, "wb") as file:
                pickle.dump(ravel_results, file)

            name_file_txt = os.path.join(DIR_RES, "full_path_ase_atoms.txt")
            with open(name_file_txt, "w") as file_txt:
                file_txt.write("{}".format(os.path.join(total_path, ase_atoms_dir)))

    
        
        print()
        print("#CONF with CLUSTER {} OUT OF {} PERCENTAGE {}".format(total_configs_bonded,
                                                                     total_configs,
                                                                     total_configs_bonded * 100 / total_configs))
    

        print("All the infos have been saved in")
        print("! {}".format(DIR_RES))




