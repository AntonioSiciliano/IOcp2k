import numpy as np

import ase, ase.io

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


import os, sys, json, pickle
from itertools import groupby
matplotlib.use("tkagg")
import pickle
from collections import defaultdict
import json


PRINT_EVERY_N = 100

def filter_atoms(ids_per_conf, info, target_comp = "Na1Cl1"):
    """
    FILTER THE COMPOSITIONS OF THE CLUSTER ACCORDING TO A TARGET COMPOSITION
    ========================================================================
    """
    new_ids_per_conf = []
    # print(info)

    for i in range(len(ids_per_conf)):
        print(info[i]['conf'])

        tmp_list = []
        for c in range(info[i]["num_clust"]):
            if info[i]['composition'][c] == target_comp:
                tmp_list.append(ids_per_conf[i][c])
        new_ids_per_conf.append(tmp_list)

    return new_ids_per_conf
    

def get_results_from_dir(my_dir, debug = False):
    """
    READ THE pkl FILE FROM THE RESULTS DIRECTORY
    ============================================

    Parameters:
    -----------
        -my_dir: the directory with the results in a pkl format
    """
    # Get the pkl file
    pkl_file = None
    for file in os.listdir(my_dir):
        if file.endswith("pkl"):
            pkl_file = os.path.join(my_dir, file)

    if debug:
        print("Reading the file...")
        print(pkl_file)
        
    # All the info at each step for the cluster, size composition atomic ids etc
    all_info = []

    # For each configurations we have for each cluster the ids of the atoms forming it
    all_ids_per_confs = []
    
    with open(pkl_file, "rb") as f:
        # The list with all the info
        all_info = pickle.load(f)

    # Get the number of configurations
    Nc = len(all_info)

    # For each configuration we have the ids of the atoms forming a cluster
    all_ids_per_confs = [all_info[ic]["id_{}{}".format(types[0], types[1])] for ic in range(Nc)]

    return Nc, all_info, all_ids_per_confs




def jacardi_rule(A, B, tol = 0.5):
    """
    SIMILARITY CHECK BETWEEN TWO ARRAYS A, B
    ========================================
    """
    intersection = len(set(A) & set(B))

    union = len(set(A) | set(B))

    index = intersection /union

    if index >= tol:
        return True
    else:
        return False

def NEW_get_results_from_dir(my_dir, debug = False):
    """
    READ THE pkl FILE FROM THE RESULTS DIRECTORY
    ============================================

    Returns a list of dicitonaries like

        clusters[0] is {'ids' : np.array([1,2,3]), 'comp' : string}

    where np.array() contains the indices of the atoms forming the clusters

    and string is the chemical compostion

    Parameters:
    -----------
        -my_dir: the directory with the results in a pkl format

    Returns:
    --------
        -all_clusters: a list of list of dictionaries of len number of MD snapshots
    """
    # For each configurations we have for each cluster the ids of the atoms forming it
    all_clusters = []
    
    # Get the pkl file
    pkl_file = None
    print("I am in {}".format(os.getcwd()))
    print("Looking into {}".format(my_dir))
    for file in os.listdir(my_dir):
        if file.endswith("pkl"):
            pkl_file = os.path.join(my_dir, file)
    if debug:
        print("Reading the file...")
        print(pkl_file)
    # All the info at each step for the cluster, size composition atomic ids etc
    all_info = []
    with open(pkl_file, "rb") as f:
        # The list with all the info
        all_info = pickle.load(f)

    # Get the number of configurations
    Nc = len(all_info)
    
    for ic in range(Nc):
        # Each snapshots contains more than one clusters in principle
        clusters = []
        for id_clust in range(all_info[ic]['num_clust']):
            clusters.append({'ids'  : all_info[ic]["id_{}{}".format(types[0], types[1])][id_clust],
                             'composition' : all_info[ic]['composition'][id_clust]})
        all_clusters.append(clusters)

    return all_clusters




# def NEWNEWNEWanalyze_clusters(all_clusters,  tol=0.5, zombie_steps=5, debug=False):
#     """
#     Track the birth, death, and lifetime of clusters across MD snapshots.

#     Parameters
#     ----------
#     all_clusters : list[list[dict]]
#         Each element is a list of cluster dicts:
#         [{'ids': np.array([...]), 'composition': 'Na3Cl2'}, ...]
#     jacardi_rule : callable
#         Function that compares two arrays of atom IDs and returns True if similar.
#     tol : float
#         Similarity tolerance for jacardi_rule.
#     zombie_steps : int
#         Max number of steps a dead cluster can "revive" before being finalized.
#     debug : bool
#         Print verbose debug info.
#     """

#     clusters_alive = []   # list of dicts: ids, composition, step0
#     clusters_death = []   # list of dicts: ids, composition, step0, step1
#     compositions_tau = [] # list of dicts: composition, all_tau (lifetimes)

#     n_steps = len(all_clusters)
#     if debug:
#         print(f"Total number of MD snapshots: {n_steps}\n")

#     for step, clusters in enumerate(all_clusters):
#         if debug:
#             print(f"\n=== STEP {step} ===")
#             print(f"Found {len(clusters)} clusters")

#         # --- 1. Check for new or existing clusters ---
#         new_alive = []
#         for cluster in clusters:
#             found = False
#             for alive in clusters_alive:
#                 if jacardi_rule(cluster["ids"], alive["ids"], tol=tol):
#                     # Same cluster, keep original birth step
#                     new_alive.append({
#                         "ids": cluster["ids"],
#                         "composition": cluster["composition"],
#                         "step0": alive["step0"]
#                     })
#                     found = True
#                     break
#             if not found:
#                 # New cluster born
#                 if debug:
#                     print(f"BIRTH | Cluster {cluster['ids']} ({cluster['composition']}) born at step {step}")
#                 new_alive.append({
#                     "ids": cluster["ids"],
#                     "composition": cluster["composition"],
#                     "step0": step
#                 })
#         clusters_alive = new_alive

#         # --- 2. Check for clusters that died ---
#         survivors = []
#         for alive in clusters_alive:
#             still_present = any(
#                 jacardi_rule(alive["ids"], c["ids"], tol=tol) for c in clusters
#             )
#             if still_present:
#                 survivors.append(alive)
#             else:
#                 if debug:
#                     print(f"DEATH | Cluster {alive['ids']} ({alive['composition']}) "
#                           f"born at {alive['step0']} died at {step}")
#                 clusters_death.append({
#                     "ids": alive["ids"],
#                     "composition": alive["composition"],
#                     "step0": alive["step0"],
#                     "step1": step
#                 })
#         clusters_alive = survivors

#         # --- 3. Handle zombie clusters (possible revival) ---
#         remaining_death = []
#         for cluster_dead in clusters_death:
#             # Check if revived
#             revived = any(
#                 jacardi_rule(cluster_dead["ids"], c["ids"], tol=tol) for c in clusters_alive
#             )
#             if revived:
#                 if debug:
#                     print(f"ZOMBIE | Cluster {cluster_dead['ids']} revived from step {cluster_dead['step0']}")
#                 # If revived, just keep its original birth time in alive list
#                 continue

#             # Otherwise, check if it's been dead too long
#             if (step - cluster_dead["step1"]) > zombie_steps:
#                 tau = cluster_dead["step1"] - cluster_dead["step0"]
#                 comp = cluster_dead["composition"]
#                 existing = next((ct for ct in compositions_tau if ct["composition"] == comp), None)
#                 if existing:
#                     existing["all_tau"].append(tau)
#                 else:
#                     compositions_tau.append({"composition": comp, "all_tau": [tau]})
#             else:
#                 remaining_death.append(cluster_dead)

#         clusters_death = remaining_death

#     # --- Handle clusters still alive at the end ---
#     for cluster_alive in clusters_alive:
#         tau = n_steps - cluster_alive["step0"]
#         comp = cluster_alive["composition"]
#         existing = next((ct for ct in compositions_tau if ct["composition"] == comp), None)
#         if existing:
#             existing["all_tau"].append(tau)
#         else:
#             compositions_tau.append({"composition": comp, "all_tau": [tau]})

#     # --- Print results ---
#     print("\n========================")
#     print("Final results")
#     for ct in compositions_tau:
#         avg_tau = np.mean(ct["all_tau"]) if ct["all_tau"] else 0
#         print(f"Composition {ct['composition']} | Average lifetime: {avg_tau:.2f} steps")
#     print("========================\n")

#     return compositions_tau





def NEW_run_analysis_lifetime(all_clusters, types = ["Na", "Cl"], debug = True, tol = 0.5, zombie_steps = 500):
    """
    RUN THE ANALYSIS ON THE AVERAGE LIFE-TIME
    =========================================

    The idea is the following. 

    1st check if there are new clusters

    2nd check if some clusters died
    

    Parameters:
    -----------
        -all_clusters: For each MD snapshot, we have a list of dictionaries. 
                       Each dictionary contains the informations about the cluster the keys are ids, compositions

        -types: list of two string (chemical types)

        -debug: bool, 

        -tol: float, the tolerance used in the Jacardi index for similarity check

        -zombie_steps: int, the number of steps after which we consider a cluster definitely death

    Returns:
    --------
        -cluster_lifetimes: list of cluster lifetimes in PICOSECONDS
    """
    if debug:
        print("Total number of MD snapshots ", len(all_clusters))
        print()

    # The cluster that are ALIVE. 
    # A list of dictionaries. Each dictionary contais the ids of the atoms in the cluster, the compostions and the first step they appeared at
    clusters_alive = []

    # The cluster that are DEATH.
    # A list of dictionaries. Each dictionary contais the ids of the atoms in the cluster, the compostions and the first and last step they appeared at
    clusters_death = []
    
    # The compositions of the clusters that are formed during the simulation
    # A list of dictionaries. Each dictionary contains the info about the chemical composition and a list of all lifetime steps
    compositions_tau = []

    # check all the configurations
    for step, clusters in enumerate(all_clusters):
        
        if debug:
            print("\n\n===========================")
            print("====> BEGIN STEP {}".format(step))
            print("The number of current clusters is ", len(clusters))
            print("Current clusters")
            for i, dummy in enumerate(clusters):
                print(i, dummy)
            
            print("Alive clusters")
            for i, dummy in enumerate(clusters_alive):
                print(i, dummy)
            print("The number of alive clusters is ", len(clusters_alive))
            print()

        if debug:
            print("\n\nLOOKING FOR NEW CLUSTERS")
        # 1st- CHECK IF THERE ARE NEW CLUSTERS FORMED
        for cluster in clusters:
            found_similar = False
            for id_clust_alive, cluster_alive in enumerate(clusters_alive):
                if jacardi_rule(cluster["ids"], cluster_alive["ids"], tol=tol):
                    correct_step = cluster_alive["step0"]
                    clusters_alive[id_clust_alive] = {
                        "ids": cluster["ids"],
                        "composition": cluster["composition"],
                        "step0": correct_step
                    }
                    found_similar = True
                    break
            if not found_similar:
                if debug:
                    print(f"BIRTH | Cluster {tuple(cluster['ids'].tolist())} "
                          f"{cluster['composition']} born on MD step {step}")
                clusters_alive.append({
                    'ids': cluster["ids"],
                    'composition': cluster["composition"],
                    'step0': step
                })
                
        if debug:
            print("\nDONE LOOKING FOR NEW CLUSTERS")
            print("Current clusters")
            for dummy in clusters:
                print(dummy)
            print("Alive clusters")
            for dummy in clusters_alive:
                print(dummy)
            print()

        # 2nd- Check if some of the clusters alive have died and eventually remove them 
        if debug:
            print("\n\nLOOKING FOR DEATH CLUSTERS")

        new_alive = []
        for i, cluster_alive in enumerate(clusters_alive):
            is_alive = any(jacardi_rule(cluster_alive["ids"], c["ids"]) for c in clusters)
            if is_alive:
                new_alive.append(cluster_alive)
            else:
                clusters_death.append({
                    "ids": cluster_alive["ids"],
                    "composition": cluster_alive["composition"],
                    "step0": cluster_alive["step0"],
                    "step1": step
                })
                if debug:
                    print("DEATH |")
                    print(clusters_death[-1])
                    print("DEATH | Cluster was born on step {} has died at {} steps".format(cluster_alive["step0"], step))
        clusters_alive = new_alive
         
        if debug:
            print("\nEND LOOKING FOR DEATH CLUSTERS")
            print("Current clusters")
            for dummy in clusters:
                print(dummy)
            print("Alive clusters")
            for dummy in clusters_alive:
                print(dummy)
            print("Death clusters")
            for dummy in clusters_death:
                print(dummy)
            print()

        
        
        if debug:
            print("\n\nLOOKING FOR CLUSTERS THAT ARE ALIVE AGAIN")
            print("Death clusters")
            for dummy in clusters_death:
                print(dummy)
            print()
            
        # 3th step Check if the death clusters are alive again after max zombie_steps
        new_death = []
        for id_clust_death, cluster_death in enumerate(clusters_death):
            revived = False
            for id_clust_alive, cluster_alive in enumerate(clusters_alive):
                is_alive_again = jacardi_rule(cluster_death["ids"], cluster_alive["ids"])
                if is_alive_again:
                    print('\nRELIVE |')
                    print(cluster_death)
                    print('reborn  at step {} after {} steps after death'.format(step, step - cluster_death['step1']))
                    # Update 
                    clusters_alive[id_clust_alive]['composition'] = cluster_alive["composition"]
                    clusters_alive[id_clust_alive]['step0']       = cluster_death['step0']
                    revived = True
                    break
            if not revived:
                # If the cluster is not alive
                # Check if it should be removed FOREVER !!!
                if (step - cluster_death['step1']) > zombie_steps:
                    print('\nKILLING FOREVER |')
                    print(cluster_death)
                    print('at step {} after {} zombie steps'.format(step, zombie_steps))
                    found = False
                    # Get the time of life
                    tau = cluster_death["step1"] - cluster_death["step0"]
                    for ict, ct in enumerate(compositions_tau):
                        if cluster_death["composition"] == ct["composition"]:
                            ct["all_tau"].append(tau)
                            found = True
                            break
                    if not found:
                        compositions_tau.append({"composition" : cluster_death["composition"], "all_tau" : [tau]})
                else:
                    new_death.append(cluster_death)
                        
        clusters_death = new_death
                            
        if debug:
            print("\nEND LOOKING FOR CLUSTERS THAT ARE ALIVE AGAIN")
            print("Death clusters")
            for dummy in clusters_death:
                print(dummy)
            print()
                    #
            
        if debug:
            print()

    if debug:
        for dummy in clusters_alive:
                print(dummy)

    # Check for the remaining clusters
    for cluster_alive in clusters_alive:
        tau = step - cluster_alive["step0"]
        comp = cluster_alive["composition"]
    
        # Check if this composition already exists
        found = False
        for ct in compositions_tau:
            if ct["composition"] == comp:
                ct["all_tau"].append(tau)
                found = True
                # important: stop searching once found
                break  
    
        # If not found, create a new entry
        if not found:
            compositions_tau.append({
                "composition": comp,
                "all_tau": [tau]
            })


    print("========================")
    print("Final results")
    for ict, ct in enumerate(compositions_tau):
        print("Composition {} steps lifetime average {}".format(ct["composition"], np.average(ct["all_tau"])))
    print("END Final results")
    print("========================")
    print()
    
    return compositions_tau

    

# def run_analysis_lifetime(ids_per_conf, info_per_conf, types = ["Na", "Cl"], debug = True):
#     """
#     RUN THE ANALYSIS ON THE AVERAGE LIFE-TIME
#     =========================================

#     The idea is the following. 

#     1st check if there are new clusters

#     2nd check if some clusters died
    

#     Parameters:
#     -----------
#         -ids_per_conf: a list of list with numpy array. For each MD snapshot, we have the atomic indices of the clusters found

#     Returns:
#     --------
#         -cluster_lifetimes: list of cluster lifetimes in PICOSECONDS
#     """
#     if debug:
#         print("Total number of MD snapshots ", len(ids_per_conf))
#         print()

#     # The cluster that are alive. The key is a tuple like (id1,id2,id3) and the corresponding value is when it appeared
#     cluster_last_step = {}
#     # The cluster life times. Each time a cluster dies we store its lifetime
#     cluster_lifetimes = []
#     # The compositions of the clusters that are formed during the simulation
#     chemical_composition = {}

#     for step, clusters in enumerate(ids_per_conf):
        
#         if debug:
#             print("\n\n===========================")
#             print("STEP {}".format(step))
#             print("The number of clusters is ", len(clusters))
#             print("The current clusters")
#             print(clusters)
#             print("The clusters of the last step")
#             print(cluster_last_step)
            
#         # 1st- Check if there are new clusters
#         for id_clust, cluster in enumerate(clusters):
#             is_there = any(np.array_equal(np.asarray(prev_cluster), cluster) for prev_cluster in cluster_last_step)
            
#             if not is_there:
#                 if debug:
#                     print("BIRTH | Cluster {} was born on step {}".format(cluster, step))
#                 cluster_last_step.update({tuple(cluster.tolist()) : step})

#                 # Get the chemical composition of the new cluster found
#                 chem_comp_cluster = info_per_conf[step]["composition"][id_clust]
#                 if chem_comp_cluster in chemical_composition.keys():
#                     chemical_composition[chem_comp_cluster] += 1
#                 else:
#                     chemical_composition.update({chem_comp_cluster : 1})

#         if debug and step == 0:
#             print("INITIALIZE...")
#             print(cluster_last_step)
#             print()

#         if debug:
#             print("Current clusters")
#             for dummy in clusters:
#                 print(dummy)
#             print("Last step clusters")
#             for dummy in cluster_last_step:
#                 print(dummy)
            

#         # 2nd- Check if some of the clusters has died and eventually remove them 
#         items_to_kill = []
#         for item in cluster_last_step.keys():
#             # Get ther composition
#             old_cluster = np.asarray(item)
#             # Check if the cluster configuration is still alive
#             is_alive = any(np.array_equal(cluster, old_cluster) for cluster in clusters)

#             if not is_alive:
#                 tau = step - cluster_last_step[item]
#                 # print(cluster_last_step[item])
#                 cluster_lifetimes.append(tau)
#                 items_to_kill.append(item)
#                 if debug:
#                     print("DEATH | Cluster {} was born on step {} has died after {} steps".format(item, cluster_last_step[item], tau))

#         # Remove the death clusters then go on
#         for item in items_to_kill:
#             cluster_last_step.pop(item)
            

#         if debug:
#             print()


#     # Get the last lifetimes
#     for item in cluster_last_step.keys():
#         tau = len(ids_per_conf) - cluster_last_step[item]
#         # print(cluster_last_step[item])
#         cluster_lifetimes.append(tau)

#     if debug:
#         plt.hist(np.asarray(cluster_lifetimes) * time_step)
#         plt.xlabel('$\\tau$ [fs]', fontsize = 15)
#         plt.tight_layout()
#         plt.tick_params(axis = 'both', labelsize = 15)
#         plt.show()

#     # Result in fs
#     return np.asarray(cluster_lifetimes) * time_step, chemical_composition 



if __name__ == '__main__':
    """
    POST PROCESSING FOR PAIRING
    ===========================


    Here we try to identify the life times of clusters.

    Units are FEMPTOSECOND

    Usage is python3 path/analyze_clusters.py dir_with_results_pkl type1 type2 dt dbg_int
    """
   
    if len(sys.argv[1:]) != 5:
        raise ValueError("python3 path/analyze_clusters.py dir_with_results_pkl type1 type2 dt dbg_int")

    #### INPUT PARAMETERS ####
    # the directory with the pkl file
    my_dir = sys.argv[1]

    # The types
    types = [sys.argv[2], sys.argv[3]]

    # Time step in femtoseconds
    time_step = float(sys.argv[4])

    # debug
    debug = bool(int(sys.argv[5]))
    #### END INPUT PARAMETERS ####


    

    print("\n==================================")
    print("THIS IS THE ANALYZE CLUSTER SCRIPT")
    print("===================================\n\n")

    print("The directory is {}".format(my_dir))
    print("Types {}".format(types))
    print("Time step {:.2f} fs".format(time_step))
    print("Debug {}".format(debug))
    print()


    # # Get the number of configurations Nc 
    # # info, a list of dictionaries with several information about the clustering 'id_A' 'id_B' 'conf' 'sizes' 'composition'
    # # ids_per_conf is  a list with all the ids forming the cluster for each snapshocts
    # Nc, info, ids_per_conf = get_results_from_dir(my_dir)
    # # filter_ids_per_conf = filter_atoms(ids_per_conf, info, target_comp = "Na1Cl1")
    # # print(ids_per_conf)
    # # print(filter_ids_per_conf)

    # # Check ho
    # # time in picoseconds
    # tau, compositions = run_analysis_lifetime(ids_per_conf, info, types = types, debug = debug)

    # print(compositions)
    # plt.plot(list(compositions.keys()), compositions.values())
    # plt.show()


    all_clusters = NEW_get_results_from_dir(my_dir)
    
    if debug:
        for item in all_clusters:
            print(item)
    compositions_tau = NEW_run_analysis_lifetime(all_clusters, types = types, debug = debug, tol = 0.5, zombie_steps = 500)
    #compositions_tau = NEWNEWNEWanalyze_clusters(all_clusters,  debug = debug)
    if debug:
        print(compositions_tau)
    # Save the compositions as well
    with open(os.path.join(my_dir, "composition.json"), "w") as json_file:
        json.dump(compositions_tau, json_file)    


        
            


