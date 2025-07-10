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

    # For each configuration we have for each cluster the ids of the atoms forming it
    all_ids_per_confs = [all_info[ic]["id_{}{}".format(types[0], types[1])] for ic in range(len(all_info))]

    return Nc, all_info, all_ids_per_confs

    

def run_analysis_lifetime(ids_per_conf, info_per_conf, types = ["Na", "Cl"], debug = True):
    """
    RUN THE ANALYSIS ON THE AVERAGE LIFE-TIME
    =========================================

    The idea is the following. 

    1st check if there are new clusters

    2nd check if some clusters died
    

    Parameters:
    -----------
        -ids_per_conf: a list of list with numpy array. For each MD snapshot, we have the atomic indices of the clusters found

    Returns:
    --------
        -cluster_lifetimes: list of cluster lifetimes in PICOSECONDS
    """
    if debug:
        print("Total number of MD snapshots ", len(ids_per_conf))
        print()

    # The cluster that are alive. The key (id1,id2,id3) and the corresponding value is when it appeared
    cluster_last_step = {}
    # The cluster life times. Each time a cluster dies we store its lifetime
    cluster_lifetimes = []
    # The compositions of the clusters that form
    chemical_composition = {}

    for step, clusters in enumerate(ids_per_conf):

        if debug:
            print("STEP {}".format(step))
            
        # 1st- Check if there are new clusters
        for id_clust, cluster in enumerate(clusters):
            is_there = any(np.array_equal(np.asarray(prev_cluster), cluster) for prev_cluster in cluster_last_step)
            
            if not is_there:
                if debug:
                    print("BIRTH | Cluster {} was born on step {}".format(cluster, step))
                cluster_last_step.update({tuple(cluster.tolist()) : step})

                # Get the chemical composition of the new cluster found
                chem_comp_cluster = info_per_conf[step]["composition"][id_clust]
                if chem_comp_cluster in chemical_composition.keys():
                    chemical_composition[chem_comp_cluster] += 1
                else:
                    chemical_composition.update({chem_comp_cluster : 1})

        if debug and step == 0:
            print("INITIALIZE...")
            print(cluster_last_step)
            print()

        if debug:
            print("Current clusters")
            for dummy in clusters:
                print(dummy)
            print("Last step clusters")
            for dummy in cluster_last_step:
                print(dummy)
            

        # 2nd- Check if some of the clusters has died and eventually remove them 
        items_to_kill = []
        for item in cluster_last_step.keys():
            # Get ther composition
            old_cluster = np.asarray(item)
            # Check if the cluster configuration is still alive
            is_alive = any(np.array_equal(cluster, old_cluster) for cluster in clusters)

            if not is_alive:
                tau = step - cluster_last_step[item]
                # print(cluster_last_step[item])
                cluster_lifetimes.append(tau)
                items_to_kill.append(item)
                if debug:
                    print("DEATH | Cluster {} was born on step {} has died after {} steps".format(item, cluster_last_step[item], tau))

        # Remove the death clusters then go on
        for item in items_to_kill:
            cluster_last_step.pop(item)
            

        if debug:
            print()


    # Get the last lifetimes
    for item in cluster_last_step.keys():
        tau = len(ids_per_conf) - cluster_last_step[item]
        # print(cluster_last_step[item])
        cluster_lifetimes.append(tau)

    if debug:
        plt.hist(np.asarray(cluster_lifetimes) * time_step)
        plt.xlabel('$\\tau$ [fs]', fontsize = 15)
        plt.tight_layout()
        plt.tick_params(axis = 'both', labelsize = 15)
        plt.show()

    # Result in fs
    return np.asarray(cluster_lifetimes) * time_step, chemical_composition 



if __name__ == '__main__':
    """
    POST PROCESSING FOR PAIRING
    ===========================

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

    # Time step in femptosecond
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


    # Get the number of configurations Nc and info, a list of dictionaries with several information about the clustering
    Nc, info, ids_per_conf = get_results_from_dir(my_dir)
    # filter_ids_per_conf = filter_atoms(ids_per_conf, info, target_comp = "Na1Cl1")
    # print(ids_per_conf)
    # print(filter_ids_per_conf)

    # Check ho
    # time in picoseconds
    tau, compositions = run_analysis_lifetime(ids_per_conf, info, types = types, debug = debug)

    print(compositions)
    # plt.plot(list(compositions.keys()), compositions.values())
    # plt.show()
    
    # Save the life times
    np.save(os.path.join(my_dir, "life_times"), tau)
    # Save the compositions as well
    with open(os.path.join(my_dir, "composition.json"), "w") as json_file:
        json.dump(compositions, json_file)    


        
            


