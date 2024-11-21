#!/usr/bin/python
from average_results import state_averaging, count_adapters
import networkx as nx
import LPA
import numpy as np
from pandas import read_csv

from conf import ITERATION_NUM, INITIAL_VECTOR_LABELS_FILE, EDGES_FILE, GRAPH_TYPE, LABELS, ATTRIBUTES_FILE, \
    RESULTS_DIR, NON_CONSECUTIVE_NODE_INDEXES
from after_simluation_plots import draw_adapter_by_time_plot, description_text_for_plots
from network_simulation import NetworkSimulation
from create_input import create_input_files
from preprocessing import load_dataset_csv

from scipy.stats import beta as beta_function
import utils

def run_simulations(run_index, results_dir):
    # Create the network from edges defined in EDGES_FILE file
    if GRAPH_TYPE == "D":
        LPNet = nx.read_edgelist(EDGES_FILE, nodetype=int, create_using=nx.DiGraph, data=[('weight', float)])
    elif GRAPH_TYPE == "U":
        LPNet = nx.read_edgelist(EDGES_FILE, nodetype=int, create_using=nx.Graph, data=[('weight', float)])
    else:
        print("The type of the graph must be U(undirected) or D(directed)")
        return
    LPNet.name = "LPA Network"

    # ATTRIBUTE_FILE
    # add the attributes from the file to the nodes
    attributes = read_csv(ATTRIBUTES_FILE)
    for _, name in enumerate(attributes.columns):
        nx.set_node_attributes(LPNet, attributes[name], name)

    # INITIAL_VLS
    # get VLs' values from file and assigns initial state
    initial_VLs = []
    with open(INITIAL_VECTOR_LABELS_FILE, 'r') as read_obj:
        for line in read_obj:
            array = [float(x) for x in line.strip().split(sep=';')]
            initial_VLs.append(array)
    initial_VLs = np.array(initial_VLs)
    initial_states = [1 if vls[0] == 0. and vls[1] == 1 else -1 for vls in initial_VLs]

    network_nodes = sorted(LPNet.nodes())
    for i, node in enumerate(network_nodes):
        # computes the perseverance of each node, since it is an attribute of the agent,
        # and it cannot change during the simulation
        LPNet.nodes[node]["perseverance"] = beta_distribution(ALPHA, BETA)
        # assign to the nodes the vector labels
        for j, label in enumerate(LABELS):
            # If the network created from the edge list file does not contain
            # all consecutive nodes and the INITIAL_VLS_FILE contains also the VLs of the nodes missing in the range:
            # execute the script providing an additional command line argument 'X':
            if NON_CONSECUTIVE_NODE_INDEXES:
                LPNet.nodes[node][label] = initial_VLs[i][j]
            else:
                LPNet.nodes[node][label] = initial_VLs[node - network_nodes[0]][j]
        # assigns the initial state
        LPNet.nodes[node]["state"] = initial_states[i]
    adapters = [node for node in LPNet.nodes() if LPNet.nodes[node]["state"] == 1]
    print(f"Initial adapters/non adapters ratio: {len(adapters)}/{len(LPNet.nodes)}")
    print(sorted(adapters))
    # na.NetworkAnalysis(LPNet).analyse()
    # exit(1)

    # run simulation
    simulation = NetworkSimulation(LPNet, LPA, ITERATION_NUM, results_dir)
    simulation.run_simulation(run_index)


def beta_distribution(alpha, beta):
    return beta_function.rvs(alpha, beta)


vector_labels_update_choices = {
    0: "same-weights",  # trivial cases (SIM-*0)
    1: "beta-dist",  # baseline
    2: "over-confidence",  # OL and OP are fixed: OP = 0.8 (confidence in my opinion) (EX0-1)
    3: "over-influenced",  # OL and OP are fixed: OL = 0.8 (too easily influenced by others) (EX0-2)
    4: "extreme-influenced",  # OL and OP are fixed: OP=0.02 and OL=0.98 (EX0-3),
    5: "simple-contagion",  # becomes adopter if at least one neighbours is
    6: "majority",  # becomes adopters if the majority of the neighbours is
    7: "extreme-confidence"
}

initialisation_choises = {
    0: "random-adapters",  # completely random percentage of adapters
    1: "adapters-with-SI",  # percentage of adapters with sharing index bigger than average
    2: "would-subscribe-attribute"  # who has responded Yes to Would_subscribe_car_sharing_if_available
}

# biases is introduced by using SI as perc for becoming adopter
# the different type of biases depends on the moment of application
all_cognitive_biases = {
    0: "no-bias",
    1: "confirmation-bias",  # if the majority of neighbours is non adopters
    2: "availability-bias",  # if the majority of neighbours is adopter
    3: "confirmation-availability-bias"  # in any case
}

percentages = [5, 20, 40]

RUNS = 5  # 30
SIMILARITY_THRESHOLD = 0.60
ALPHA = 2
BETA = 2
VL_UPDATE_METHOD = vector_labels_update_choices[4]
INITIALISATION = initialisation_choises[2]
INITIAL_ADAPTERS_PERC = -2
APPLY_COGNITIVE_BIAS = all_cognitive_biases[0]
counter = 1
if __name__ == '__main__':
    for init_type in initialisation_choises.values():
        INITIALISATION = init_type
        for perc in percentages:
            INITIAL_ADAPTERS_PERC = perc
            path = RESULTS_DIR.split("/")
            sim_id = path[-1]
            new_sim_id = sim_id + f"-{counter}"
            dir = RESULTS_DIR.replace(sim_id, new_sim_id)
            utils.create_if_not_exist(dir)
            print(RESULTS_DIR)
            for run in range(0, RUNS):
                print(f"---- Run {run} ----")
                data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
                create_input_files(data, [
                    "sha_ind_norm",
                    "Gender",
                    "Education",
                    "Income_level",
                    "Age",
                    "Would_subscribe_car_sharing_if_available"],
                                   similarity_threshold=SIMILARITY_THRESHOLD,
                                   initialisation=INITIALISATION,
                                   perc_of_adapters=INITIAL_ADAPTERS_PERC
                                   )
                run_simulations(run, dir)

            states = state_averaging(dir)
            title, additional_text = description_text_for_plots(VL_UPDATE_METHOD, sim_id,
                                                                sim_threshold=SIMILARITY_THRESHOLD,
                                                                vl_update=VL_UPDATE_METHOD,
                                                                initialisation=INITIALISATION,
                                                                adapters_perc=INITIAL_ADAPTERS_PERC,
                                                                cognitive_bias= APPLY_COGNITIVE_BIAS,
                                                                alpha=ALPHA,
                                                                beta=BETA,)
            draw_adapter_by_time_plot(states, dir, title=title, additional_text=additional_text)
            draw_adapter_by_time_plot(states, dir, title=title, additional_text='')
            utils.write_simulation_readme_file(dir,
                                               vl_update=VL_UPDATE_METHOD,
                                               initialisation=init_type,
                                               adapters_perc=perc,
                                               similarity_threshold=SIMILARITY_THRESHOLD,
                                               cognitive_bias=APPLY_COGNITIVE_BIAS,
                                               states=states)
            counter += 1
