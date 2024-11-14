#!/usr/bin/python
from average_results import state_averaging
import networkx as nx
import sys
import LPA
import numpy as np
from pandas import read_csv

from conf import ITERATION_NUM, INITIAL_VECTOR_LABELS_FILE, EDGES_FILE, GRAPH_TYPE, LABELS, ATTRIBUTES_FILE, RESULTS_DIR
from after_simluation_plots import draw_adapter_by_time_plot
from network_simulation import NetworkSimulation
from create_input import create_input_files
from preprocessing import load_dataset_csv

from scipy.stats import beta as beta_function


def run_simulations(run_index):
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
            if len(sys.argv) == 7:
                LPNet.nodes[node][label] = initial_VLs[i][j]
            else:
                LPNet.nodes[node][label] = initial_VLs[node - network_nodes[0]][j]
        # assigns the initial state
        LPNet.nodes[node]["state"] = initial_states[i]
    adapters = [node for node in LPNet.nodes if LPNet.nodes[node]["state"] == 1]
    print(f"Initial adapters/non adapters ratio: {len(adapters)}/{len(LPNet.nodes)}")

    # na.NetworkAnalysis(LPNet).analyse()
    # exit(1)

    # run simulation
    simulation = NetworkSimulation(LPNet, LPA, ITERATION_NUM)
    simulation.run_simulation(run_index)


def beta_distribution(alpha, beta):
    return beta_function.rvs(alpha, beta)


RUNS = 10  # 30
SIMILARITY_THRESHOLD = 0.60
ALPHA = 2
BETA = 2
USE_SHARING_INDEX = True
all_choices = {
    0: "same-weights",  # trivial cases (SIM-*0)
    1: "beta-dist",  # baseline
    2: "over-confidence",  # OL and OP are fixed: OP = 0.8 (confidence in my opinion) (EX0-1)
    3: "over-influenced"  # OL and OP are fixed: OL = 0.8 (too easily influenced by others) (EX0-2)
}
STATE_CHANGING_METHOD = all_choices[2]
if __name__ == '__main__':
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
                           )
        run_simulations(run)

    states = state_averaging(RESULTS_DIR)
    additional_text = ("Adapters: WOULD_SUBSCRIBE_CAR_SHARING (133)\n"
                       #  + f"Perseverance: 0.8, Plasticity: 0.2\n"
                       + f"Perseverance: 1 / (k+1), "
                       + f"Plasticity: 1 / (k+1)\n"
                       # + f"Plasticity: scaled similarity weights\n"
                       # + f"Perseverance: beta distribution (alpha = {ALPHA}, beta = {BETA})\n"
                       + f"Similarity threshold: {SIMILARITY_THRESHOLD}\n"
                       + f"Vector label changing: NO BIAS\nState changing:WITH"
                         f"{'' if USE_SHARING_INDEX else 'OUT'} INDEX")
    # additional_text = ("Adapters: WOULD_SUBSCRIBE_CAR_SHARING (133)\n"
    #                    + f"Perseverance: 1 / (k+1), "
    #                      + f"Plasticity: 1 / (k+1)\n"
    #                    + f"Similarity threshold: {SIMILARITY_THRESHOLD}\n"
    #                    + f"Vector label changing: NO BIAS\nState changing:
    #                    WITH{'' if USE_SHARING_INDEX else 'OUT'} INDEX")
    draw_adapter_by_time_plot(states, RESULTS_DIR, title="Number of adapters by time\n"
                                                         "(BASELINE with same weights - SIM B0-A0)",
                              # "(EXTRAS - OL 0.8, OP 0.2)",
                              additional_text=additional_text)
