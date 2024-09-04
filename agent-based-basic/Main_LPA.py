#!/usr/bin/python
import UTILS

from average_results import average_vls_results, average_state_results
import networkx as nx, sys, LPA, NETWORKSIMULATION
import numpy as np
from pandas import read_csv

from conf import ITERATION_NUM, INITIAL_VECTOR_LABELS_FILE, EDGES_FILE, GRAPH_TYPE, LABELS, ATTRIBUTES_FILE, RESULTS_DIR
from create_input import create_input
from preprocessing import load_dataset_csv
import RESULTPLOTTER

RUNS = 10

INDEX_DNA_COLUMN_NAME = "sha_ind_norm"


def main(run_index):
    # Create the network from edges defined in EDGES_FILE file
    if GRAPH_TYPE == "D":
        LPNet = nx.read_edgelist(EDGES_FILE, nodetype=int, create_using=nx.DiGraph, data=[('weight', float)])
    elif GRAPH_TYPE == "U":
        LPNet = nx.read_edgelist(EDGES_FILE, nodetype=int, create_using=nx.Graph, data=[('weight', float)])
    else:
        print("The type of the graph must be U(undirected) or D(directed)")
        return

    print(f"{len(LPNet.nodes)} nodes: {[node for node in LPNet.nodes]}")

    # ATTRIBUTE_FILE
    # add the attributes from the file to the nodes
    attributes = read_csv(ATTRIBUTES_FILE)
    for _, name in enumerate(attributes.columns):
        nx.set_node_attributes(LPNet, attributes[name], name)

    # Get VLs' values from file
    initial_VLs = []
    with open(INITIAL_VECTOR_LABELS_FILE, 'r') as read_obj:
        for line in read_obj:
            array = [float(x) for x in line.strip().split(sep=';')]
            initial_VLs.append(array)
    initial_VLs = np.array(initial_VLs)
    initial_states = [1 if vls[0] == 0. and vls[1] == 1 else -1 for vls in initial_VLs]
    print(initial_states)

    network_nodes = sorted(LPNet.nodes())
    # Initialize nodes' VLs
    for i, node in enumerate(network_nodes):
        for j, label in enumerate(LABELS):
            if len(sys.argv) == 7:
                LPNet.nodes[node][label] = initial_VLs[i][j]
            else:
                LPNet.nodes[node][label] = initial_VLs[node - network_nodes[0]][j]

        LPNet.nodes[node]["state"] = initial_states[i]
    # adapters = [node for node in LPNet.nodes if LPNet.nodes[node]["state"] == 1]
    # print(f"Initial adapters: {len(adapters)}")
    # Run simulation
    simulation = NETWORKSIMULATION.NetworkSimulation(LPNet, LPA, ITERATION_NUM)
    simulation.run_simulation(run_index)


if __name__ == '__main__':
    # UTILS.print_pickled_file("./work/results/sim_01/avg_results_states.pickled")
    # exit(1)
    for run in range(1, RUNS):
        print(f"---- Run {run} ----")
        data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
        create_input(data, [INDEX_DNA_COLUMN_NAME, "Gender", "Education", "Income_level"])
        main(run)
    average_vls_results(RESULTS_DIR, "avg_results_vls.pickled")
    average_state_results(RESULTS_DIR, "avg_results_states.pickled")

    plotter = RESULTPLOTTER.ResultPlotter([f"{RESULTS_DIR}/avg_results_states.pickled"])
    plotter.draw_adapter_by_time_plot()
