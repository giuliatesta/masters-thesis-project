#!/usr/bin/python
import os

import utils

from average_results import state_averaging
import networkx as nx, sys, LPA, network_simulation
import numpy as np
from pandas import read_csv

from conf import ITERATION_NUM, INITIAL_VECTOR_LABELS_FILE, EDGES_FILE, GRAPH_TYPE, LABELS, ATTRIBUTES_FILE, RESULTS_DIR
from create_input import create_input
from preprocessing import load_dataset_csv
from simulation_result_plotter import draw_adapter_by_time_plot, plot_multiple_adapters_by_time
from network_simulation import NetworkSimulation
from network_analysis import NetworkAnalysis
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
    LPNet.name = "LPA Network"

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

    network_nodes = sorted(LPNet.nodes())
    # Initialize nodes' VLs
    for i, node in enumerate(network_nodes):
        for j, label in enumerate(LABELS):
            if len(sys.argv) == 7:
                LPNet.nodes[node][label] = initial_VLs[i][j]
            else:
                LPNet.nodes[node][label] = initial_VLs[node - network_nodes[0]][j]

        LPNet.nodes[node]["state"] = initial_states[i]
    adapters = [node for node in LPNet.nodes if LPNet.nodes[node]["state"] == 1]

    # Run simulation
    simulation = NetworkSimulation(LPNet, LPA, ITERATION_NUM)
    simulation.run_simulation(run_index)


RUNS = 30
if __name__ == '__main__':
    for run in range(0, RUNS):
        print(f"---- Run {run} ----")
        data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
        create_input(data, [INDEX_DNA_COLUMN_NAME, "Gender", "Education", "Income_level", "Age"])
        main(run)

    states = state_averaging(RESULTS_DIR)
    draw_adapter_by_time_plot(states, RESULTS_DIR,
                    title="Number of adapters by time (only trust OLD - SIM 23)")
    # plot_multiple_adapters_by_time()
