#!/usr/bin/python

import networkx as nx, csv, sys, LPA, NETWORKSIMULATION
import numpy as np
from pandas import read_csv

from conf import ITERATION_NUM, LABELS_INIT_VALUES_FILE, EDGES_FILE, GRAPH_TYPE, LABELS, ATTRIBUTES_FILE
from create_input import create_input
from preprocessing import load_dataset_csv
import RESULTPLOTTER


def main():
    # Create the network from edges defined in EDGES_FILE file
    if GRAPH_TYPE == "D":
        LPNet = nx.read_edgelist(EDGES_FILE, nodetype=int, create_using=nx.DiGraph, data=[('weight', float)])
    elif GRAPH_TYPE == "U":
        LPNet = nx.read_edgelist(EDGES_FILE, nodetype=int, create_using=nx.Graph, data=[('weight', float)])
    else:
        print("The type of the graph must be U(undirected) or D(directed)")
        return

    # ATTRIBUTE_FILE
    # add the attributes from the file to the nodes
    attributes = read_csv(ATTRIBUTES_FILE, header=None)
    for i, _ in enumerate(LABELS):
        nx.set_node_attributes(LPNet, attributes[i], f"attribute-{i}")

    # Get VLs' values from file
    with open(LABELS_INIT_VALUES_FILE, 'r') as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        initial_VLs = list(csv_reader)

    network_nodes = sorted(LPNet.nodes())
    print(LPNet.nodes())
    # Initialize nodes' VLs
    for i, node in enumerate(network_nodes):
        for j, label in enumerate(LABELS):
            if len(sys.argv) == 6:
                LPNet.nodes[node][label] = initial_VLs[i][j]
            else:
                LPNet.nodes[node][label] = initial_VLs[node - network_nodes[0]][j]

    # Run simulation
    simulation = NETWORKSIMULATION.NetworkSimulation(LPNet, LPA, ITERATION_NUM)
    simulation.run_simulation()


if __name__ == '__main__':
    data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
    create_input(data, ["sha_ind_norm", "Gender", "Education", "Income_level"])
    main()
    # files = []
    # thresholds = np.arange(0, 1.1, 0.1)
    # for threshold in thresholds:
    #     print(f"THRESHOLD: {threshold}")
    #     DNA_THRESHOLD = threshold
    #     main()
    #     files.append(f"./work/results/trial_0_LPStates_L0_{threshold}.pickled")
    # plotter = RESULTPLOTTER.ResultPlotter(files)
    # plotter.draw_adapter_by_time_different_thresholds_plot(thresholds)
    plotter = RESULTPLOTTER.ResultPlotter(["./work/results/trial_0_LPStates_L0.pickled"])
    plotter.draw_adapter_by_time_plot()
