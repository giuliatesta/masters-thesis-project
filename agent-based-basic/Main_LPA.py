#!/usr/bin/python

import networkx as nx, sys, LPA, NETWORKSIMULATION
import numpy as np
from pandas import read_csv

from conf import ITERATION_NUM, INITIAL_VECTOR_LABELS_FILE, EDGES_FILE, GRAPH_TYPE, LABELS, ATTRIBUTES_FILE
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
    initial_VLs = []
    with open(INITIAL_VECTOR_LABELS_FILE, 'r') as read_obj:
        for line in read_obj:
            array = [float(x) for x in line.strip().split(sep=';')]
            initial_VLs.append(array)
    initial_VLs = np.array(initial_VLs)

    network_nodes = sorted(LPNet.nodes())
    # Initialize nodes' VLs
    for i, node in enumerate(network_nodes):
        for j, label in enumerate(LABELS):
            if len(sys.argv) == 7:
                LPNet.nodes[node][label] = initial_VLs[i][j]
            else:
                LPNet.nodes[node][label] = initial_VLs[node - network_nodes[0]][j]

        LPNet.nodes[node]["state"] = 1 if initial_VLs[i][0] < initial_VLs[i][1] else -1
    # Run simulation
    simulation = NETWORKSIMULATION.NetworkSimulation(LPNet, LPA, ITERATION_NUM)
    simulation.run_simulation()


if __name__ == '__main__':
    data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
    create_input(data, ["sha_ind_norm", "Gender", "Education", "Income_level"])
    main()
    # plotter = RESULTPLOTTER.ResultPlotter(["./work/results/trial_0_LPStates_L0_L1_3.pickled"])
    # plotter.heatmap()
