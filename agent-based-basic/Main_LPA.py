#!/usr/bin/python

import networkx as nx, csv, sys, LPA, NETWORKSIMULATION
from conf import ITERATION_NUM, LABELS_INIT_VALUES_FILE, EDGES_FILE, GRAPH_TYPE, LABELS
from create_input import create_input
from preprocessing import load_dataset_csv


def main():
    # Create the network from edges defined in EDGES_FILE file
    if GRAPH_TYPE == "D":
        LPNet = nx.read_edgelist(EDGES_FILE, nodetype=int, create_using=nx.DiGraph)
    elif GRAPH_TYPE == "U":
        LPNet = nx.read_edgelist(EDGES_FILE, nodetype=int, create_using=nx.Graph)
    else:
        print("The type of the graph must be U(undirected) or D(directed)")
        return

    # Get VLs' values from file
    with open(LABELS_INIT_VALUES_FILE, 'r') as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        initial_VLs = list(csv_reader)

    network_nodes = sorted(LPNet.nodes())
    # Initialize nodes' VLs
    for i, node in enumerate(network_nodes):
        for j, label in enumerate(LABELS):
            if (len(sys.argv) == 6):
                LPNet.nodes[node][label] = initial_VLs[i][j]
            else:
                LPNet.nodes[node][label] = initial_VLs[node - network_nodes[0]][j]

    # Run simulation
    simulation = NETWORKSIMULATION.NetworkSimulation(LPNet, LPA, ITERATION_NUM)
    simulation.runSimulation()


if __name__ == '__main__':
    data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
    for column in data.columns:
        if "_DNA" in column:
            create_input(data, [column])
            main()
