import numpy as np
import pandas as pd

from network_building import create_network

NUMBER_OF_RECORDS = 1000
INITIAL_ADAPTERS_PERC = 50


def create_input(data, LABELS):
    data.reset_index(drop=True, inplace=True)
    data = data.head(NUMBER_OF_RECORDS)

    # labels file only has L0, L1, L2
    labels = pd.DataFrame(generate_labels(len(LABELS)))
    labels.to_csv(path_or_buf="work/LABELS", index=False, header=False, sep=";")

    # initial vector labels
    initial_vls = pd.DataFrame(generate_binary_pairs(NUMBER_OF_RECORDS, INITIAL_ADAPTERS_PERC))
    initial_vls.to_csv(path_or_buf="work/INITIAL_VLS", index=False, header=False, sep=";")

    # attributes file
    # needs the comma as separator since some values contain the semicolon
    attributes = pd.DataFrame(data[LABELS])
    attributes.to_csv(path_or_buf="work/ATTRIBUTES", index=False, header=False, sep=",")

    # create the network to extract the edges between nodes
    graph = create_network(data, similarity_threshold=0.7, name="Travel Survey gower similarity network")
    edges = pd.DataFrame(graph.edges)
    edges["weight"] = [float(data['weight']) for u, v, data in graph.edges(data=True) if 'weight' in data]
    edges.to_csv(path_or_buf="work/EDGES", index=False, header=False, sep=" ")


def generate_binary_pairs(size, percentage_of_ones):
    num_ones = int(size * (percentage_of_ones / 100))

    # Create an array with the specified number of ones and the remaining zeros
    binary_array = np.array([1] * num_ones + [0] * (size - num_ones))

    # Shuffle the array to randomize the order of 1s and 0s
    np.random.shuffle(binary_array)

    return binary_array


def generate_labels(size):
    return [f'L{i}' for i in range(size)]
