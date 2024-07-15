import numpy as np
import pandas as pd

from network_building import create_network

NUMBER_OF_RECORDS = 1000
INITIAL_ADAPTERS_PERC = 10


def create_input(data, LABELS):
    data.reset_index(drop=True, inplace=True)
    data = data.head(NUMBER_OF_RECORDS)

    # labels file only has L0, L1, L2
    labels = pd.DataFrame(generate_labels(len(LABELS)))
    labels.to_csv(path_or_buf="work/LABELS", index=False, header=False, sep=";")

    # attributes file
    # needs the comma as separator since some values contain the semicolon
    attributes = pd.DataFrame(data[LABELS])
    attributes.to_csv(path_or_buf="work/ATTRIBUTES", index=False, header=False, sep=",")
    indexes_DNA = attributes.iloc[:, 0]

    # initial vector labels
    initial_vls = pd.DataFrame(generate_initial_vls_with_index(indexes_DNA, INITIAL_ADAPTERS_PERC), )
    initial_vls.to_csv(path_or_buf="work/INITIAL_VLS", index=False, header=False, sep=";")


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


def generate_initial_vls_with_index(indexes, percentage_of_adapters):
    if percentage_of_adapters > 1:
        percentage_of_adapters = percentage_of_adapters/100

    indexes_length = len(indexes)
    # compute the average index DNA value
    avg_DNA = sum([float(i) for i in indexes]) / indexes_length
    vls = np.zeros(indexes_length, dtype=int)
    possible_adapters_index = []

    # find the indexes that are greater than the average
    for i, value in indexes.items():
        if float(value) >= avg_DNA:
            possible_adapters_index.append(i)

    # out of the possible adapters, extract the percentage_of_adapters % to be adapters
    possible_adapters_length = len(possible_adapters_index)
    number_of_initial_adapters = int(possible_adapters_length * percentage_of_adapters)

    # randomly choose indices to set to 1
    indices = np.random.choice(possible_adapters_length, number_of_initial_adapters)

    # set the chosen indices to 1
    vls[indices] = 1
    return vls
