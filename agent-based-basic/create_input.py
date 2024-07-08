import numpy as np
import pandas as pd

from network_building import create_network
from preprocessing import load_dataset_csv, filter_by

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
    attributes = pd.DataFrame(data[LABELS])
    attributes.to_csv(path_or_buf="work/ATTRIBUTES", index=False, header=False, sep=";")

    graph = create_network(data, similarity_threshold=0.7, name="Travel Survey gower similarity network")
    edges = pd.DataFrame(graph.edges)
    edges.to_csv(path_or_buf="work/EDGES", index=False, header=False, sep=" ")


def generate_binary_pairs(size, percentage_of_01):
    """
    Generates a 2D array of binary pairs (00 or 01) with a given percentage of 01 pairs.

    Parameters:
    - size: The number of binary pairs.
    - percentage_of_01: The percentage of 01 pairs in the array (between 0 and 100).

    Returns:
    - A numpy array of binary pairs with the specified percentage of 01 pairs.
    """
    # Calculate the number of 01 pairs based on the given percentage
    num_01_pairs = int(size * (percentage_of_01 / 100))

    # Create an array with the specified number of 01 pairs and the remaining 00 pairs
    pairs = np.array([[0, 1]] * num_01_pairs + [[0, 0]] * (size - num_01_pairs))

    # Shuffle the array to randomize the order of pairs
    np.random.shuffle(pairs)

    return pairs


def generate_labels(size):
    return [f'L{i}' for i in range(size)]
