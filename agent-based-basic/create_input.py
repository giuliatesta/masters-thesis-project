import json

import numpy as np
import pandas as pd

from network_building import create_network

NUMBER_OF_RECORDS = 1000
INITIAL_ADAPTERS_PERC = 40


def create_input(data, LABELS):
    data.reset_index(drop=True, inplace=True)
    data = data.head(NUMBER_OF_RECORDS)

    # attributes file
    # needs the comma as separator since some values contain the semicolon
    attributes = transform_categorical_values(pd.DataFrame(data[LABELS]))
    # with headers othrwise we lose the name of the attributes
    attributes.to_csv(path_or_buf="work/ATTRIBUTES", index=False, sep=",")
    indexes_DNA = attributes.iloc[:, 0]

    # labels file only has L0, L1, L2
    with open("work/LABELS", 'w') as f:
        f.write("L0;L1")

    # create the network to extract the edges between nodes
    graph = create_network(data, similarity_threshold=0.5, name="Travel Survey gower similarity network")
    edges = pd.DataFrame(graph.edges)
    edges["weight"] = [float(data['weight']) for u, v, data in graph.edges(data=True) if 'weight' in data]
    edges.to_csv(path_or_buf="work/EDGES", index=False, header=False, sep=" ")

    nodes = pd.concat([edges[0], edges[1]]).unique()

    # initial vector labels
    initial_vls = pd.DataFrame(generate_initial_vls_with_index(nodes, indexes_DNA, INITIAL_ADAPTERS_PERC), dtype=float)
    initial_vls.to_csv(path_or_buf="work/INITIAL_VLS", index=False, header=False, sep=";")


def generate_binary_pairs(size, percentage_of_ones):
    num_ones = int(size * (percentage_of_ones / 100))

    # Create an array with the specified number of ones and the remaining zeros
    binary_array = np.array([1] * num_ones + [0] * (size - num_ones))

    # Shuffle the array to randomize the order of 1s and 0s
    np.random.shuffle(binary_array)

    return binary_array


# the initial adapters are chosen from the list of nodes that will be actually present in the network
# if considering all (from 0 to 1000), there is the risk of making adpters nodes that will not be present in the LPNet
# which is built given the edges (meaning that nodes without edges are not considered in LPNet)
# the choice is made between 654 nodes with non-consecutive indices (that's why the DNAs needs to be filtered)

def generate_initial_vls_with_index(nodes, values, percentage_of_adapters):
    if percentage_of_adapters > 1:
        percentage_of_adapters = percentage_of_adapters / 100

    filtered_values = {i: values[i] for i in nodes}  # contains the values the nodes with their original index

    # compute the average index DNA value
    avg_DNA = sum([float(i) for i in filtered_values.values()]) / len(filtered_values)
    vls = np.array([[1.0, 0.0] for _ in range(len(values))])
    # the initial vls are len of values (1000) - assignment of adapters easier
    # no problems related to indexes out of bounds

    print(f"Average index DNA: {avg_DNA}")
    possible_adapters = []

    # find the indexes that are greater than the average
    for i, value in filtered_values.items():
        if float(value) >= avg_DNA:
            possible_adapters.append(i)

    # out of the possible adapters, extract the percentage_of_adapters % to be adapters
    possible_adapters_length = len(possible_adapters)
    number_of_initial_adapters = int(possible_adapters_length * percentage_of_adapters)

    # randomly choose indices to set to 1
    indices = np.random.choice(possible_adapters, number_of_initial_adapters, replace=False)
    print(f"Initial adapters indices: {indices}")
    # set the chosen indices to 1
    for index in indices:
        vls[index] = [0.0, 1.0]
    print(f"Initial adapter/non-adapter ratio: {len(indices)}/{len(filtered_values)}")
    return vls


def transform_categorical_values(df):
    df.convert_dtypes()
    mask = df.apply(lambda x: x.str.len() > 10)

    # the numerical attributes are cast into numbers
    # the string attributes are transformed if too long
    legend_json = {}
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            if df[column].apply(lambda x: isinstance(x, str) and len(x) > 10).any():
                transformed_column, legend = transform_long_strings(df[column])
                legend_json.update({column: legend})
                df[column] = transformed_column
    with open("./work/ATTRIBUTE_LEGEND", 'w') as json_file:
        json.dump(legend_json, json_file, indent=4)
    return df


def transform_long_strings(column):
    df = pd.DataFrame(column)
    mapping = {}
    for value in column:
        if value not in mapping.values():
            # add to the mapping the new association
            key = len(mapping) + 1
            mapping[key] = value
            # replace the string with its corresponding integer
            df[column == value] = key
    return df, mapping
