import json

import numpy as np
import pandas as pd

from network_building import create_network

NUMBER_OF_RECORDS = 1000
INITIAL_ADAPTERS_PERC = 100


def create_input(data, LABELS):
    data.reset_index(drop=True, inplace=True)
    data = data.head(NUMBER_OF_RECORDS)

    # attributes file
    # needs the comma as separator since some values contain the semicolon
    attributes = transform_categorical_values(pd.DataFrame(data[LABELS]))
    attributes.to_csv(path_or_buf="work/ATTRIBUTES", index=False, header=False, sep=",")
    indexes_DNA = attributes.iloc[:, 0]

    # initial vector labels
    initial_vls = pd.DataFrame(generate_initial_vls_with_index(indexes_DNA, INITIAL_ADAPTERS_PERC), )
    initial_vls.to_csv(path_or_buf="work/INITIAL_VLS", index=False, header=False, sep=";")

    # labels file only has L0, L1, L2
    with open("work/LABELS", 'w') as f:
        f.write("L0;L1")

    # create the network to extract the edges between nodes
    graph = create_network(data, similarity_threshold=0.6, name="Travel Survey gower similarity network")
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


def generate_initial_vls_with_index(indexes, percentage_of_adapters):
    if percentage_of_adapters > 1:
        percentage_of_adapters = percentage_of_adapters / 100

    indexes_length = len(indexes)
    # compute the average index DNA value
    avg_DNA = sum([float(i) for i in indexes]) / indexes_length
    vls = np.array([[1.0, 0.0] for _ in range(indexes_length)])
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
    for index in indices:
        vls[index] = [0.0, 1.0]
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

