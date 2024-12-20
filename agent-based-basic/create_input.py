import json
from math import ceil

import numpy as np
import pandas as pd

from create_network import init_network

NUMBER_OF_RECORDS = 1000


# builds the initial files used as input for the simulations
# - ATTRIBUTES: contains the attributes of each node (usually, gender, age, education level,
# sharing DNA and whether they would subscribe to a car sharing service if available)
# - EDGES: contains the pairs of nodes that have an edge in the network
# - INITIAL_VLS: contains the initial values of the vector labels of the nodes in the network
# - LABELS: contains only the labels names
def create_input_files(data, LABELS, similarity_threshold=0.5, initialisation="", perc_of_adapters=90):
    data.reset_index(drop=True, inplace=True)
    # keeps only the first NUMBER_OF_RECORDS rows
    data = data.head(NUMBER_OF_RECORDS)

    # attributes file
    # needs the comma as separator since some values contain the semicolon
    attributes = transform_categorical_values(pd.DataFrame(data[LABELS]))
    # with headers otherwise we lose the name of the attributes
    attributes.to_csv(path_or_buf="work/ATTRIBUTES", index=False, sep=",")

    # labels file only has L0 and L1
    with open("work/LABELS", 'w') as f:
        f.write("L0;L1")
        f.close()

    # create the network to extract the edges between nodes
    graph = init_network(data, similarity_threshold=similarity_threshold,
                         name="Travel Survey similarity network")
    edges = pd.DataFrame(graph.edges)
    edges["weight"] = [float(data['weight']) for u, v, data in graph.edges(data=True) if 'weight' in data]
    edges.to_csv(path_or_buf="work/EDGES", index=False, header=False, sep=" ")

    # connected_components_over_threshold(data, "Number of connected components vs. Threshold")
    potentially_adapter_ids = [node for node in graph.nodes() if
                               graph.nodes()[node]["Would_subscribe_car_sharing_if_available_new"] == 2]
    print(
        f"Nodes indices of agent that answered 'Yes' to Would_subscribe_car_sharing_if_available ({len(potentially_adapter_ids)}): ")
    print(sorted(potentially_adapter_ids))
    # initial vector labels
    vls = []
    if initialisation == "adapters-with-SI":
        indexes_DNA = attributes.iloc[:, 0]
        vls = generate_initial_vls_with_index(graph.nodes(), indexes_DNA, perc_of_adapters)
    if initialisation == "random-adapters":
        vls = generate_initial_vls_randomly(graph.nodes(), perc_of_adapters)
    if initialisation == "would-subscribe-attribute":
        would_subscribe_car_sharing = attributes.iloc[:, -1]
        vls = generate_vector_labels_based_on_attribute(graph.nodes(), would_subscribe_car_sharing, 100)
    initial_vls = pd.DataFrame(vls, dtype=float)
    initial_vls.to_csv(path_or_buf="work/INITIAL_VLS", index=False, header=False, sep=";")
    return graph


# the initial adapters are chosen based on the value of the column "Would_subscribe_car_sharing_if_available" column
# if 2: Yes, instead of purchasing a new car;
#       Yes without any influence on my car ownership;
#       Yes and I would give up one car I currently own
# if 1: Maybe yes, maybe not. I would need to test the service before taking a decision
# if 0: No, I would not be interested in this service
def generate_vector_labels_based_on_attribute(nodes, attribute_values, percentage_of_adapters=1):
    possible_adapters = [node for node in nodes if
                         nodes[node]["Would_subscribe_car_sharing_if_available_new"] == 2]
    vls = np.array([[1.0, 0.0] for _ in range(len(attribute_values))])
    possible_adapters_length = len(possible_adapters)
    number_of_initial_adapters = ceil(possible_adapters_length * (percentage_of_adapters / 100))
    indices = np.random.choice(possible_adapters, number_of_initial_adapters, replace=False)
    print(f"Indices of {percentage_of_adapters}% of agents that have been initialised as adapters:\n{sorted(indices)}")
    for index in indices:
        vls[index] = [0.0, 1.0]
    return vls


# the initial adapters are chosen from the list of nodes that will be actually present in the network
# if considering all (from 0 to 1000), there is the risk of making adapters nodes that will not be present in the LPNet
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
    print(possible_adapters_length)
    # randomly choose indices to set to 1
    indices = np.random.choice(possible_adapters, number_of_initial_adapters, replace=False)
    print(f"Initial adapters indices: {sorted(indices)}")
    # set the chosen indices to 1
    for index in indices:
        vls[index] = [0.0, 1.0]
    print(f"Initial adapters: {len(indices)}")
    return vls


def generate_initial_vls_randomly(nodes, percentage_of_adapters):
    if percentage_of_adapters > 1:
        percentage_of_adapters = percentage_of_adapters / 100

    vls = np.array([[1.0, 0.0] for _ in range(len(nodes))])
    number_of_initial_adapters = int(len(nodes) * percentage_of_adapters)
    # randomly choose indices to set to 1
    indices = np.random.choice(nodes, number_of_initial_adapters, replace=False)
    print(f"Initial adapters indices: {sorted(indices)}")
    # set the chosen indices to 1
    for index in indices:
        vls[index] = [0.0, 1.0]
    print(f"Initial adapters: {len(indices)}")
    return vls


def transform_categorical_values(df):
    df.convert_dtypes()
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
