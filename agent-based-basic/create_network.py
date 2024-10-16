import itertools
import os
import string

import matplotlib.pyplot as plt
from datetime import date

import numpy as np
import pandas as pd
from networkx import Graph, draw, number_connected_components, draw_networkx_nodes, draw_networkx_edges, \
    draw_networkx_labels, spring_layout
from pandas import DataFrame
from gower import gower_matrix

from preprocessing import remove_all_except_for, remove_column
from sklearn.metrics.pairwise import cosine_similarity

# creates the network, adds the nodes and the edges based on the similarity value between the nodes is bigger then threshold
def init_network(df: DataFrame, similarity_threshold: float, name="Network", no_logs=False):
    if not no_logs:
        print("Creating the network...")
    # creates the graph
    graph = Graph(name=name)

    # add all the nodes
    # sort the dataframe by index to ensure consistent iteration order
    nodes_counter = 0
    for index, row in df.iterrows():
        graph.add_node(index, **row.to_dict())
        nodes_counter += 1
    if not no_logs:
        print(f"Added {nodes_counter} nodes")

    similarities = get_similarities(df, "gower")
    # distribution_over_nodes_count(similarities, title=name, x_label="gower's similarity values")
    counter = 0
    for i, j in itertools.combinations(df.index, 2):  # creates all combinations of possible edges
        similarity = similarities[i, j]
        if similarity >= similarity_threshold:
            graph.add_edge(i, j, weight=similarity)
            counter += 1
    if not no_logs:
        print(f"Added {counter} edges")
        print("Done.")
    return graph


# returns multiple similarity measurements: Gower's distance, NumPy, SciKit, or "homemade" Cosine distance
def get_similarities(df: DataFrame, similarity_metric: string):
    # computes the gower's distance
    # a smaller distance value indicates higher similarities between data points
    if similarity_metric == "gower":
        return 1.0 - gower_matrix(df)

    # uses the numpy function dot to compute the cosine function
    if similarity_metric == "numpy-cosine":
        def numpy_cosine_formula(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        return cosine_similarity_matrix(one_hot_encoding(df), numpy_cosine_formula)

    # uses the ready-to-use scikit-learn cosine similarity function
    if similarity_metric == "scikit-learn-cosine":
        return cosine_similarity(one_hot_encoding(df))

    # computes manually the cosine function
    if similarity_metric == "cosine":
        def homemade_cosine_formula(a, b):
            dot_product = sum(a1 * b2 for a1, b2 in zip(a, b))
            norm_a = np.sqrt(sum(pow(val, 2) for val in a))
            norm_b = np.sqrt(sum(pow(val, 2) for val in b))
            return dot_product / (norm_a * norm_b)

        return cosine_similarity_matrix(one_hot_encoding(df), homemade_cosine_formula)


def cosine_similarity_matrix(encoded_df: DataFrame, cosine_function):
    size = len(encoded_df)
    similarity_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            similarity_matrix[i, j] = cosine_function(encoded_df.iloc[i], encoded_df.iloc[j])
    return similarity_matrix


def one_hot_encoding(df: DataFrame):
    return pd.get_dummies(df)

# draws the network, with its nodes and edges
def plot_network(graph: Graph, file_path=""):
    plt.figure(figsize=(20, 15))
    plt.title(label=graph.name)
    pos = spring_layout(graph)
    labels = {node: node for node in graph.nodes()}
    draw_networkx_nodes(graph, pos=pos, node_size=80)
    draw_networkx_edges(graph, pos=pos, width=0.5)
    draw_networkx_labels(graph, pos=pos, font_size=8, labels=labels)
    draw(graph, with_labels=True, alpha=0.75)
    if file_path == "":
        file_path = get_file_path(graph_name=graph.name)
    print(f"Saving in {file_path}")

    plt.legend([
                f'Connected components: {number_connected_components(graph)}',
                f'Nodes: {graph.number_of_nodes()}, '
                f'Edges: {graph.number_of_edges()}'],
               loc='upper right')
    plt.savefig(file_path, dpi= 800)
    plt.close()
    print("Done.")


def get_file_path(graph_name):
    dir_name = f"./plots/{date.today().strftime('%Y%m%d')}"
    if not os.path.exists(dir_name):
        print(f"Creating directory {dir_name}")
        os.makedirs(dir_name)

    return f"{dir_name}/{graph_name}.png"


# for creating the networks with only one column at the time.
def create_networks_using_one_column(df: DataFrame, threshold: float):
    data = df
    for column in data.columns:
        removed_data = remove_all_except_for(data, [column])
        graph = init_network(removed_data, similarity_threshold=threshold,
                             name=f"Network with only {column}", no_logs=True)
        plot_network(graph)
        data = df


# for creating the networks removing one column at the time
def create_networks_removing_one_column(df: DataFrame, threshold: float):
    data = df
    for column in data.columns:
        removed_data = remove_column(data, column_name=column)
        graph = init_network(removed_data, similarity_threshold=threshold,
                             name=f"Network with only {column}")
        plot_network(graph)
        data = df
