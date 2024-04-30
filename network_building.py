import itertools
import os

import matplotlib.pyplot as plt
from datetime import date

from networkx import Graph, draw
from pandas import DataFrame
from gower import gower_matrix

from statistics import gower_matrix_distribution


def create_network(df: DataFrame, similarity_threshold: float, name="Network", no_logs=False):
    if not no_logs:
        print("Creating the network...")
    # creates the graph
    graph = Graph(name=name)
    # add all the nodes
    print(f"data: {len(df)}")
    for index, row in df.iterrows():
        graph.add_node(index, attr_dict=row.to_dict())
    if not no_logs:
        print(f"Added {len(df)} nodes")
    # computes the Gower's distance for creating the edges in the network
    # a smaller distance value indicates higher similarities between data points
    similarities = 1.0 - gower_matrix(df)
    # gower_matrix_distribution(similarities)
    counter = 0
    for edge in itertools.combinations(range(len(df)), 2):  # creates all combinations of possible edges
        i = edge[0]
        j = edge[1]
        similarity = similarities[i, j]
        if similarity >= similarity_threshold:
            graph.add_edge(i, j, weight=similarity)
            counter = counter + 1
    if not no_logs:
        print(f"Added {counter} edges")
        print("Done.")
    return graph


def plot_network(graph: Graph, file_path=""):
    print(f"Plotting the network {graph.name}")
    plt.figure(figsize=(20, 15))
    plt.title(label=graph.name)
    draw(graph, with_labels=True, alpha=0.75)
    if file_path == "":
        file_path = get_file_path(graph_name=graph.name)
    print(f"Saving in {file_path}")
    plt.savefig(file_path)
    plt.close()
    print("Done.")


def get_file_path(graph_name):
    dir_name = f"./plots/{date.today().strftime('%Y%m%d')}"
    if not os.path.exists(dir_name):
        print(f"Creating directory {dir_name}")
        os.makedirs(dir_name)

    return f"{dir_name}/{graph_name}.png"
