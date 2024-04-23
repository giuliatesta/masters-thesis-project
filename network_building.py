import itertools
import os

import matplotlib.pyplot as plt
from datetime import date

from networkx import Graph, draw, get_edge_attributes, draw_networkx_edge_labels, spring_layout
from pandas import DataFrame
from gower import gower_matrix

from statistics import gower_matrix_distribution

# a threshold is required to evaluate whether create the edge
# based on the gower's distance of two nodes
SIMILARITY_THRESHOLD = 0.75



def create_network(df: DataFrame, name="Network"):
    print("Creating the network...")
    # creates the graph
    graph = Graph(name=name)
    # add all the nodes
    for index, row in df.iterrows():
        graph.add_node(index, attr_dict=row.to_dict())
    print(f"Added {len(df)} nodes")
    # computes the Gower's distance for creating the edges in the network
    # a smaller distance value indicates higher similarities between data points
    similarities = 1 - gower_matrix(df)
    gower_matrix_distribution(similarities)
    counter = 0
    for edge in itertools.combinations(range(len(df)), 2):  # creates all combinations of possible edges
        i = edge[0]
        j = edge[1]
        similarity = similarities[i, j]
        if similarity >= SIMILARITY_THRESHOLD:
            graph.add_edge(i, j, weight=similarity)
            counter = counter + 1
    print(f"Added {counter} edges")
    print("Done.")
    return graph


def plot_network(graph: Graph, file_path=""):
    print("Plotting the network...")
    plt.figure(figsize=(20, 15))
    pos = spring_layout(graph)
    # draw(graph, pos, node_size=50, node_color="#f4c2c2", alpha=0.75)
    # draw_networkx_labels(graph, pos, labels=graph.nodes, font_size=5)
    draw(graph, with_labels=True)
    plt.title(graph.name)
    if file_path == "":
        file_path = get_file_path(graph_name=graph.name)
    print(f"Saving in {file_path}")
    plt.savefig(file_path)
    print("Done.")


def get_file_path(graph_name):
    dir_name = f"./plots/{date.today().strftime('%Y%m%d')}"
    if not os.path.exists(dir_name):
        print(f"Creating directory {dir_name}")
        os.makedirs(dir_name)

    return f"{dir_name}/{graph_name}-{SIMILARITY_THRESHOLD}.png"