import itertools
import os

import matplotlib.pyplot as plt
from datetime import date

from networkx import Graph, draw, get_edge_attributes, draw_networkx_edge_labels, spring_layout
from pandas import DataFrame
from gower import gower_matrix

# a threshold is required to evaluate whether create the edge
# based on the gower's distance of two nodes
DISTANCE_THRESHOLD = 0.75


def create_network(df: DataFrame, name="Network"):
    print("Creating the network...")
    # creates the graph
    graph = Graph(name=name)
    # add all the nodes
    for index, row in df.iterrows():
        graph.add_node(index, attr_dict=row.to_dict())
    print(f"Added {len(df)} nodes")
    # computes the Gower's distance for creating the edges in the network
    # a smaller distance value indicates higher similarity between data points
    gower_distance = 1 - gower_matrix(df)
    counter = 0
    for edge in itertools.combinations(range(len(df)), 2):  # creates all combinations of possible edges
        i = edge[0]
        j = edge[1]
        distance = gower_distance[i, j]
        if distance >= DISTANCE_THRESHOLD:
            graph.add_edge(i, j, weight=distance)
            counter = counter + 1
    print(f"Added {counter} edges")
    print("Done.")
    return graph


def plot_network(graph: Graph, file_path=""):
    print("Plotting the network...")
    plt.figure(figsize=(20, 15))
    pos = spring_layout(graph)
    draw(graph, pos, node_size=50, node_color="#f4c2c2", alpha=0.75)
    draw_networkx_edge_labels(graph, pos, edge_labels=get_edge_attributes(graph, 'weight'), label_pos=0.5, font_size=5)
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

    return f"{dir_name}/{graph_name}-{DISTANCE_THRESHOLD}.png"