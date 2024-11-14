import itertools
import os

import matplotlib.pyplot as plt
from datetime import date

from networkx import Graph, draw, number_connected_components, draw_networkx_nodes, draw_networkx_edges, \
    draw_networkx_labels, spring_layout
from pandas import DataFrame
from gower import gower_matrix


# creates the network, adds the nodes and the edges based on the similarity value
# between the nodes is bigger then threshold
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

    similarities = get_similarities(df)
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


# returns multiple similarity measurements: Gower's distance
def get_similarities(df: DataFrame):
    # computes the Gower's distance
    # a smaller distance value indicates higher similarities between data points
    return 1.0 - gower_matrix(df)


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
    plt.savefig(file_path, dpi=800)
    plt.close()
    print("Done.")


def get_file_path(graph_name):
    dir_name = f"./plots/{date.today().strftime('%Y%m%d')}"
    if not os.path.exists(dir_name):
        print(f"Creating directory {dir_name}")
        os.makedirs(dir_name)

    return f"{dir_name}/{graph_name}.png"
