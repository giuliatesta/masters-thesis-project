import itertools
import matplotlib.pyplot as plt
from datetime import date

from networkx import Graph, draw, get_edge_attributes, draw_networkx_edge_labels,     random_layout
from pandas import DataFrame
from gower import gower_matrix

# a threshold is required to evaluate whether create the edge
# based on the gower's distance of two nodes
DISTANCE_THRESHOLD = 0.95


def create_network(df: DataFrame, name="Network"):
    print("Creating the network...")
    # creates the graph
    graph = Graph(name=name)
    # add all the nodes
    for index, row in df.iterrows():
        graph.add_node(index, attr_dict=row.to_dict())

    # computes the Gower's distance for creating the edges in the network
    # a smaller distance value indicates higher similarity between data points
    gower_distance = 1 - gower_matrix(df)
    for edge in itertools.combinations(range(len(df)), 2):  # creates all combinations of possible edges
        i = edge[0]
        j = edge[1]
        distance = gower_distance[i, j]
        if distance >= DISTANCE_THRESHOLD:
            graph.add_edge(i, j, weight=distance)
    print("Done.")
    return graph


def plot_network(graph: Graph):
    print("Plotting the network...")
    plt.figure()
    pos = random_layout(graph)
    draw(graph, pos, with_labels=True, node_size=200, node_color="skyblue", font_size=8)
    # draw_networkx_edge_labels(graph, pos, edge_labels=get_edge_attributes(graph, 'weight'))
    plt.title(graph.name)
    plt.savefig(f"./plots/{graph.name}-{DISTANCE_THRESHOLD}-{date.today().strftime('%Y%m%d')}.png")
    print("Done.")

