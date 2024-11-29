from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as c
from networkx import number_connected_components
from pandas import DataFrame
from scipy.stats import beta
import plotly.graph_objects as go


# plots the distribution of the similarity metrix over the nodes of the network
def similarity_matrix_distribution_over_nodes(similarity_matrix, title="", x_label="", bins_count=20):
    print(f"Plotting the distribution {title}")
    plt.figure()
    cmap = plt.get_cmap('plasma', 1000)
    custom_colors = [c.rgb2hex(cmap(i)) for i in range(cmap.N)]
    plt.hist(similarity_matrix, bins=bins_count, color=custom_colors)
    plt.xlabel(x_label)
    plt.ylabel("Number of nodes")
    new_title = f"Distribution {title}"
    plt.title(new_title)
    plt.grid(True)
    plt.savefig(f"../plots/statistics/{new_title}-{bins_count}-bins.png", dpi=1200)
    plt.close()
    print("Done.")


def format_double(d):
    return "{:.2f}".format(d)


# plots the number of connected components in th network as the similarity threshold increases from 0 to 1.0
def connected_components_over_threshold(df: DataFrame, title=""):
    from create_network import init_network
    print("Plotting number of components over the threshold values...")
    components = []
    thresholds = np.arange(0, 1.0, 0.05)
    for threshold in thresholds:
        graph = init_network(df, similarity_threshold=threshold, no_logs=True,
                             name=f"{title}-{format_double(threshold)}")
        n = number_connected_components(graph)
        components.append(n)

    plt.figure()
    plt.plot(thresholds, components, marker="o", color="mediumorchid")
    for i, _ in enumerate(components):
        print(f"({thresholds[i]}, {components[i]}")
        plt.text(float(thresholds[i]), components[i], f"({thresholds[i]:.2f}, {components[i]})", fontsize=5,
                 ha="center")
    plt.xlabel("Threshold")
    plt.ylabel("Number of components")
    plt.title("Number of connected components vs. Threshold")
    plt.grid(True)
    plt.savefig("../plots/statistics/gower_components_over_threshold.png", dpi=1200)
    print("Done.")


# plots the frequency of the degrees in the network
# (how many nodes have a specific degree)
def nodes_by_degree_distribution(graph, title, file_name):
    degree_count = Counter(degree for _, degree in graph.degree())
    sorted_degrees = sorted(degree_count.items())
    degrees, counts = zip(*sorted_degrees)
    plt.figure()
    df = pd.DataFrame()

    df["Degrees"] = degrees
    df["Counts"] = counts
    df.to_csv(path_or_buf="./degrees.cvs", index=False)

    plt.bar(degrees, counts)
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title(title)
    plt.savefig(f"./network_analysis/degrees/{file_name}.png", dpi=1000)


def plot_beta_distributions():
    # Define the x range (0 to 1) for the beta distributions
    x = np.linspace(0, 1, 100)

    # Define the three sets of alpha and beta parameters
    params = [
        (2, 2, "alpha=2, beta=2"),
        (2, 5, "alpha=2, beta=5"),
        (5, 2, "alpha=5, beta=2")
    ]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each beta distribution
    for alpha, beta_val, label in params:
        y = beta.pdf(x, alpha, beta_val)
        plt.plot(x, y, label=label)

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid()
    plt.title('Beta Distributions')
    plt.legend()

    # Show the plot
    plt.savefig("../plots/beta_distribution.png", dpi=1200)


def plotly_network_plot(graph):
    fig = go.Figure()

    # Define layout
    pos = nx.spring_layout(graph)

    # Add nodes
    node_x = [pos[node][0] for node in graph.nodes()]
    node_y = [pos[node][1] for node in graph.nodes()]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='blue')))

    # Add edges
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='black')))

    # Show interactive plot
    fig.show()

def plot_step_function():
    x = np.linspace(0, 1, 1000)
    y = np.where(x < 0.5, 0, 1)
    plt.step(x, y, where='post', color='blue', label='Confirmation Bias')

    plt.step(x, np.where(x > 0.5, 1, np.nan), where='post', color='red', label='Availability Bias')

    plt.title("Bias application based on Majority of Neighbours being Adapters")
    plt.xlabel("Percentage of Adapter Neighbours")
    plt.ylabel("Bias Applied")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.xticks(np.linspace(0, 1, 11), labels=[f"{int(i * 100)}%" for i in np.linspace(0, 1, 11)])
    plt.legend()

    # Show the plot
    #plt.show()
    plt.savefig("../plots/bias_application_step_function.png", dpi=1200)
