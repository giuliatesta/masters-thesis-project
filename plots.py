from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from networkx import number_connected_components
from pandas import DataFrame


def distribution_over_nodes_count(matrix, title="", x_label="", bins=20):
    print(f"Plotting the distribution {title}")
    plt.figure()
    n, bins, patches = plt.hist(matrix, bins=bins)
    cmap = plt.get_cmap('plasma')
    bin_heights = (n - np.min(n)) / (np.max(n) - np.min(n))
    for patch, color_value in zip(patches, bin_heights):
        patch.set_facecolor(cmap(color_value))

    plt.xlabel(x_label)
    plt.ylabel("Number of nodes")
    new_title = f"Distribution {title}"
    plt.title(new_title)
    plt.grid(True)
    plt.savefig(f"./plots/statistics/{new_title}-{bins}bins.png", dpi=1200)
    plt.close()
    print("Done.")


def format_double(d):
    return "{:.2f}".format(d)


def components_over_threshold(df: DataFrame, title=""):
    from network_building import create_network, plot_network
    print("Plotting number of components over the threshold values...")
    components = []
    thresholds = np.arange(0, 1.0, 0.05)
    for threshold in thresholds:
        graph = create_network(df, similarity_threshold=threshold, no_logs=True,
                               name=f"{title}-{format_double(threshold)}")
        n = number_connected_components(graph)
        # plot_network(graph)
        components.append(n)

    plt.figure()
    plt.plot(thresholds, components, marker="o", color="mediumorchid")
    for i, _ in enumerate(components):
        print(f"({thresholds[i]}, {components[i]}")
        plt.text(thresholds[i], components[i], f"({thresholds[i]:.2f}, {components[i]})", fontsize=5, ha="center")
    plt.xlabel("Threshold")
    plt.ylabel("Number of components")
    plt.title("Number of connected components vs. Threshold")
    plt.grid(True)
    plt.savefig(f"./plots/statistics/{title}.png", dpi=1200)
    print("Done.")


# the number of nodes that have a specific degree...
def nodes_by_degree_distribution(graph):
    #node_degree_list = [(1, 3), (2, 2), (3, 3), (4, 1), (5, 2)]
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
    plt.title('Degree Distribution of Nodes (similarity threshold = 0.35)')
    # sns.barplot(x=degrees, y=counts, legend=True, palette="plasma")
    # plt.xlabel('degree')
    # plt.ylabel('Number of Nodes')
    # plt.title('Number of Nodes by Degree')
    plt.savefig("/Users/giuliatesta/PycharmProjects/masters-thesis-project/plots/statistics/nodes_by_degree_35.png", dpi=1000)

