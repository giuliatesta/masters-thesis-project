import numpy as np
from matplotlib import pyplot as plt
from networkx import number_connected_components
from pandas import DataFrame

def distribution_over_nodes_count(matrix, title="", x_label="", bins=20):
    print(f"Plotting the distribution {title}")
    plt.figure()
    plt.hist(matrix, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel("Number of nodes")
    new_title = f"Distribution {title}"
    plt.title(new_title)
    plt.grid(True)
    plt.savefig(f"./plots/statistics/{new_title}-{bins}bins.png")
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

    plt.plot(thresholds, components, marker="o")
    for i, _ in enumerate(components):
        print(f"({thresholds[i]}, {components[i]}")
        plt.text(thresholds[i], components[i], f"({thresholds[i]:.2f}, {components[i]})", fontsize=6, ha="center")
    plt.xlabel("Threshold")
    plt.ylabel("Number of components")
    plt.title("Number of connected components vs. Threshold")
    plt.grid(True)
    plt.savefig(f"./plots/statistics/{title}.png")
    print("Done.")
