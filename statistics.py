import numpy as np
from matplotlib import pyplot as plt
from networkx import number_connected_components
from pandas import DataFrame


def gower_matrix_distribution(matrix, title="Distribution of Gower's Matrix Values"):
    print("Plotting the gower's matrix distribution...")
    plt.figure()
    plt.hist(matrix, bins=20)
    plt.xlabel("Gower's matrix values")
    plt.ylabel('Number of nodes')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"./plots/statistics/{title}.png")
    plt.close()
    print("Done.")


def components_over_threshold(df: DataFrame):
    from network_building import create_network
    print("Plotting number of components over the threshold values...")
    components = []
    thresholds = np.arange(0, 1.0, 0.05)
    for threshold in thresholds:
        graph = create_network(df, similarity_threshold=threshold, no_logs=True)
        components.append(number_connected_components(graph))

    plt.figure()
    plt.plot(thresholds, components, marker='o')
    for i, _ in enumerate(components):
        plt.text(thresholds[i], components[i], f"({thresholds[i]:.2f}, {components[i]})", fontsize=6, ha='center')
    plt.xlabel('Threshold')
    plt.ylabel('Number of components')
    plt.title('Number of connected components vs. Threshold')
    plt.grid(True)
    plt.savefig("./plots/statistics/components_over_threshold.png")
    print("Done.")
